# ------------------------------------------------------------------------
# Some methods are modified from Sacred (https://github.com/IDSIA/sacred)
# The portions of following classes and methods are Copyright (c) 2014 Klaus Greff.
# https://github.com/IDSIA/sacred/blob/master/LICENSE.txt
# ------------------------------------------------------------------------

import os
import subprocess
import sys
from collections import Mapping
from contextlib import contextmanager
from contextlib import redirect_stdout
from copy import copy
from tempfile import NamedTemporaryFile

import sacred
from sacred import commandline_options
from sacred.experiment import gather_command_line_options
from sacred.host_info import get_host_info
from sacred.initialize import get_config_modifications, get_command, get_configuration, get_scaffolding_and_config_name, \
    gather_ingredients_topological, initialize_logging, distribute_config_updates, distribute_presets, \
    create_scaffolding
from sacred.randomness import set_global_seed
from sacred.run import Run as RunBase
from sacred.settings import SETTINGS
from sacred.stdout_capturing import no_tee, tee_output_python, CapturedStdout, flush
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.utils import (
    convert_to_nested_dict,
    iterate_flattened,
    set_by_dotted_path,
    recursive_update,
    join_paths,
    SacredInterrupt
)


class Experiment(sacred.Experiment):
    def __init__(
            self,
            name=None,
            ingredients=(),
            interactive=False,
            base_dir=None,
            additional_host_info=None,
            additional_cli_options=None,
            save_git_info=False,
            use_mongo=True,
    ):
        '''
        automatically append observers and replace self.captured_out_filter with modified apply_backspaces_and_linefeeds
        to fixes the problem around stdout of tqdm progress bar

        Args:
            name:
            ingredients:
            interactive:
            base_dir:
            additional_host_info:
            additional_cli_options:
            save_git_info:
            use_mongo:
        '''
        super().__init__(name, ingredients, interactive, base_dir, additional_host_info,
                         additional_cli_options, save_git_info)
        if use_mongo:
            if 'MONGO_AUTH' in os.environ and 'MONGO_DB' in os.environ:
                url = 'mongodb://{}/?authMechanism=SCRAM-SHA-1'.format(os.environ['MONGO_AUTH'])
                self.observers.append(sacred.observers.MongoObserver(url=url, db_name=os.environ['MONGO_DB']))

        self.observers.append(sacred.observers.FileStorageObserver(os.path.join('sacred_runs', self.path)))

        # it fixes stdout problem
        self.captured_out_filter = apply_backspaces_and_linefeeds

    def deco_main(self, f):
        self.main(f)

        def wrapper(**kwargs):
            melted_kwargs = recursively_make_dict(kwargs)  # for luigi's frozen dict
            self.add_config(melted_kwargs)
            return self.run()

        return wrapper

    def _create_run(
            self,
            command_name=None,
            config_updates=None,
            named_configs=(),
            info=None,
            meta_info=None,
            options=None,
    ):
        command_name = command_name or self.default_command
        if command_name is None:
            raise RuntimeError(
                "No command found to be run. Specify a command "
                "or define a main function."
            )

        default_options = self.get_default_options()
        if options:
            default_options.update(options)
        options = default_options

        # call option hooks
        for oh in self.option_hooks:
            oh(options=options)

        run = create_run(
            self,
            command_name,
            config_updates,
            named_configs=named_configs,
            force=options.get(commandline_options.force_option.get_flag(), False),
            log_level=options.get(commandline_options.loglevel_option.get_flag(), None),
        )
        if info is not None:
            run.info.update(info)

        run.meta_info["command"] = command_name
        run.meta_info["options"] = options

        if meta_info:
            run.meta_info.update(meta_info)

        options_list = gather_command_line_options() + self.additional_cli_options
        for option in options_list:
            option_value = options.get(option.get_flag(), False)
            if option_value:
                option.apply(option_value, run)

        self.current_run = run

        return run

class Run(RunBase):
    def __call__(self, *args):
        r"""Start this run.

        Parameters
        ----------
        \*args
            parameters passed to the main function

        Returns
        -------
            the return value of the main function

        """
        if self.start_time is not None:
            raise RuntimeError(
                "A run can only be started once. "
                "(Last start was {})".format(self.start_time)
            )

        if self.unobserved:
            self.observers = []
        else:
            self.observers = sorted(self.observers, key=lambda x: -x.priority)

        self.warn_if_unobserved()
        set_global_seed(self.config["seed"])

        if self.capture_mode is None and not self.observers:
            capture_mode = "no"
        else:
            capture_mode = self.capture_mode
        capture_mode, capture_stdout = get_stdcapturer(capture_mode)
        self.run_logger.debug('Using capture mode "%s"', capture_mode)

        if self.queue_only:
            self._emit_queued()
            return
        try:
            with capture_stdout() as self._output_file:
                self._emit_started()
                self._start_heartbeat()
                self._execute_pre_run_hooks()
                self.result = self.main_function(*args)
                self._execute_post_run_hooks()
                if self.result is not None:
                    self.run_logger.info("Result: {}".format(self.result))
                elapsed_time = self._stop_time()
                self.run_logger.info("Completed after %s", elapsed_time)
                self._get_captured_output()
            self._stop_heartbeat()
            self._emit_completed(self.result)
        except (SacredInterrupt, KeyboardInterrupt) as e:
            self._stop_heartbeat()
            status = getattr(e, "STATUS", "INTERRUPTED")
            self._emit_interrupted(status)
            raise
        except BaseException:
            exc_type, exc_value, trace = sys.exc_info()
            self._stop_heartbeat()
            self._emit_failed(exc_type, exc_value, trace.tb_next)
            raise
        finally:
            self._warn_about_failed_observers()
            self._wait_for_observers()

        return self.result

    def add_artifact(self, filename, name=None, metadata=None, content_type=None):
        with redirect_stdout(open(os.devnull, 'w')):
            super().add_artifact(filename, name, metadata, content_type)


@contextmanager
def tee_output_fd():
    """Duplicate stdout and stderr to a file on the file descriptor level."""
    with NamedTemporaryFile(mode="w+", newline='') as target:
        # with NamedTemporaryFile(mode="w+", newline='') as target:
        original_stdout_fd = 1
        original_stderr_fd = 2
        target_fd = target.fileno()

        # Save a copy of the original stdout and stderr file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        try:
            # start_new_session=True to move process to a new process group
            # this is done to avoid receiving KeyboardInterrupts (see #149)
            tee_stdout = subprocess.Popen(
                ["tee", "-a", target.name],
                start_new_session=True,
                stdin=subprocess.PIPE,
                stdout=1,
            )
            tee_stderr = subprocess.Popen(
                ["tee", "-a", target.name],
                start_new_session=True,
                stdin=subprocess.PIPE,
                stdout=2,
            )
        except (FileNotFoundError, OSError, AttributeError):
            # No tee found in this operating system. Trying to use a python
            # implementation of tee. However this is slow and error-prone.
            tee_stdout = subprocess.Popen(
                [sys.executable, "-m", "sacred.pytee"],
                stdin=subprocess.PIPE,
                stderr=target_fd,
            )
            tee_stderr = subprocess.Popen(
                [sys.executable, "-m", "sacred.pytee"],
                stdin=subprocess.PIPE,
                stdout=target_fd,
            )

        flush()
        os.dup2(tee_stdout.stdin.fileno(), original_stdout_fd)
        os.dup2(tee_stderr.stdin.fileno(), original_stderr_fd)
        out = CapturedStdout(target)

        try:
            yield out  # let the caller do their printing
        finally:
            flush()

            # then redirect stdout back to the saved fd
            tee_stdout.stdin.close()
            tee_stderr.stdin.close()

            # restore original fds
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            tee_stdout.wait(timeout=1)
            tee_stderr.wait(timeout=1)

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            out.finalize()


def get_stdcapturer(mode=None):
    mode = mode if mode is not None else SETTINGS.CAPTURE_MODE
    capture_options = {"no": no_tee, "fd": tee_output_fd, "sys": tee_output_python}
    if mode not in capture_options:
        raise KeyError(
            "Unknown capture mode '{}'. Available options are {}".format(
                mode, sorted(capture_options.keys())
            )
        )
    return mode, capture_options[mode]


def recursively_make_dict(value):
    if isinstance(value, Mapping):
        return dict(((k, recursively_make_dict(v)) for k, v in value.items()))
    return value


def create_run(
        experiment,
        command_name,
        config_updates=None,
        named_configs=(),
        force=False,
        log_level=None,
):
    sorted_ingredients = gather_ingredients_topological(experiment)
    scaffolding = create_scaffolding(experiment, sorted_ingredients)
    # get all split non-empty prefixes sorted from deepest to shallowest
    prefixes = sorted(
        [s.split(".") for s in scaffolding if s != ""],
        reverse=True,
        key=lambda p: len(p),
    )

    # --------- configuration process -------------------

    # Phase 1: Config updates
    config_updates = config_updates or {}
    config_updates = convert_to_nested_dict(config_updates)
    root_logger, run_logger = initialize_logging(experiment, scaffolding, log_level)
    distribute_config_updates(prefixes, scaffolding, config_updates)

    # Phase 2: Named Configs
    for ncfg in named_configs:
        scaff, cfg_name = get_scaffolding_and_config_name(ncfg, scaffolding)
        scaff.gather_fallbacks()
        ncfg_updates = scaff.run_named_config(cfg_name)
        distribute_presets(prefixes, scaffolding, ncfg_updates)
        for ncfg_key, value in iterate_flattened(ncfg_updates):
            set_by_dotted_path(config_updates, join_paths(scaff.path, ncfg_key), value)

    distribute_config_updates(prefixes, scaffolding, config_updates)

    # Phase 3: Normal config scopes
    for scaffold in scaffolding.values():
        scaffold.gather_fallbacks()
        scaffold.set_up_config()

        # update global config
        config = get_configuration(scaffolding)
        # run config hooks
        config_hook_updates = scaffold.run_config_hooks(
            config, command_name, run_logger
        )
        recursive_update(scaffold.config, config_hook_updates)

    # Phase 4: finalize seeding
    for scaffold in reversed(list(scaffolding.values())):
        scaffold.set_up_seed()  # partially recursive

    config = get_configuration(scaffolding)
    config_modifications = get_config_modifications(scaffolding)

    # ----------------------------------------------------

    experiment_info = experiment.get_experiment_info()
    host_info = get_host_info(experiment.additional_host_info)
    main_function = get_command(scaffolding, command_name)
    pre_runs = [pr for ing in sorted_ingredients for pr in ing.pre_run_hooks]
    post_runs = [pr for ing in sorted_ingredients for pr in ing.post_run_hooks]

    run = Run(
        config,
        config_modifications,
        main_function,
        copy(experiment.observers),
        root_logger,
        run_logger,
        experiment_info,
        host_info,
        pre_runs,
        post_runs,
        experiment.captured_out_filter,
    )

    if hasattr(main_function, "unobserved"):
        run.unobserved = main_function.unobserved

    run.force = force

    for scaffold in scaffolding.values():
        scaffold.finalize_initialization(run=run)

    return run

