
import sys

from luigi.cmdline import luigi_run
from luigi.cmdline_parser import CmdlineParser

from modules.myluigi import print_tree
from tasks import *

# This is expected to be run from the shell in the form of "python3 main.py {task_class_name} {parameters}"
# To understand the way to pass parameters, see https://luigi.readthedocs.io/en/stable/running_luigi.html#running-from-the-command-line
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])

    # by adding --tree arg, you can see the dependencies of the tasks and the status of Done/Pendings of them
    # when this arg set, the task will not run.
    if '--tree' in sys.argv:
        del sys.argv[sys.argv.index('--tree')]
        cmdline_args = sys.argv[1:]
        with CmdlineParser.global_instance(cmdline_args) as cp:
            print(print_tree(cp.get_task_obj()))
        sys.exit()

    # by adding `--removes {task_class_name}` it removes all the intermediate files between {TASK_NAME} and the target task you specified at the begining of the shell command.
    if '--removes' in sys.argv:
        argv_copy = sys.argv.copy()
        tree_arg_idx = sys.argv.index('--removes')
        del sys.argv[tree_arg_idx:tree_arg_idx + 2]
        argv_removal = list(('MoveOutputs', *argv_copy[tree_arg_idx:tree_arg_idx + 2]))
        with CmdlineParser.global_instance(argv_removal) as cp:
            target_task = CmdlineParser(sys.argv[1:]).get_task_obj()
            luigi.build([MoveOutputs(target_task=target_task, **cp._get_task_kwargs())], local_scheduler=True)

    sys.exit(luigi_run(sys.argv[1:]))
