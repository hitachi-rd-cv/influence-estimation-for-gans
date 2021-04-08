# ------------------------------------------------------------------------
# A method print_tree is modified from Luigi (https://github.com/spotify/luigi)
# Copyright 2012-2019 Spotify AB
# https://github.com/spotify/luigi/blob/master/LICENSE
# ------------------------------------------------------------------------


import os
import warnings
from collections import OrderedDict

import luigi
import toml
from luigi.task import flatten
from luigi.tools.deps_tree import bcolors

from modules.utils import normalize_list_recursively


def get_input_path_recur(inputs):
    if isinstance(inputs, list):
        inputs_return = []
        for input in inputs:
            inputs_return.append(get_input_path_recur(input))
    elif isinstance(inputs, dict):
        inputs_return = {}
        for name, input in inputs.items():
            inputs_return[name] = get_input_path_recur(input)
    else:
        inputs_return = inputs.path

    return inputs_return


def get_input_path_recur_by_replacing(inputs, rename_dict):
    if isinstance(inputs, list):
        inputs_return = []
        for input in inputs:
            inputs_return.append(get_input_path_recur_by_replacing(input, rename_dict))
        return inputs_return
    else:
        input_path = inputs.path
        if input_path in rename_dict:
            return rename_dict[input_path]
        else:
            return input_path


def get_downstream_tasks_recur(task, target_query=None, query_type='family'):
    target_tasks = OrderedDict()

    if query_type == 'family':
        is_target_task = task.task_family == target_query
    elif query_type == 'id':
        is_target_task = task.task_id == target_query
    elif query_type == 'any':
        assert target_query is None
        is_target_task = True
    else:
        raise ValueError(query_type)

    if is_target_task:
        target_tasks.update({task.task_id: task})

    required_tasks_tmp = task.requires()
    if required_tasks_tmp is None:
        return target_tasks

    else:
        if isinstance(required_tasks_tmp, (list, tuple, dict)):
            if isinstance(required_tasks_tmp, dict):
                required_tasks_tmp = list(required_tasks_tmp.values())
            required_tasks = normalize_list_recursively(required_tasks_tmp)
        else:
            required_tasks = [required_tasks_tmp]

        for required_task in required_tasks:
            target_tasks_child = get_downstream_tasks_recur(required_task, target_query, query_type)
            if target_tasks_child:
                target_tasks.update(target_tasks_child)

        if target_tasks:
            target_tasks.update({task.task_id: task})
            return target_tasks
        else:
            return target_tasks


class TaskBase(luigi.Task):
    """TaskBase
    Base class inherited by most of Tasks

    Attributes:
        processed_dir: root directory of the output it is insignificant for determining the name of output directly
    """

    processed_dir = luigi.Parameter('processed', significant=False)

    def run(self, *args, **kwargs):
        '''
        do not overwrite this class in the child classes.
        This make temporary_path and pass it to the user-define run_within_temporary_path method.
        The task scripts are expected to be written in the run_within_temporary_path and any output file are expected to be placed in the temp_output_path.
        This ensures the upper stream tasks are never ran unless the down stream tasks is done.
        TaskBase (and its child class) regards the downstream task as Done one when the output directory exists.
        Using temp_output_path ensures the existance of the output directory always means that the task is successfully processed.

        Args:
            *args:
            **kwargs:

        Returns: None

        '''
        with self.output().temporary_path() as temp_output_path:
            os.makedirs(temp_output_path, exist_ok=True)
            # save parameters
            param_toml_path = os.path.join(temp_output_path, 'params.toml')
            toml.dump(self.param_kwargs, open(param_toml_path, 'w'))
            self.run_within_temporary_path(temp_output_path)

    def output(self):
        '''
        do not overwrite this class in the child classes.
        this is executed in self.run to get unique output directly.
        Each combination of the task parameter leads its unique hash contained in self.task id.
        It enables output directory to be determined automatically and ensures the same task with different parameters are never overwritten.

        Returns: luigi.LocalTarget

        '''
        return luigi.LocalTarget(os.path.join(self.processed_dir, self.task_id))

    def run_within_temporary_path(self, output_path):
        '''
        define the scripts of the tasks in child classes.

        Args:
            output_path: temporary_path which renamed after successfully finishing the Task.

        Returns: None
        '''
        raise NotImplementedError

    def get_input_dir(self):
        '''
        call self.input to get input dir path strings. when output of self.input() is nested it recursively run self.input() to get path strings.
        Returns: list of paths

        '''
        return get_input_path_recur(self.input())


class ListParameter(luigi.ListParameter):
    def parse(self, x):
        # for avoiding many escapes \ for string list on shell and adding []
        # assumes x is list of str without " and [] or list of number
        if isinstance(x, list):
            return x
        elif all([not xx.isdigit() for xx in x]):
            x = '["' + x.replace(',', '","') + '"]'
        else:
            x = '[' + x + ']'
        return super().parse(x)


def print_tree(task, indent='', last=True):
    '''
    Return a string representation of the tasks, their statuses/parameters in a dependency tree format
    '''
    # dont bother printing out warnings about tasks with no output
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Task .* without outputs has no custom complete\\(\\) method')
        is_task_complete = task.complete()
    is_complete = (bcolors.OKGREEN + 'COMPLETE' if is_task_complete else bcolors.OKBLUE + 'PENDING') + bcolors.ENDC
    name = task.task_id
    params = task.to_str_params(only_significant=True)
    result = '\n' + indent
    if (last):
        result += '└─--'
        indent += '   '
    else:
        result += '|--'
        indent += '|  '
    result += '[{0}-{1} ({2})]'.format(name, params, is_complete)
    children = flatten(task.requires())
    for index, child in enumerate(children):
        result += print_tree(child, indent, (index + 1) == len(children))
    return result
