import os
import shutil

import luigi
from luigi.util import inherits

from config import get_all_params
from modules import myluigi


@inherits(*get_all_params())
class MoveOutputs(luigi.Task):
    target_task = luigi.TaskParameter()
    removes = myluigi.ListParameter()

    def requires(self):
        import datetime
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        backup_dir = os.path.join('./processed/removed_caches/', timestamp)
        tasks = {}
        for task in self.removes:
            tasks.update(myluigi.get_downstream_tasks_recur(self.target_task, task))

        if self.target_task.task_family in self.removes:
            tasks.update({self.target_task.task_id: self.target_task})

        requires = []
        for task in tasks.values():
            if task.complete():
                task_name = task.task_family
                src = task.output().path
                requires.append(
                    MoveOutput(task_name=task_name, src=src, dst=os.path.join(backup_dir, os.path.basename(src))))

        return requires


class MoveOutput(luigi.Task):
    src = luigi.Parameter()
    dst = luigi.Parameter()
    task_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.dst)

    def run(self):
        shutil.move(self.src, self.dst)
        print('Moved output dir of "{}":{} -> {}'.format(self.task_name, self.src, self.dst))
