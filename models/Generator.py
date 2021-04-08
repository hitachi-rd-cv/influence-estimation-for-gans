import os

from models import NNBase
from modules.plot import plot_tiled_images
from modules.utils import get_smallest_largest_val_indices, dump


class Generator(NNBase):
    name = 'generator'
    scopes = 'generator'
    scope_suffixes = '_g'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_sample_images(self, dataset, basename, batch_size=64, **kwargs):
        x = self.sess.run(self.x_gen, dataset[:batch_size])
        plot_tiled_images(x, os.path.join(self.dir, basename), _run=self._run)

    def dump_harmers(self, dataset, harmers_idxs, harmful_scores, **kwargs):
        '''
        dump instances of harmful and helpful instances
        Args:
            dataset (modules.dataset.MyDataset): training dataset
            harmers_idxs: predicted indices of harmful instances (not used in this class)
            harmful_scores: scores of harmfulness of the training instances
            **kwargs:

        Returns: None

        '''
        x = dataset[:][self.x]

        help_idxs, harm_idxs = get_smallest_largest_val_indices(harmful_scores, 64)
        help_x, harm_x = x[help_idxs], x[harm_idxs]
        help_path, harm_path = os.path.join(self.dir, 'helpful.png'), os.path.join(self.dir, 'harmful.png')

        plot_tiled_images(help_x, help_path, _run=self._run)
        plot_tiled_images(harm_x, harm_path, _run=self._run)

        dump(help_x, os.path.join(self.dir, 'help_x.pkl'))
        dump(harm_x, os.path.join(self.dir, 'harm_x.pkl'))

        y = dataset[:][self.y]
        help_y, harm_y = y[help_idxs], y[harm_idxs]

        dump(help_y, os.path.join(self.dir, 'help_y.pkl'))
        dump(harm_y, os.path.join(self.dir, 'harm_y.pkl'))

    def dump_visual_improvements(self, params_original, params_cleansed, dataset_test, batch_size=64):
        '''
        dump generated instances of before and after the data cleansing
        Args:
            params_original (list): parameters before the data cleansing
            params_cleansed (list): parameters after the data cleansing
            dataset_test (modules.dataset.MyDataset): dataset which contains test latent variables
            batch_size: batch size of the generation

        Returns: None

        '''
        self.saver.restore(params_original)
        x_gen_ori = self.sess.run(self.x_gen, dataset_test[:batch_size])
        self.saver.restore(params_cleansed)
        x_gen_cl = self.sess.run(self.x_gen, dataset_test[:batch_size])

        dump(x_gen_ori, os.path.join(self.dir, 'original.pkl'))
        dump(x_gen_cl, os.path.join(self.dir, 'cleansed.pkl'))

    def eval_improvements(self, params_original, params_cleansed, dataset_test, eval_metric, metric_kwargs):
        '''
        calculate test evaluation metric before and after the data cleansing.

        Args:
            params_original (list): parameters before the data cleansing
            params_cleansed (list): parameters after the data cleansing
            dataset_test (modules.dataset.MyDataset): dataset which contains test latent variables and maybe test instances.
            eval_metric: name of the ops or tensor of metric to be evaluated
            metric_kwargs: kwargs of eval_metric

        Returns:
            eval_metric_no_removal: test evaluation metric before the data cleansing
            eval_metric_cleansed: test evaluation metric after the data cleansing
        '''
        self.saver.restore(params_original)
        eval_metric_no_removal = self.eval_metric(eval_metric, dataset_test[:], **metric_kwargs)
        self.saver.restore(params_cleansed)
        eval_metric_cleansed = self.eval_metric(eval_metric, dataset_test[:], **metric_kwargs)

        return eval_metric_no_removal, eval_metric_cleansed
