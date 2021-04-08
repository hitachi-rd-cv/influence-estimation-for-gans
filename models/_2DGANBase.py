import os

import numpy as np
import pandas as pd

from models import GANBase
from modules.plot import get_plot
from modules.utils import dump


class _2DGANBase(GANBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_sample_images(self, dataset, basename, batch_size=64, **kwargs):
        x, x_gen = self.sess.run([self.x, self.x_gen], dataset[:])
        path = os.path.join(self.dir, basename)
        self.gen_sample_images_from_arrays(x, x_gen, path)

    def gen_sample_images_from_arrays(self, x, x_gen, path):
        get_plot(path, [x, x_gen], ('b', 'g'))

    def dump_harmers(self, dataset, harmers_idxs, harmful_scores, **kwargs):
        is_harmful = np.isin(np.arange(dataset.sample_size), harmers_idxs)

        x = dataset[:][self.x]
        path = '{}/2c.png'.format(self.dir)
        vals = [x[~is_harmful], x[is_harmful]]
        get_plot(path, vals, ('b', 'r'), _run=self._run)
        path = '{}/grad.png'.format(self.dir)
        get_plot(path, x, harmful_scores, _run=self._run)
        path = '{}/contour.png'.format(self.dir)
        get_plot(path, np.concatenate([x, np.expand_dims(harmful_scores, 1)], axis=1), mode='contour', figsize=(15, 15),
                 _run=self._run)

        dump(harmful_scores, os.path.join(self.dir, 'harmful_scores.pkl'))
        dump(harmers_idxs, os.path.join(self.dir, 'harmers_idxs.pkl'))
        dump(x, os.path.join(self.dir, 'x.pkl'))

    @staticmethod
    def get_data_from_x(x, kind):
        return pd.DataFrame({'x': x[:, 0], 'y': x[:, 1], 'data': [kind] * len(x)})

    def dump_visual_improvements(self, params_original, params_cleansed, dataset_test):
        self.saver.restore(params_original)
        x_gen_ori = self.sess.run(self.x_gen, dataset_test[:])
        self.saver.restore(params_cleansed)
        x_gen_clean = self.sess.run(self.x_gen, dataset_test[:])

        df_ori = self.get_data_from_x(x_gen_ori, 'Generated (before cleansing)')
        df_clean = self.get_data_from_x(x_gen_clean, 'Generated (after cleansing)')

        dump(df_ori, os.path.join(self.dir, 'df_ori.pkl'))
        dump(df_clean, os.path.join(self.dir, 'df_clean.pkl'))
        # path = os.path.join(self.dir, 'test_true_kde.png')
        # self.get_plot(path, x, 'Reds', n_levels=5, figsize=(10, 10), mode='kde')
        # self.jointplot(path, x[:, 0], x[:, 1], kind='kde')
