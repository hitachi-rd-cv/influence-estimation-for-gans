import os
import re
import shutil
from glob import glob

import numpy as np
from luigi.util import inherits
from scipy.stats import ttest_1samp

from config import RangeCleansingParameter, TotalizeValidParameter, TotalizeCleansingParameter, \
    TotalizeCleansingWrtEvalParameter
from modules import myluigi
from modules.mysacred import Experiment as SacredExperiment
from modules.utils import load, dump, parse_json
from tasks.experiment import EvalCleansing, Valid

@inherits(Valid)
class TotalizeValid(TotalizeValidParameter, myluigi.TaskBase):
    '''
    This task collect the output of Valid by changing the random seed to plot Kendal's tau and Jaccard index with the error fills.
    The output is the list of,
        - list of p values
        - list of mean values of Kendal's tau and Jaccard index
        - list of standard deviations of Kendal's tau and Jaccard index
    Their length are the same the number of the random seeds.
    '''
    def requires(self):
        requires = []
        for metric in self.metrics:
            requires_retrace_afters = []
            for retrace_after in self.retrace_afters:
                requires_seeds = []
                for seed in self.seeds:
                    requires_seeds.append(self.clone(Valid, seed=seed, retrace_after=retrace_after, metric=metric))
                requires_retrace_afters.append(requires_seeds)
            requires.append(requires_retrace_afters)

        return requires

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        self.params['input_dirs'] = self.get_input_dir()

        self.params.update({k: v for k, v in self.param_kwargs.items() if k not in self.params.keys()})
        os.makedirs(temp_output_path, exist_ok=True)
        ex = SacredExperiment(self.__class__.__name__)
        ex.deco_main(self.main)(out_dir=temp_output_path, **self.params)

    @staticmethod
    def main(_run, input_dirs, out_dir, retrace_afters, metrics, seeds, log_scale, nepoch, exts):
        vals = 'tau', 'jaccard'

        for val in vals:
            for i, metric in enumerate(metrics):
                means, stds, ps = [], [], []
                for j, retrace_after in enumerate(retrace_afters):
                    metric_vals = []
                    for k, seed in enumerate(seeds):
                        shutil.copy(os.path.join(input_dirs[i][j][k], 'lie_error.png'),
                                    os.path.join(out_dir, f'lie_error_{metric}_r{retrace_after}_s{seed}.png'))
                        metric_dic = parse_json(os.path.join(input_dirs[i][j][k], 'result.json'))
                        metric_vals.append(metric_dic[val])

                    mean = np.mean(metric_vals)
                    std = np.std(metric_vals)
                    if val == 'tau':
                        expected_random_val = 0.
                    elif val == 'jaccard':
                        expected_random_val = load(os.path.join(input_dirs[i][j][0], 'jaccard_random.pkl'))
                    else:
                        raise ValueError
                    t, p = ttest_1samp(metric_vals, expected_random_val)
                    print('val: {}, metric:{}, retrace_afters:{}, mean: {}, std: {}, p_val: {}'.format(val, metric,
                                                                                                       retrace_after,
                                                                                                       mean, std, p))

                    means.append(mean)
                    stds.append(std)
                    ps.append(p)

                means = np.asarray(means)
                stds = np.asarray(stds)
                ps = np.asarray(ps)
                dump(means, os.path.join(out_dir, f'means_{val}_{metric}.pkl'))
                dump(stds, os.path.join(out_dir, f'stds_{val}_{metric}.pkl'))
                dump(ps, os.path.join(out_dir, f'ps_{val}_{metric}.pkl'))

    def get_kwargs(self):
        kwargs = dict(seeds=self.seeds,
                      metrics=self.metrics,
                      retrace_afters=self.retrace_afters,
                      nepoch=self.nepochs,
                      log_scale=self.log_scale,
                      exts=('png', 'pdf'))
        return kwargs


@inherits(EvalCleansing)
class RangeCleansing(RangeCleansingParameter, myluigi.TaskBase):
    '''
    This task collect the result of the EvalCleansing changing the removal rate (fraction of the training dataset which are removed when the data cleansing)
    The output is,
        - list of removal rates
        - list of the value of test GAN evaluation metric before the data cleansing
        - list of the value of test GAN evaluation metric after the data cleansing
    Their length are the same.
    '''
    def requires(self):
        removal_rates = self.removal_rates
        assert len(removal_rates) == len(set(removal_rates))
        assert np.all(np.argsort(removal_rates) == np.arange(len(removal_rates)))

        requires = []
        for rate in removal_rates:
            requires.append(self.clone(EvalCleansing, removal_rate=rate))
        return requires

    def run_within_temporary_path(self, output_path):
        self.params = self.get_kwargs()
        self.params['input_dirs'] = self.get_input_dir()
        self.params.update({k: v for k, v in self.param_kwargs.items() if k not in self.params.keys()})
        input_dirs = self.params['input_dirs']

        metrics_dic = {
            'metric_no_removal': [],
            'metric_cleansed': [],
        }

        for input_dir, removal_rate in zip(input_dirs, self.params['removal_rates']):
            metric_dic = load(os.path.join(input_dir, 'result.pkl'))

            # no_removal, cleansed
            for k in metric_dic:
                metrics_dic[k].append(metric_dic[k])

            print('removal_rate: {}, metric_no_removal: {}, metric_cleansed: {}'.format(
                removal_rate,
                metric_dic['metric_no_removal'],
                metric_dic['metric_cleansed'],
                # approx_diffs_sum
            ))

        os.makedirs(output_path, exist_ok=True)
        dump(metrics_dic, os.path.join(output_path, 'metrics_dic.pkl'))

    def get_kwargs(self):
        kwargs = dict(
            removal_rates=self.removal_rates
        )

        return kwargs

@inherits(RangeCleansing)
class TotalizeCleansing(TotalizeCleansingParameter, myluigi.TaskBase):
    '''
    This task collect result of RangeCleansing by changing the scoring approach and random seeds.
    It evaluates the p values.
    The output is the dictionary which is like
    {
        name_of_the_scoring_approach(metric):
            [
                list_of_scores_by_changing_removal_rate_using_random_seed_0,
                list_of_scores_by_changing_removal_rate_using_random_seed_1,
                ...
                list_of_scores_by_changing_removal_rate_using_random_seed_10
            ]
    }
    It also outputs the p values with respect to removal rates.
    '''
    def requires(self):
        seeds = self.seeds
        metrics = self.metrics
        requires = []
        for seed in seeds:
            requires_seed = []
            for metric in metrics:
                requires_seed.append(self.clone(RangeCleansing, seed=seed, metric=metric))
            requires.append(requires_seed)

        return requires

    def run_within_temporary_path(self, output_path):
        params = self.get_kwargs()
        input_dirs = self.get_input_dir()
        os.makedirs(output_path, exist_ok=True)
        ex = SacredExperiment(self.__class__.__name__)
        ex.deco_main(self.main)(input_dirs=input_dirs, out_dir=output_path, **params)

    @staticmethod
    def main(_run, input_dirs, out_dir, seeds, metrics):
        # collect results across the random seeds
        result_dic_all = {name: [] for name in metrics}
        result_dic_all['metric_no_removal'] = []

        for seed, input_dirs_seed in zip(seeds, input_dirs):
            result_dic_all['metric_no_removal'].append(
                load(os.path.join(input_dirs_seed[0], 'metrics_dic.pkl'))['metric_no_removal'])
            for name, input_dir in zip(metrics, input_dirs_seed):
                eval_metric_dic = load(os.path.join(input_dir, 'metrics_dic.pkl'))
                # result_dic_all[name].append(eval_metric_dic['metric_cleansed']-eval_metric_dic['metric_no_removal'])
                result_dic_all[name].append(eval_metric_dic['metric_cleansed'])

        dump(result_dic_all, os.path.join(out_dir, 'result_dic.pkl'))

        # cal p values
        metric_no_removal = np.asarray(result_dic_all['metric_no_removal'])
        result_dic_all_diff = {k: np.asarray(v) - metric_no_removal for k, v in result_dic_all.items()}

        ps_dict = {}
        for k, eval_metric_vals in result_dic_all_diff.items():
            # alternative hyposis val = popmean = 0
            ps = []
            for val in eval_metric_vals.T:
                t, p = ttest_1samp(val, popmean=0)
                ps.append(p)
            ps_dict[k] = ps

        dump(ps_dict, os.path.join(out_dir, 'ps_dict.pkl'))

    def get_kwargs(self):
        kwargs = dict(
            seeds=self.seeds,
            metrics=self.metrics,
        )
        return kwargs


@inherits(TotalizeCleansing)
class TotalizeCleansingWrtEval(TotalizeCleansingWrtEvalParameter, myluigi.TaskBase):
    '''
    This task collect the results of TotalizeCleansing by changing the metric for test GAN evaluation metric.
    The output is the renamed files of TotalizeCleansing outputs.
    The renaming is done by adding prefix of the name of metric for test GAN evaluation metric.
    '''

    def requires(self):
        eval_metrics = self.eval_metrics
        requires = []
        for eval_metric in eval_metrics:
            requires.append(self.clone(TotalizeCleansing, eval_metric=eval_metric))
        return requires

    def run_within_temporary_path(self, output_dir):
        self.params = self.get_kwargs()
        self.params['input_dirs'] = self.get_input_dir()
        self.params.update({k: v for k, v in self.param_kwargs.items() if k not in self.params.keys()})
        ex = SacredExperiment(self.__class__.__name__)
        ex.deco_main(self.main)(output_dir=output_dir, **self.params)

    @staticmethod
    def main(output_dir, input_dirs, eval_metrics, exts=('pdf', 'png', 'csv', 'txt', 'svg', 'pkl'), _run=None,
             **kwargs):
        for input_dir, prefix in zip(input_dirs, eval_metrics):
            glob(os.path.join(input_dir, '**', '*'), recursive=True)
            artifacts_input = [x for x in glob(os.path.join(input_dir, '**', '*'), recursive=True) if
                               re.search('.*\.[{}]'.format(''.join([f'({ext})' for ext in exts])), x)]

            artifacts_output = [os.path.join(output_dir, prefix + '_' + os.path.basename(x)) for x in artifacts_input]
            for src, dst in zip(artifacts_input, artifacts_output):
                shutil.copy(src, dst)
                print(dst)

    def get_kwargs(self):
        kwargs = dict(
            eval_metrics=self.eval_metrics,
        )
        return kwargs
