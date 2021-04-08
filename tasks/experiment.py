import os

import luigi
import pandas as pd
import tensorflow as tf
from luigi.util import inherits
from scipy.stats import kendalltau
from sklearn.metrics import jaccard_score

import models
from config import TrainClassifierParameter, TrainParameter, \
    InfluenceEstimationParameter, CleansingParameter, \
    EvalCleansingParameter, CounterFactualSGDParameter, ValidParameter, BaselineScoringParameter
from experiments.main import main as experiment
from modules import myluigi
from modules.mysacred import Experiment as SacredExperiment
from modules.plot import gen_error_fig
from modules.utils import load, dump, get_lowest_or_highest_score_indices, expected_indepencent_jaccard, \
    dump_json
from tasks.dataset import MakeGeneratorDataset


@inherits(MakeGeneratorDataset)
class TrainClassifier(TrainClassifierParameter, myluigi.TaskBase):
    '''
    This task trains classifier of mnist for computing features and class probability.
    The output is the trained parameters of the classifier
    '''
    def requires(self):
        return [self.clone(MakeGeneratorDataset)]

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        with tf.Graph().as_default():
            return experiment(weight_dir=temp_output_path, **self.params)

    def get_kwargs(self):
        if self.classifier in [models.CNNMNIST.name]:
            kwargs = dict(
                model_type=self.classifier,
                params=dict(batch_size=128,
                            h_dim=128,
                            weight_decay=1e-3,
                            lrs={'classifier': 1e-2},
                            lr_scheduler='constant'),
                dataset_info={'nclasses': 10,
                              'x_shape': (28, 28, 1)},
                nepochs=50,
                fetches=['loss', 'acc'],
                mode='vanilla_train',
                train_kwargs={},
                seed=self.seed,
                option={
                    'scale': False,
                    'metric_kwargs': {}
                }
            )
        else:
            raise ValueError(self.classifier)

        input = self.input()
        dataset_dirs = {
            'train': '{}/valid'.format(input[0].path),
            'test': '{}/valid'.format(input[0].path),
            'valid': '{}/valid'.format(input[0].path),
        }
        kwargs['dataset_dirs'] = dataset_dirs
        return kwargs


@inherits(MakeGeneratorDataset, TrainClassifier)
class Train(TrainParameter, myluigi.TaskBase):
    '''
    This task performs ASGD training.
    The output is the trained parameters and stored information (e.g., intermediate parameters, latent variables, and mini-batch indices)
    '''
    dirname = 'train'

    def get_kwargs(self):
        base_config = self.get_base_kwargs()
        kwargs = dict(
            **base_config,
            **dict(
                mode='train',
                option={
                    'metric_kwargs': {},
                })
        )

        kwargs['dataset_dirs'] = self.get_dataset_dirs()
        kwargs['option']['metric_kwargs'] = self.get_metric_kwargs()
        return kwargs

    def requires(self):
        requires = {}
        if self.a in [models.SmallMulVarGaussGAN.name]:
            requires.update({MakeGeneratorDataset.task_family: self.clone(MakeGeneratorDataset)})
        else:
            requires.update({MakeGeneratorDataset.task_family: self.clone(MakeGeneratorDataset),
                             TrainClassifier.task_family: self.clone(TrainClassifier)})

        return requires

    def get_dataset_dirs(self):
        inputs = self.input()
        dataset_dirs = {
            'train': '{}/train'.format(inputs[MakeGeneratorDataset.task_family].path),
            'valid': '{}/valid'.format(inputs[MakeGeneratorDataset.task_family].path),
            'test': '{}/test'.format(inputs[MakeGeneratorDataset.task_family].path),
        }

        return dataset_dirs

    def get_metric_kwargs(self):
        metric_kwargs = {}
        inputs = self.input()
        if TrainClassifier.task_family in inputs:  # w
            metric_kwargs['classifier_conf'] = dict(weight_dir=inputs[TrainClassifier.task_family].path,
                                                    **self.requires()[TrainClassifier.task_family].get_kwargs())
        return metric_kwargs

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        ex = SacredExperiment(self.__class__.__name__)
        with tf.Graph().as_default():
            ex.deco_main(experiment)(weight_dir=temp_output_path, **self.params)

@inherits(Train)
class InfluenceEstimation(InfluenceEstimationParameter, myluigi.TaskBase):
    '''
    This task computes influence on metric.
    The output is the array of influence on the metric of the training instances suggested by influence on the metric.
    '''
    def requires(self):
        return [self.clone(Train)]

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        ex = SacredExperiment(self.__class__.__name__)
        with tf.Graph().as_default():
            ex.deco_main(experiment)(weight_dir=temp_output_path, **self.params)

    def get_kwargs(self):
        base_config = self.get_base_kwargs()
        kwargs = dict(
            **base_config,
            **dict(
                mode='lininfl',
                option={
                    'metric': self.metric,
                    'metric_kwargs': {},
                    'infl_args': {
                        'converge_check': self.converge_check,
                        'damping': self.damping
                    },
                    'ncfsgd': self.ncfsgd,
                }
            ))
        input_dirs = self.get_input_dir()
        kwargs['option']['original_weight_dir'] = input_dirs[0]
        kwargs['dataset_dirs'] = self.requires()[0].get_kwargs()['dataset_dirs']
        kwargs['option']['metric_kwargs'] = self.requires()[0].get_kwargs()['option']['metric_kwargs']

        return kwargs


@inherits(Train)
class BaselineScoring(BaselineScoringParameter, myluigi.TaskBase):
    '''
    This task computes scores of baseline approaches.
    The output is the array of harmful scores of the training instances suggested by a baseline approach.
    '''
    def requires(self):
        return [self.clone(Train)]

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        ex = SacredExperiment(self.__class__.__name__)
        with tf.Graph().as_default():
            ex.deco_main(experiment)(weight_dir=temp_output_path, **self.params)

    def get_kwargs(self):
        base_config = self.get_base_kwargs()
        kwargs = dict(
            **base_config,
            **dict(
                mode='baseline',
                option={
                    'metric': self.metric,
                    'metric_kwargs': {},
                }
            ))
        input_dirs = self.get_input_dir()
        kwargs['option']['original_weight_dir'] = input_dirs[0]
        kwargs['dataset_dirs'] = self.requires()[0].get_kwargs()['dataset_dirs']
        kwargs['option']['metric_kwargs'] = self.requires()[0].get_kwargs()['option']['metric_kwargs']

        return kwargs


@inherits(InfluenceEstimation, BaselineScoring)
class Cleansing(CleansingParameter, myluigi.TaskBase):
    '''
    This task selects harmful instances to be removed based on influence on metric or a baseline approach.
    Then it performs the counterfactual ASGD without selected harmful instances.
    The output is the trained parameter after the data cleansing.
    '''
    def requires(self):
        if self.metric in ['inception_score', 'log_inception_score', 'fid', 'loss_d', 'log_likelihood',
                           'log_likelihood_kde']:
            return [self.clone(InfluenceEstimation)]
        elif self.metric in ['random', 'if', 'if_data']:
            return [self.clone(BaselineScoring)]
        else:
            raise ValueError(self.metric)

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        ex = SacredExperiment(self.__class__.__name__)
        with tf.Graph().as_default():
            ex.deco_main(experiment)(weight_dir=temp_output_path, **self.params)

    def get_kwargs(self):
        base_config = self.get_base_kwargs()
        kwargs = dict(
            **base_config,
            **dict(
                mode='cleansing',
                option={
                    'metric': self.metric,
                    'removal_rate': self.removal_rate,
                    'metric_kwargs': {},
                    # 'cleansing_metrics': self.cleansing_metrics
                }
            ))

        kwargs['option']['lininfl_dir'] = self.input()[0].path
        kwargs['option']['original_weight_dir'] = self.requires()[0].get_kwargs()['option']['original_weight_dir']
        kwargs['dataset_dirs'] = self.requires()[0].get_kwargs()['dataset_dirs']
        kwargs['option']['metric_kwargs'] = self.requires()[0].get_kwargs()['option']['metric_kwargs']

        return kwargs


@inherits(Cleansing)
class EvalCleansing(EvalCleansingParameter, myluigi.TaskBase):
    '''
    It evaluates test GAN evaluation metrics on the model before and after the data cleansing.
    The output is,
        - the values of test GAN evaluation metrics on the model before and after the data cleansing
        - visual examples of generated samples on the model before and after the data cleansing
    '''
    metric = luigi.Parameter('loss_d')

    def requires(self):
        return [self.clone(Cleansing)]

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        ex = SacredExperiment(self.__class__.__name__)
        with tf.Graph().as_default():
            ex.deco_main(experiment)(weight_dir=temp_output_path, **self.params)

    def get_kwargs(self):
        base_config = self.get_base_kwargs()
        kwargs = dict(
            **base_config,
            **dict(
                mode='eval_cleansing',
                option={
                    'eval_metric': self.eval_metric,
                    'metric_kwargs': {},
                    'use_valid': self.use_valid
                }
            ))
        input_dirs = self.get_input_dir()
        kwargs['option']['cleansed_weight_dir'] = input_dirs[0]
        kwargs['option']['original_weight_dir'] = self.requires()[0].get_kwargs()['option']['original_weight_dir']
        kwargs['dataset_dirs'] = self.requires()[0].get_kwargs()['dataset_dirs']
        kwargs['option']['metric_kwargs'] = self.requires()[0].get_kwargs()['option']['metric_kwargs']

        return kwargs


@inherits(Train)
class CounterFactualSGD(CounterFactualSGDParameter, myluigi.TaskBase):
    '''
    It performs actual counterfactual ASGD to compute true influence on the metric.
    The output is the array of the true influences on the metrics of the training instances.
    '''
    def requires(self):
        return [self.clone(Train)]

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()

        ex = SacredExperiment(self.__class__.__name__)
        with tf.Graph().as_default():
            ex.deco_main(experiment)(weight_dir=temp_output_path, **self.params)

    def get_kwargs(self):
        base_config = self.get_base_kwargs()
        kwargs = dict(
            **base_config,
            **dict(
                mode='cfsgd',
                option={
                    'metric': self.metric,
                    'metric_kwargs': {},
                    'ncfsgd': self.ncfsgd,
                }
            ))
        input_dirs = self.get_input_dir()
        kwargs['option']['original_weight_dir'] = input_dirs[0]
        kwargs['dataset_dirs'] = self.requires()[0].get_kwargs()['dataset_dirs']
        kwargs['option']['metric_kwargs'] = self.requires()[0].get_kwargs()['option']['metric_kwargs']
        return kwargs


@inherits(InfluenceEstimation, CounterFactualSGD)
class Valid(ValidParameter, myluigi.TaskBase):
    '''
    It collects true and predicted influence on metric from CounterFactualSGD and InfluenceEstimation, respectively.
    The output is the value of Kental's tau and Jaccard Index.
    '''
    def requires(self):
        return [self.clone(InfluenceEstimation),
                self.clone(CounterFactualSGD)]

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        input_dirs = self.get_input_dir()
        self.params['lininfl_dir'] = input_dirs[0]
        self.params['cfsgd_dir'] = input_dirs[1]
        self.params.update({k: v for k, v in self.param_kwargs.items() if k not in self.params.keys()})
        os.makedirs(temp_output_path, exist_ok=True)
        ex = SacredExperiment(self.__class__.__name__)
        ex.deco_main(self.main)(out_dir=temp_output_path, **self.params)

    @staticmethod
    def main(out_dir, lininfl_dir, cfsgd_dir, jaccard_size, exts, _run, **kwargs):
        approx_diffs_selected = load(os.path.join(lininfl_dir, 'approx_diffs.pkl'))
        actual_diffs_selected = load(os.path.join(cfsgd_dir, 'actual_diffs.pkl'))

        df = pd.DataFrame({'actual': actual_diffs_selected, 'approx': approx_diffs_selected})
        csv_path = os.path.join(out_dir, 'lie_error.csv')
        df.to_csv(csv_path)
        _run.add_artifact(csv_path)

        # gen error scatter figure
        fig = gen_error_fig(actual_diffs_selected, approx_diffs_selected,
                            _run,
                            title=None,
                            xlabel='Actual influence on metric',
                            ylabel='Predicted influence on metric',
                            score=None)
        dump(fig, os.path.join(out_dir, 'fig.pkl'))
        for ext in exts:
            path = os.path.join(out_dir, 'lie_error.{}'.format(ext))
            fig.savefig(path)
            if _run is not None:
                _run.add_artifact(path)

        # metrics
        ## kendalltau
        tau, _ = kendalltau(actual_diffs_selected, approx_diffs_selected)

        ## jaccard index
        nsamples = int(len(actual_diffs_selected) * jaccard_size)
        assert nsamples % 2 == 0
        assert nsamples / 2 > 0
        actual_is_influential = get_lowest_or_highest_score_indices(actual_diffs_selected, int(nsamples / 2),
                                                                    int(nsamples / 2))
        approx_is_influential = get_lowest_or_highest_score_indices(approx_diffs_selected, int(nsamples / 2),
                                                                    int(nsamples / 2))
        jaccard = jaccard_score(actual_is_influential, approx_is_influential)

        if _run is not None:
            _run.log_scalar('kendall_tau', tau)
            _run.log_scalar('jaccard_score', jaccard)

        jaccard_random = expected_indepencent_jaccard(len(actual_diffs_selected), nsamples)

        dump_json(dict(tau=tau,
                       jaccard=jaccard),
                  os.path.join(out_dir, 'result.json'))

        dump(jaccard_random, os.path.join(out_dir, 'jaccard_random.pkl'))

        return tau, jaccard

    def get_kwargs(self):
        kwargs = {'jaccard_size': self.jaccard_size,
                  'exts': ('png', 'pdf')}
        return kwargs
