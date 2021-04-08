import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tfdeterminism import patch

patch()

import experiments
import models
from modules.utils import dump, load
from modules.dataset import load_and_create_dataset


def dump_info(model, weight_dir, z_valid):
    '''
    dump ASGD training info as pkl

    Args:
        model (models.NNBase.NNBase): model instance
        weight_dir (str): directory in which pickled training info will be dumped
        z_valid (array_like): latent variables of validation dataset

    Returns: None

    '''
    params_latest = model.saver.get_current_vars()
    dump(params_latest, os.path.join(weight_dir, 'params_latest.pkl'))
    dump(model.stored_info, os.path.join(weight_dir, 'stored_info.pkl'))
    if z_valid is not None:
        dump(z_valid, os.path.join(weight_dir, 'z_valid.pkl'))


def load_info(model, weight_dir):
    '''
    load ASGD training info from pickles

    Args:
        model (models.NNBase.NNBase): model instance
        weight_dir (str): directory in which pickled training info has been dumped

    Returns:

    '''
    params_latest = load(os.path.join(weight_dir, 'params_latest.pkl'))
    stored_info = load(os.path.join(weight_dir, 'stored_info.pkl'))

    if issubclass(model.__class__, models.GANBase):
        z_valid = load(os.path.join(weight_dir, 'z_valid.pkl'))
    else:
        z_valid = None

    return stored_info, params_latest, z_valid


def main(model_type, dataset_dirs, params, mode, option, nepochs, dataset_info, weight_dir,
         train_kwargs, fetches, seed=1204, tf_quiet=True, _run=None):
    '''
    Run experiments

    Args:
        model_type (str): name of the model architecture
        dataset_dirs (dict): dict of paths of datasets (keys are 'train', 'valid', and 'test')
        params (dict): hyperparameters for the model
        mode (str): one of,
            'vanilla_train': training without storing info (for classifier)
            'train': training by storing info (actual asgd training)
            'lininfl': estimating influence on something (e.g., inception score, discriminator loss)
            'cfsgd': performing actual counterfactual ASGD
            'cleansing': data cleansing using influence on GAN evaluation metric
            'baseline': scoring instnaces using baseline approaches
            'cleansing_wo_infl': data cleansing using baseline selection
            'eval_cleansing': evaluate test GAN evaluation metric before and after the data cleansing
        option (dict): options for experiments
            'metric' (str): name of metric tensor on which influence is computed (used when mode in ['baseline' scoring])
            'ncfsgd' (int): number of instances of which influence on the metric is computed (used when mode in ['cfasgd', 'lininfl'])
            'removal_rate' (float): the fraction of the removed instance n_c (0. <= n_c <= 1.)
            'metric_kwargs' (dict): kwargs passed to self.eval_metric
            'original_weight_dir' (str): model output dir before the data cleansing (used when mode in ['cfsgd', 'lininfl', 'cleansing', 'cleansing_wo_infl', 'baseline'])
            'lininfl_dir': dir in which output of 'lininfl' mode is placed  (used when mode == 'cleansing')
            'use_valid'(bool): when True, the test GAN evaluation metric is calculated on the validation dataset. use False to calculate on the test dataset. (used when mode == 'val_cleansing')
            'cleansed_weight_dir':  model output dir after the data cleansing (used when mode == 'eval_cleansing'])
        nepochs: number of epochs of training
        dataset_info (dict):
        weight_dir (str): output directory
        train_kwargs (dict): kwargs of model.sgd_train
        fetches (list): names of tensors that are evaluated in the training
        seed (Int): random seed
        tf_quiet (Boolean): for tf.logging.set_verbosity
        _run (Sacred.run): Sacred.Run instance for tracking experimental results and parameters

    Returns:

    '''
    if tf_quiet:
        tf.logging.set_verbosity(tf.logging.ERROR)

    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    Model = models.models_dict[model_type]

    x = tf.placeholder(Model.dtype, (None, *dataset_info['x_shape']), 'x')
    y = tf.placeholder(Model.dtype, (None, dataset_info['nclasses']), 'y')
    if issubclass(Model, models.Generator):
        z = tf.placeholder(Model.dtype, (None, params['z_dim']), 'z')
    else:
        z = None

    model = Model(params, weight_dir, x=x, y=y, z=z, _run=_run)

    # set validation loss fetch during training
    # fetches = [getattr(model, x) for x in fetches]
    dataset_train = load_and_create_dataset(dataset_dirs['train'], x_ph=x, y_ph=y, name='train')
    dataset_valid = load_and_create_dataset(dataset_dirs['valid'], x_ph=x, y_ph=y, name='valid')
    dataset_test = load_and_create_dataset(dataset_dirs['test'], x_ph=x, y_ph=y, name='test')

    if issubclass(Model, models.Generator):
        z_valid = np.random.normal(size=[dataset_valid.sample_size, params['z_dim']])
        dataset_valid.update({z: z_valid})
        z_test = np.random.normal(size=[dataset_test.sample_size, params['z_dim']])
        dataset_test.update({z: z_test})

    if mode == 'vanilla_train':
        model.sgd_train(dataset_train,
                        nepochs,
                        metrics=fetches,
                        dataset_valid=dataset_test,
                        **train_kwargs,
                        metric_kwargs=option['metric_kwargs'])

        params_cleansed = model.saver.get_current_vars()
        dump(params_cleansed, os.path.join(weight_dir, 'params_latest.pkl'))

        return

    elif mode == 'train':
        model.sgd_train(dataset_train,
                        nepochs,
                        mode='store',
                        metrics=fetches,
                        dataset_valid=dataset_test,
                        **train_kwargs,
                        metric_kwargs=option['metric_kwargs'])

        if issubclass(Model, models.GANBase):
            z_valid = dataset_valid.feed_dict[model.z]
        else:
            z_valid = None

        dump_info(model, weight_dir, z_valid)

    elif mode in ['cfsgd', 'lininfl', 'cleansing', 'cleansing_wo_infl', 'baseline']:
        trained_weight_dir = option['original_weight_dir']
        stored_info, params_cleansed, z_valid = load_info(model, trained_weight_dir)
        if z_valid is not None:
            dataset_valid.update({z: z_valid})
        model.stored_info = stored_info
        model.saver.restore(params_cleansed)

        if mode == 'lininfl':
            approx_diffs = experiments.lininfl(_run=_run,
                                               option=option,
                                               nepochs=nepochs,
                                               weight_dir=weight_dir,
                                               model=model,
                                               dataset_train=dataset_train,
                                               dataset_valid=dataset_valid,
                                               train_kwargs=train_kwargs,
                                               fetches=fetches,
                                               metric_kwargs=option['metric_kwargs'])

            # gen csv
            df = pd.DataFrame({'approx': approx_diffs})
            csv_path = os.path.join(weight_dir, 'approx_diffs.csv')
            df.to_csv(csv_path)
            _run.add_artifact(csv_path)

            # save variables for reproduction
            dump(approx_diffs, os.path.join(weight_dir, 'approx_diffs.pkl'))

            return

        elif mode == 'baseline':
            approx_diffs = experiments.baseline_scoring(_run=_run,
                                                        option=option,
                                                        model=model,
                                                        dataset_train=dataset_train,
                                                        metric_kwargs=option['metric_kwargs'],
                                                        train_dataset_dir=dataset_dirs['train'])

            # gen csv
            df = pd.DataFrame({'approx': approx_diffs})
            csv_path = os.path.join(weight_dir, 'approx_diffs.csv')
            df.to_csv(csv_path)
            _run.add_artifact(csv_path)

            # save variables for reproduction
            dump(approx_diffs, os.path.join(weight_dir, 'approx_diffs.pkl'))

            return

        elif mode == 'cfsgd':
            metric_no_removal = model.eval_metric(option['metric'], dataset_valid[:], **option['metric_kwargs'])
            actual_diffs = experiments.cfsgd(_run=_run,
                                             option=option,
                                             nepochs=nepochs,
                                             weight_dir=weight_dir,
                                             model=model,
                                             dataset_train=dataset_train,
                                             dataset_valid=dataset_valid,
                                             train_kwargs=train_kwargs,
                                             fetches=fetches,
                                             metric_no_removal=metric_no_removal,
                                             metric_kwargs=option['metric_kwargs'])

            # gen csv
            df = pd.DataFrame({'actual': actual_diffs})
            csv_path = os.path.join(weight_dir, 'actual_diffs.csv')
            df.to_csv(csv_path)
            _run.add_artifact(csv_path)

            # save variables for reproduction
            dump(actual_diffs, os.path.join(weight_dir, 'actual_diffs.pkl'))

            return

        elif mode == 'cleansing':
            approx_diffs = load(os.path.join(option['lininfl_dir'], 'approx_diffs.pkl'))

            experiments.cleansing(_run=_run,
                                  option=option,
                                  nepochs=nepochs,
                                  weight_dir=weight_dir,
                                  model=model,
                                  dataset_train=dataset_train,
                                  dataset_valid=dataset_valid,
                                  dataset_test=dataset_test,
                                  train_kwargs=train_kwargs,
                                  fetches=fetches,
                                  metric_kwargs=option['metric_kwargs'],
                                  approx_diffs=approx_diffs,
                                  train_dataset_dir=dataset_dirs['train'])

            params_cleansed = model.saver.get_current_vars()
            dump(params_cleansed, os.path.join(weight_dir, 'params_cleansed.pkl'))

    elif mode == 'eval_cleansing':
        if option['use_valid']:
            dataset_test = dataset_valid

        params_original = load(os.path.join(option['original_weight_dir'], 'params_latest.pkl'))
        params_cleansed = load(os.path.join(option['cleansed_weight_dir'], 'params_cleansed.pkl'))

        if mode == 'eval_cleansing':
            eval_metric_no_removal, eval_metric_cleansed = model.eval_improvements(params_original=params_original,
                                                                                   params_cleansed=params_cleansed,
                                                                                   dataset_test=dataset_test,
                                                                                   eval_metric=option['eval_metric'],
                                                                                   metric_kwargs=option['metric_kwargs'])

            result = dict(metric_no_removal=eval_metric_no_removal, metric_cleansed=eval_metric_cleansed)
            dump(result, os.path.join(weight_dir, 'result.pkl'))

            if _run is not None:
                for k, v in result.items():
                    _run.log_scalar(k, v)
            print(result)

            model.dump_visual_improvements(params_original, params_cleansed, dataset_test)
            return eval_metric_cleansed

    else:
        raise ValueError('Invalid mode: {}'.format(mode))
