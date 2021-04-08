import enum

import luigi

import models
from modules import myluigi


class DatasetShape(enum.Enum):
    '''
    shapes of a single instance of MNIST and 2D-Normal (MULVAL_GAUSSGAN)
    '''
    MNIST = (28, 28, 1)
    MULVAR_GAUSSGAN = [2]


def get_dataset_info(dataset_name, nclasses):
    '''
    return dataset information dict according to dataset_name and nclasses

    Args:
        dataset_name: for MNIST, use "mnist" and for 2d-normal use "mulvargaussian"
        nclasses: nclasses of the dataset. we used 10 for mnist and we force it 1 when 2d-normal

    Returns: dict
    '''

    if dataset_name in ['mnist']:
        dataset_info = {'nclasses': nclasses,
                        'x_shape': DatasetShape.MNIST.value}
    elif dataset_name in ['mulvargaussian']:
        dataset_info = {'nclasses': 1,
                        'x_shape': DatasetShape.MULVAR_GAUSSGAN.value}
    else:
        raise ValueError(dataset_name)
    return dataset_info


class HyperParameter(luigi.Config):
    '''
    luigi config class.
    it inherited by some tasks
    it only contains the hyper parameters of the training and architectures.
    see comments to understand the relation between variables and what it represents.

    Attributes:
        ld: learning rate of the discriminator
        lg: learning rate of the discriminator
        ld_v: not used
        lg_v: not used
        lr: learning rate for the classifier
        w: weight decay coefficient of the classifier params
        wd: weight decay coefficient of the discriminator params
        wg: weight decay coefficient of the generator params
        bs: batchsize
        a: architecture name. option of choosing it can be found in the attributes in the classes defined in ./model/*.py
        bn_g: when True, Layer Normalization is used to the discriminator params
        bn_d: when True, Layer Normalization is used to the generator params
        sim: when True, GANs are trained by the simultaneous SGD, otherwise, by the alternative SGD
        h_g_dim: the number of the nodes or channels of hidden layers of the generator
        h_d_dim: the number of the nodes or channels of the hidden layers of the discriminator
        h_dim: the number of the hidden layers of the classifier
        z_dim: the number of the dimensions of the generator
        retrace_after: the epoch number until which the ASGD-Influence traces back from the latest epoch. e.g., when you set 49 for the 50-epoch training, it only traces back the last 1 epoch.
        loss_method: the type of the loss function. "modified_minmax" is proposed as an practical one in the original GAN paper.
        nlayers_g: the number of the hidden layers of the generator
        nlayers_d: the number of the hidden layers of the discriminator
        lr_scheduler: learning rate scheduling. when 'constant' it does not change through the entire training, when 'ghadimi' it uses the scheduling proposed in " Stochastic first-and zeroth-order methods for nonconvex stochastic programming", 2013
    '''
    ld = luigi.FloatParameter(1e-5)
    lg = luigi.FloatParameter(1e-5)
    ld_v = luigi.FloatParameter(1e-5)
    lg_v = luigi.FloatParameter(1e-5)
    lr = luigi.FloatParameter(1e-5)
    w = luigi.FloatParameter(0.)
    wd = luigi.FloatParameter(0.)
    wg = luigi.FloatParameter(0.)
    bs = luigi.IntParameter(64)
    a = luigi.Parameter('smallgan')
    bn_g = luigi.BoolParameter()
    bn_d = luigi.BoolParameter()
    sim = luigi.BoolParameter()
    h_g_dim = luigi.IntParameter(8)
    h_d_dim = luigi.IntParameter(8)
    h_dim = luigi.IntParameter(8)
    z_dim = luigi.IntParameter(10)
    retrace_after = luigi.IntParameter(0)
    loss_method = luigi.Parameter('modified_minmax')
    nlayers_g = luigi.IntParameter(2)
    nlayers_d = luigi.IntParameter(2)
    lr_scheduler = luigi.Parameter('constant')

    def get_hyperparams(self):
        '''
        returns the dictionary of the hyper-parameters that passed to the Models

        Returns: dict of kwargs of Models
        '''
        params_base = dict(
            batch_size=self.bs,
            lr_scheduler=self.lr_scheduler,
            retrace_after=self.retrace_after
        )
        if self.a in [models.SmallCNNGAN.name]:
            params = dict(
                **params_base,
                **dict(lrs={'discriminator': self.ld, 'generator': self.lg},
                       z_dim=self.z_dim,
                       weight_decay_g=self.wg,
                       weight_decay_d=self.wd,
                       use_bn_d=self.bn_d,
                       use_bn_g=self.bn_g,
                       h_g_dim=self.h_g_dim,
                       h_d_dim=self.h_d_dim,
                       sim=self.sim,
                       loss_method=self.loss_method)
            )
        elif self.a in [models.SmallMulVarGaussGAN.name]:
            params = dict(
                **params_base,
                **dict(lrs={'discriminator': self.ld, 'generator': self.lg},
                       h_g_layer=self.nlayers_g,
                       h_d_layer=self.nlayers_d,
                       h_g_dim=self.h_g_dim,
                       h_d_dim=self.h_d_dim,
                       use_bn_d=self.bn_d,
                       use_bn_g=self.bn_g,
                       z_dim=self.z_dim,
                       weight_decay_g=self.wg,
                       weight_decay_d=self.wd,
                       sim=self.sim,
                       loss_method=self.loss_method)
            )
        elif self.a in [models.CNNMNIST.name]:
            params = dict(
                **params_base,
                **dict(lr=self.lr,
                       weight_decay=self.w)
            )
        elif self.a in ['inception']:
            params = dict(
                **params_base,
                **dict(lr=self.lr,
                       weight_decay=self.w)
            )
        else:
            raise ValueError('invalid model: {}'.format(self.a))

        return params


class TrainClassifierParameter(luigi.Config):
    classifier = luigi.Parameter('mlpmnist')
    seed = luigi.IntParameter(2)


class DownloadGeneratorDatasetParameter(luigi.Config):
    '''
    Attributes:
        d: name of the dataset
    '''
    d = luigi.Parameter('mnist')


class MakeGeneratorDatasetParameter(luigi.Config):
    '''
    Attributes:
        seed:
        d:
        nc: number of classes
        s_tr: the size of the training dataset
        s_va: the size of the validation dataset
        s_te: the size of the test dataset
    '''
    seed = luigi.IntParameter(2)
    d = luigi.Parameter('mnist')
    nc = luigi.IntParameter(10)
    s_tr = luigi.IntParameter(50000)
    s_va = luigi.IntParameter(10000)
    s_te = luigi.IntParameter(10000)


class ExperimentParameterBase(HyperParameter):
    seed = luigi.IntParameter(2)
    fetches = myluigi.ListParameter([])
    eval_interval = luigi.IntParameter(1)
    d = luigi.Parameter('mnist')
    nc = luigi.IntParameter(10)
    a = luigi.Parameter('smallgan')
    nepochs = luigi.IntParameter(10)

    def get_base_kwargs(self):
        params = self.get_hyperparams()
        dataset_info = get_dataset_info(self.d, self.nc)
        base_config = dict(
            model_type=self.a,
            params=params,
            nepochs=self.nepochs,
            fetches=self.fetches,
            dataset_info=dataset_info,
            train_kwargs={'eval_interval': self.eval_interval},
            seed=self.seed,
        )
        return base_config


class InfluenceEstimationParameter(ExperimentParameterBase):
    '''
    Attributes:
        ncfsgd: number of the instances for which the influence on something is calculated
        converge_check: when True, it does not influence but hvp
        damping: experimental. it is not used in the paper setting. i.e, it was set to 0.
        metric: the evaluation metric used for the influence estimation
    '''
    ncfsgd = luigi.IntParameter(-1)
    converge_check = luigi.BoolParameter()
    damping = luigi.FloatParameter(0.)
    metric = luigi.Parameter('loss_d')


class BaselineScoringParameter(ExperimentParameterBase):
    metric = luigi.Parameter('loss_d')


class CounterFactualSGDParameter(ExperimentParameterBase):
    ncfsgd = luigi.IntParameter(-1)
    metric = luigi.Parameter('loss_d')


class CleansingParameter(ExperimentParameterBase):
    '''
    Attributes:
        removal_rate: the fraction of the removed instance n_c (0. <= n_c <= 1.)
        metric:
    '''
    removal_rate = luigi.FloatParameter(0.1)  # the fraction of the removed instance n_c (0. <= n_c <= 1.)
    metric = luigi.Parameter('loss_d')


class EvalCleansingParameter(ExperimentParameterBase):
    '''
    Attributes:
        use_valid: when True, the test GAN evaluation metric is calculated on the validation dataset. use False to calculate on the test dataset.
        eval_metric: the name of the GAN evaluation metric for calculating test GAN evaluation metric after the data cleansing
    '''
    use_valid = luigi.BoolParameter(False)
    eval_metric = luigi.Parameter('fid')


class TrainParameter(ExperimentParameterBase):
    pass


class ValidParameter(HyperParameter):
    '''
    Attributes:
        jaccard_size: the fraction of the training dataset that used for obtaining jaccard score between true and predicted highly-influenial instances.
    '''
    jaccard_size = luigi.FloatParameter(0.1)


class RangeCleansingParameter(luigi.Config):
    removal_rates = myluigi.ListParameter([])


class TotalizeCleansingParameter(luigi.Config):
    '''
    Attributes:
        metrics:
        removal_rates:
        seeds: the list of the seeds for the repeated trainings
        eval_metric:
    '''
    metrics = myluigi.ListParameter([])
    removal_rates = myluigi.ListParameter([])
    seeds = myluigi.ListParameter([])
    eval_metric = luigi.Parameter('fid')


class TotalizeCleansingWrtEvalParameter(luigi.Config):
    eval_metrics = myluigi.ListParameter([])


class TotalizeValidParameter(luigi.Config):
    '''
    Attributes:
        log_scale: plot with log_scale horizontal axis
        metrics:
        retrace_afters: the list of "retrace_after"
        seeds:
        nepochs:
    '''
    log_scale = luigi.BoolParameter()
    metrics = myluigi.ListParameter([])
    retrace_afters = myluigi.ListParameter([])
    seeds = myluigi.ListParameter([])
    nepochs = luigi.IntParameter(10)


def get_all_params():
    return [x for x in globals().values() if type(x) is luigi.task_register.Register]
