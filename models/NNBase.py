import os
from collections import OrderedDict
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

from modules.tf_ops import MySaver, zeros_from_arrays, add_arrays, mul_const_by_arrays

from modules.utils import get_minibatch_indices

class NNBase(object):
    name = 'nnbase'
    scopes = ['base']
    scope_suffixes = ['_b']
    dtype = tf.float32

    def __init__(self, hypparams, model_dir, x, y=None, z=None, _run=None):
        os.makedirs(model_dir, exist_ok=True)
        self.dir = model_dir
        self._run = _run
        self.x = x
        self.y = y
        self.z = z
        self.hypparams = hypparams

        # build graph and initialize
        self.losses, self.optimizers, self.vars, self.lrs, self.reg_losses, self.bare_losses = self.build(x=x, y=y, z=z)
        for scope, var in self.vars.items():
            print('scope: {}'.format(scope))
            self.print_variable_info(var)

        # initialize
        self.init = tf.global_variables_initializer()

        self.sess = self.create_gpu_session()
        self.sess.run(self.init)

        # get saver
        self.saver = MySaver(tf.trainable_variables(), self.sess)

    def create_gpu_session(self):
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        return tf.Session(config=config)

    def build(self, x, y, z):
        '''
        make computational graph

        Args:
            x (Tensor): input
            y (Tensor): output, None for Generator
            z (Tensor): latent variable, None for Classifier

        Returns:
            losses (list): list of loss tensors passed to the optimizers
            optimizers (list): list of optimizer ops
            vars (list): list of the list of Variables
            lrs (list): list of learning rates tensors
            reg_losses (list): list of regularization loss tensors
            bare_losses (list): list of non-regularized loss tensors
        '''
        raise NotImplementedError

    def init_stored_info(self):
        '''
        initialize self.stored_info used for the counterfactual training

        Returns: None

        '''
        raise NotImplementedError

    def print_variable_info(self, variables):
        # print out variables information
        nparams_total = 0
        pt_param = PrettyTable(['index', 'param', 'shape', 'size', 'total'])
        for i, w in enumerate(variables):
            param_shape = w.shape.as_list()
            nparams = np.prod(param_shape).astype(np.int32)
            nparams_total += nparams
            pt_param.add_row([i, w.name, param_shape, nparams, nparams_total])
        print(pt_param)

    def sgd_train(self, dataset_train, nepochs, dataset_valid=None, metrics=None,
                  quiet=False, mode='vanilla', eval_interval=1, j=-1, metric_kwargs={}):
        '''
        run ASGD training.

        Args:
            dataset_train (modules.dataset.MyDataset): dataset used for the training
            nepochs: number of training epochs
            dataset_valid (modules.dataset.MyDataset): dataset used for the validation
            metrics: name of ops or tensor that are evaluated by self.eval_metric at the end of the epoch
            quiet: when True, stop printing
            mode:
                'vanilla': starts training from 0 epoch with newly sampled minibatch indices. no save of the information
                'store': same as vanilla trainig but save the information (e.g., parameters, latent variables learning rates, epoch num) after the epoch specified in self.hypparams['retrace_after'] for counterfactual training
                'counterfactual': restart training using the stored information. it can restarts from intermediate epoch if the the epoch num starts from intermidiate epoch)
            eval_interval (int): interval of epochs how frequently self.eval_metric is ran
            j (array_like): indices of removed instances from minibatches. used only if mode == 'counterfactual'.
            metric_kwargs: kwargs passed to self.eval_metric

        Returns: None

        '''

        if mode in ['vanilla']:
            epoch_start = 0
            minibatch_indices_of_epochs = []
            for epoch in range(epoch_start, nepochs):
                minibatch_indices_of_epochs.append(get_minibatch_indices(dataset_train.sample_size,
                                                                         self.hypparams['batch_size'],
                                                                         append_remainder=False))
            get_opt_scopes = self.get_opt_scopes
            get_minibatch = self.sample_minibatch
            gen_sample_images = self.gen_sample_images

        elif mode in ['store']:
            epoch_start = 0
            minibatch_indices_of_epochs = []
            for epoch in range(epoch_start, nepochs):
                minibatch_indices_of_epochs.append(get_minibatch_indices(dataset_train.sample_size,
                                                                         self.hypparams['batch_size'],
                                                                         append_remainder=False))
            get_opt_scopes = self.get_opt_scopes
            get_minibatch = self.sample_minibatch_by_storing_info
            gen_sample_images = self.gen_sample_images

        elif mode == 'counterfactual':
            self.saver.restore(self.stored_info['params'][0])
            epoch_start = self.stored_info['epoch'][0]
            nepochs_retrace = nepochs - epoch_start
            minibatch_indices_of_steps = np.asarray(self.stored_info['indices'])
            assert minibatch_indices_of_steps.shape[1] == (self.hypparams['batch_size'])
            minibatch_indices_of_epochs = np.reshape(minibatch_indices_of_steps, [nepochs_retrace, -1, self.hypparams['batch_size']])
            get_opt_scopes = self.restore_opt_scopes
            get_minibatch = self.restore_minibatch
            gen_sample_images = lambda *args, **kwargs: None

        else:
            raise ValueError('invalid mode: {}'.format(mode))

        # refresh feed_dict indice
        start_time = time.time()
        step = 0

        for epoch, minibatch_indices_of_epoch, in zip(range(epoch_start, nepochs), minibatch_indices_of_epochs):
            # mini batch train
            with tqdm(minibatch_indices_of_epoch, disable=quiet, ascii=True) as pbar:
                for minibatch_indices in pbar:
                    pbar.set_description('Epoch: [{}/{}]'.format(epoch, nepochs))

                    # update feed dict
                    scopes = get_opt_scopes(step)
                    feed_dict = get_minibatch(dataset_train, minibatch_indices, step, scopes, epoch)

                    # param update
                    if mode == 'counterfactual':
                        if len(np.intersect1d(j, minibatch_indices)) > 0:
                            feed_dict[self.x] = feed_dict[self.x][np.logical_not(np.isin(minibatch_indices, j))]

                    optimizers = self.get_optimizers(scopes, epoch)
                    self.sess.run(optimizers, feed_dict=feed_dict)

                    step += 1

            cal_time = time.time() - start_time

            if metrics is not None and dataset_valid is not None:
                if epoch % eval_interval == 0:
                    dic = OrderedDict()
                    for metric in metrics:
                        if 'lr'in metric:
                            val = self.sess.run(self.get_metric(metric), feed_dict)
                        else:
                            val = self.eval_metric(metric, dataset_valid[:], **metric_kwargs)
                        if self._run is not None:
                            self._run.log_scalar(metric, float(val), step=epoch)
                        dic[metric] = val

                    if cal_time is not None:
                        dic['duration'] = cal_time

                    print(', '.join(['{}={}'.format(k, v) for k, v in dic.items()]))

            if dataset_valid is not None:
                gen_sample_images(dataset_valid, basename='step_{}.png'.format(epoch))

    def get_lr_feed_dict(self, step, scopes):
        if self.hypparams['lr_scheduler'] == 'ghadimi':
            return {self.lrs[scope]: self.hypparams['lrs'][scope] * (step+1) ** -0.5 for scope in scopes}
        elif self.hypparams['lr_scheduler'] == 'constant':
            return {self.lrs[scope]: self.hypparams['lrs'][scope] for scope in scopes}
        else:
            raise ValueError(self.hypparams['lr_scheduler'])

    def get_optimizers(self, scopes, epoch):
        return [self.optimizers[scope] for scope in scopes]

    def store_sgd_info(self, feed_dict, indices, step, optimizer_index, epoch):
        '''
        Store information for the counterfactual training

        Args:
            feed_dict: feed_dict of mini-batch
            indices: indices of mini-batch instances in the dataset
            step: number of the current training step
            optimizer_index: index of the optimizer used for the training
            epoch: number of the current epoch

        Returns: None

        '''
        raise NotImplementedError

    def get_metric(self, name, **kwargs):
        '''
        return tensor defined in the global score.
        name string is used to find the target tensor.

        Args:
            name: name of pre-defined tensor to be returned
            **kwargs:

        Returns: Tensor

        '''
        return tf.get_default_graph().get_tensor_by_name("{}:0".format(name))

    def eval_metric(self, name, feed_dict, **kwargs):
        '''
        get metric tensor using name string.
        then evaluate the tensor with given feed_dict to get numpy result

        Args:
            name: name of the tensor to be evaluated
            feed_dict: feed dict used for the evaluation
            **kwargs:

        Returns: Numpy scalar value

        '''
        metric_tensor = self.get_metric(name, **kwargs)
        return self.sess.run(metric_tensor, feed_dict)

    def add_asgd_influence_ops(self):
        raise NotImplementedError

    def get_classifier_ops(self, x, classifier_conf=None):
        '''
        append classifier ops to get logits or features of input x.

        Args:
            x: tensor of the input
            classifier_conf: dict of the hyper-parameters of classifier

        Returns:

        '''

        raise NotImplementedError

    def restore_opt_scopes(self, step):
        '''
        get list of names of scopes of original training at the given step

        Args:
            step: number of the current step

        Returns: scopes (list)

        '''
        return self.stored_info['scopes'][step]

    def get_opt_scopes(self, step):
        '''
        returns the list of names of the scopes of which parameters are trained

        Args:
            step: current number of step

        Returns: scopes (list)

        '''
        return self.scopes

    def gen_sample_images(self, dataset, basename, batch_size=64, **kwargs):
        '''
        generate and save samples.
        Generator type models overwrite method and Classifier type models use this to pass the generation.

        Args:
            dataset (modules.dataset.MyDataset): dataset that contains latent variables
            basename: basename of the saved image.
            batch_size: batch size of the generation
            **kwargs:

        Returns: None

        '''

        pass

    def sample_minibatch(self, dataset, minibatch_indices, step, scopes, *args, **kwargs):
        '''
        get minibatch feed dict given dataset and indices
        also sample latent variables and include them to feed dict
        learning rates are also determined according to the number of steps and included in the feed dict

        Args:
            dataset (modules.dataset.MyDataset): the whole training dataset
            minibatch_indices: indices in the dataset which construct the minibatch
            step: number of the current step
            scopes: scopes of optimizers to train
            *args:
            **kwargs:

        Returns: feed_dict (dict)

        '''
        feed_dict = dataset[minibatch_indices]
        feed_dict.update(self.get_lr_feed_dict(step, scopes))
        return feed_dict

    def sample_minibatch_by_storing_info(self, dataset, minibatch_indices, step, scopes, epoch):
        '''
        get minibatch feed dict and run self.store_sgd_info method

        Args:
            dataset (modules.dataset.MyDataset): the whole training dataset
            minibatch_indices: indices in the dataset which construct the minibatch
            step: number of the current step
            scopes: scopes of optimizers to train
            epoch: number of the current epoch

        Returns: feed_dict (dict)

        '''
        raise NotImplementedError

    def restore_minibatch(self, dataset, minibatch_indices, step, scopes, epoch):
        '''
        Get_items of instances from dataset given indices of the mini-batch to make a mini-bathch feed_dict.
        Then update feed_dict by stored latent variables (if saved) and learning rates

        Args:
            dataset (modules.dataset.MyDataset): the whole training dataset
            minibatch_indices: indices in the dataset which construct the minibatch
            step: number of the current step
            scopes: scopes of optimizers to train
            epoch: number of the current epoch

        Returns: feed_dict (dict)

        '''
        raise NotImplementedError

    def run_with_batches(self, fetch, dataset, batch_size=None):
        '''
        For memory efficiency, run fetch tensor with separated dataset.
        Then concatenate or average result.

        Args:
            fetch: tensor of which sess.run is computed
            dataset: dataset with which fetch tensor is computed
            batch_size: batch_size of a singel run

        Returns:
            numpy result of Session.run of fetch

        '''
        if batch_size is None or dataset.sample_size <= batch_size:
            return self.sess.run(fetch, dataset[:])

        else:
            assert dataset.sample_size % batch_size == 0
            if type(fetch) is list:
                out = zeros_from_arrays(fetch)
                update_func = lambda old, new, coef: add_arrays(old, mul_const_by_arrays(new, coef))
            else:
                shape = fetch._shape_as_list()
                if shape is None:
                    shape = [None]
                if None in shape:  # concatenate mode
                    assert shape[0] is None
                    out = None

                    def update_func(old, new, coef):
                        if old is None:
                            out = new
                        else:
                            out = np.concatenate([old, new])
                        return out
                else:  # update mode
                    out = np.zeros(shape)
                    update_func = lambda old, new, coef: out + new * coef

            idxs_of_minibatches = get_minibatch_indices(dataset.sample_size, batch_size, append_remainder=False, original_order=True)
            for idxs in idxs_of_minibatches:
                # calculate coefficient
                coef = batch_size / dataset.sample_size
                # evaluate
                tmp = self.sess.run(fetch, dataset[idxs])
                # update
                out = update_func(out, tmp, coef)

            return out
