import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_probability as tfp
# from tensorflow.contrib.layers import instance_norm as norm
from tensorflow.contrib.layers import layer_norm as norm
from tqdm import tqdm

from models import Generator
from modules.tf_ops import inception_score, log_inception_score, generator_loss, discriminator_loss, influence_loss, \
    matmul_arrays, partial_hessian_vector_product
from modules.utils import normalize_lists, merge_dicts


class GANBase(Generator):
    name = 'generator'
    scopes = 'discriminator', 'generator'
    scope_suffixes = '_d', '_g'
    batch_size_eval = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_stored_info()

    def generator(self, z, weight_decay, use_bn, reuse=False):
        '''
        construct computation graph between latent variable z and generated input

        Args:
            z (Tensor): tensor or placeholder of the latent variable
            weight_decay: coefficient of the weight regularization decay
            use_bn: when True, use Layer Normalization
            reuse: when True, it reuses pre-defines variables

        Returns: x_gen (Tensor): generated input which shape is same as the input

        '''
        raise NotImplementedError

    def discriminator(self, x, weight_decay, use_bn, reuse=False):
        '''
        construct computational graph between the true or generated input between logits

        Args:
            x (Tensor): tensor or placeholder of true or generated inputs
            weight_decay (Tensor or Float): coefficient of the weight regularization decay
            use_bn (Boolean): when True, use Layer Normalization
            reuse (Boolean): when True, it reuses pre-defines variables

        Returns: logits: scalar

        '''
        raise NotImplementedError

    def get_logits_and_x_gen(self, x, z):
        '''
        call generator and discriminator to gent logits and generated inputs

        Args:
            x: tensor of inputs
            z: tensor of latent variables

        Returns:
            logits_real (Tensor): logits of discriminator given real inputs
            logits_gen (Tensor): logits of the discriminator given generated inputs
            x_gen (Tensor): generated inputs

        '''
        logits_real = self.discriminator(x, self.hypparams['weight_decay_d'], self.hypparams['use_bn_d'])
        x_gen = self.generator(z, self.hypparams['weight_decay_g'], self.hypparams['use_bn_g'])
        logits_gen = self.discriminator(x_gen, self.hypparams['weight_decay_d'], self.hypparams['use_bn_d'], reuse=True)
        return logits_real, logits_gen, x_gen

    def build(self, x, z, *args, **kargs):
        '''
        See overwritten method in NNBase for docs

        '''
        logits_real, logits_gen, x_gen = self.get_logits_and_x_gen(x, z)

        optimizers_, lrs, bare_losses, reg_losses, losses, vars = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        for scope, suffix in zip(self.scopes, self.scope_suffixes):
            if scope == 'discriminator':
                bare_loss = discriminator_loss(logits_real, logits_gen, self.hypparams['loss_method'],
                                               'loss_bare' + suffix)
            elif scope == 'generator':
                bare_loss = generator_loss(logits_gen, self.hypparams['loss_method'], 'loss_bare' + suffix)
            else:
                raise ValueError

            reg_ops = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope) or tf.constant(0., self.dtype)
            reg_loss = tf.reduce_sum(reg_ops, name='loss_reg' + suffix)
            loss = tf.add(bare_loss, reg_loss, name='loss' + suffix)

            lr = tf.placeholder(tf.float32, None, name='lr' + suffix)
            optimizer_ = tf.compat.v1.train.GradientDescentOptimizer(lr, name='sgd' + suffix)

            var = [v for v in tf.trainable_variables() if scope in v.name]

            optimizers_[scope] = optimizer_
            lrs[scope] = lr
            bare_losses[scope] = bare_loss
            reg_losses[scope] = reg_loss
            losses[scope] = loss
            vars[scope] = var

        if self.hypparams['sim']:
            # applying grads must wait for gradient calculation of both g and d variables
            grads_and_vars = {scope: optimizers_[scope].compute_gradients(losses[scope], vars[scope]) for scope in
                              self.scopes}
            normalized_grads = [x[0] for x in normalize_lists(grads_and_vars.values())]
            with tf.control_dependencies(normalized_grads):
                opt_d = optimizers_['discriminator'].apply_gradients(grads_and_vars['discriminator'])
                with tf.control_dependencies([opt_d]):
                    opt_g = optimizers_['generator'].apply_gradients(grads_and_vars['generator'])
                    optimizers = OrderedDict((('discriminator', tf.no_op()), ('generator', opt_g)))
        else:
            optimizers = OrderedDict(
                (scope, optimizers_[scope].minimize(losses[scope], var_list=vars[scope])) for scope in self.scopes)

        self.x_gen = x_gen
        self.logits_real = logits_real
        return losses, optimizers, vars, lrs, reg_losses, bare_losses

    def sample_minibatch(self, dataset, minibatch_indices, step, scopes, *args, **kwargs):
        '''
        See overwritten method in NNBase for docs
        '''
        z = np.random.normal(size=[len(minibatch_indices), self.hypparams['z_dim']])
        feed_dict = dataset[minibatch_indices]
        feed_dict.update({self.z: z})
        feed_dict.update(self.get_lr_feed_dict(step, scopes))
        return feed_dict

    def sample_minibatch_by_storing_info(self, dataset, minibatch_indices, step, scopes, epoch):
        '''
        See overwritten method in NNBase for docs
        '''
        feed_dict = self.sample_minibatch(dataset, minibatch_indices, step, scopes)
        self.store_sgd_info(feed_dict, minibatch_indices, step, scopes, epoch)
        return feed_dict

    def restore_minibatch(self, dataset, minibatch_indices, step, scopes, epoch):
        '''
        See overwritten method in NNBase for docs
        '''
        feed_dict = dataset[minibatch_indices]
        feed_dict.update({self.z: self.stored_info['z'][step]})
        feed_dict.update({self.lrs[scope]: self.stored_info['lrs'][step][scope] for scope in scopes})
        return feed_dict

    def init_stored_info(self):
        '''
        See overwritten method in NNBase for docs
        '''
        # initialize attributes
        self.stored_info = {}
        self.stored_info['indices'] = []
        self.stored_info['params'] = []
        self.stored_info['lrs'] = []
        self.stored_info['scopes'] = []
        self.stored_info['step'] = []
        self.stored_info['epoch'] = []
        self.stored_info['z'] = []

    def store_sgd_info(self, feed_dict, indices, step, scopes, epoch):
        '''
        See overwritten method in NNBase for docs
        '''
        if epoch >= self.hypparams['retrace_after']:
            self.stored_info['indices'].append(indices)
            self.stored_info['params'].append(self.saver.get_current_vars())
            self.stored_info['lrs'].append(OrderedDict((scope, feed_dict[self.lrs[scope]]) for scope in scopes))
            self.stored_info['scopes'].append(scopes)
            self.stored_info['step'].append(step)
            self.stored_info['epoch'].append(epoch)
            self.stored_info['z'].append(feed_dict[self.z])

    def get_inseption_score(self, classifier_conf, log=False):
        '''
        append classifier ops to get class probabilities.
        then obtain the tensor of (log) inception score

        Args:
            classifier_conf: dict of the hyper-parameters of classifier
            log: if True, it returns log inception score

        Returns:
            (log)_inception_score (Tensor): a scalar tensor of (log) inception score

        '''
        preds, _, _ = self.get_classifier_ops(self.x_gen, classifier_conf)
        if log:
            return log_inception_score(preds)
        else:
            return inception_score(preds)

    def get_frechet_classifier_distance(self, classifier_conf):
        '''
        append classifier ops to get features of true and generated x
        then obtain FID tensor between two features

        Args:
            classifier_conf: dict of the hyper-parameters of classifier

        Returns:
            fid (Tensor): a scalar tensor of FID

        '''
        _, _, features_real = self.get_classifier_ops(self.x, classifier_conf)
        _, _, features_gen = self.get_classifier_ops(self.x_gen, classifier_conf)
        fid = tfgan.eval.frechet_classifier_distance_from_activations(features_real, features_gen)
        return fid

    def get_metric(self, name, classifier_conf=None, n=None, **kwargs):
        '''
        return tensor according to the name

        Args:
            name: name of the tensor or metric
            classifier_conf: dict of the hyper-parameters of classifier. required when name in ['(log)_inception_score', 'fid']
            n: number of samples. required when name == 'log_likelihood_kde'
            **kwargs:

        Returns: Tensor

        '''
        if name == 'inception_score':
            return self.get_inseption_score(classifier_conf)
        elif name == 'log_inception_score':
            return self.get_inseption_score(classifier_conf, log=True)
        elif name == 'log_likelihood_kde':
            return self.get_log_likelihood_kde(n=n)
        elif name == 'fid':
            return self.get_frechet_classifier_distance(classifier_conf)
        elif name == 'weight_sum':
            return tf.reduce_sum([tf.reduce_sum([tf.reduce_sum(y) for y in x]) for x in self.vars])
        else:
            return super().get_metric(name=name, classifier_conf=classifier_conf, **kwargs)

    def get_log_likelihood_kde(self, n):
        '''
        return tensor of Average Log-likelihood

        Args:
            n (int): number of samples

        Returns: Tensor of the average log-likelihood

        '''
        f = lambda x: tfp.distributions.Independent(tfp.distributions.Normal(
            loc=x, scale=tf.constant(1., self.dtype)))
        kde = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=tf.ones(n, dtype=self.dtype) / n),
            components_distribution=f(self.x_gen))
        return tf.reduce_mean(kde.log_prob(self.x), name='log_likelihood')

    def eval_metric(self, name, feed_dict, **kwargs):
        '''
        See overwritten method in NNBase for docs
        '''
        # number of samples is required when computing average log-likelihood
        n = feed_dict[self.z].shape[0]
        return super().eval_metric(name, feed_dict, **kwargs, n=n)

    def dense(self, x, units, activation, weight_decay=0., use_bn=False, name=None, **kwargs):
        out = tf.layers.dense(x,
                              units,
                              activation=activation,
                              # kernel_initializer=tf.initializers.truncated_normal(stddev=1e-2),
                              # kernel_initializer=tf.initializers.random_uniform(-bound, bound),
                              # bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                              name=name,
                              **kwargs)
        if use_bn:
            out = norm(out)

        return out

    def conv2d(self, input_, output_dim, ks, s, activation, use_bn, padding='same', name="conv2d", weight_decay=0.,
               use_bias=True):
        with tf.variable_scope(name):
            out = tf.layers.conv2d(input_, output_dim, ks, [s] * 2, padding=padding, activation=activation,
                                   use_bias=use_bias,
                                   # bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
            if use_bn:
                out = norm(out)

        return out

    def deconv2d(self, input_, output_dim, ks, s, activation, use_bn, padding='same', name="deconv2d", weight_decay=0.,
                 use_bias=True):
        with tf.variable_scope(name):
            out = tf.layers.conv2d_transpose(input_, output_dim, ks, [s] * 2, padding=padding, activation=activation,
                                             use_bias=use_bias,
                                             # bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
            if use_bn:
                out = norm(out)

        return out

    def get_opt_scopes(self, step):
        '''
        See overwritten method in NNBase for docs
        '''
        if self.hypparams['sim']:
            return self.scopes
        else:
            return [self.scopes[step % 2]]

    def get_jacobian_vector_prod(self):
        '''
        obtain the computational graph of Jacobian vector product between across multiple losses and multiple set of parameters.

        Returns:
            hvp (dict): dict of vector tensor of hessian vector product. key is the scope of the parameter (e.g., "discriminator") on that parameters the gradients are computed.

        '''
        jvps = OrderedDict()
        for scope, scope_ad in zip(self.scopes, self.scopes[::-1]):
            # partial_Hessian @ u = d/(d*partial_params) (d*loss/(d*all_params) @ u)
            jvp_diag = partial_hessian_vector_product(self.losses[scope], self.vars[scope], self.vars[scope],
                                                      self.u_phs[scope])
            jvp_nondiag = partial_hessian_vector_product(self.losses[scope_ad], self.vars[scope_ad], self.vars[scope],
                                                         self.u_phs[scope_ad])
            jvps[scope] = [x + y for x, y in zip(jvp_diag, jvp_nondiag)]

        return jvps

    def get_influence(self):
        '''
        obtain tensor of influence on something.
        it is given by inner product between u and gradient of the discriminator of j-th instance
        here, it only define the gradient operation and inner product between the gradients and placeholder of discriminator part of u

        Returns: influence (Tensor): a scalar tensor

        '''
        grads = tf.gradients(influence_loss(self.logits_real, self.hypparams['loss_method'], 'influence_loss'),
                             self.vars['discriminator'])
        influence = matmul_arrays(grads, self.u_phs['discriminator'])

        return influence

    def add_asgd_influence_ops(self):
        '''
        define computational graph of the Jacobian vector product (jvp) and influence.
        For linear influence, placeholder dict u_phs which has same shape as variables are defined.

        Returns: None

        '''
        self.u_phs = OrderedDict()
        for scope, var in self.vars.items():
            self.u_phs[scope] = [tf.placeholder(self.dtype, w.get_shape(), w._shared_name) for w in var]

        # assigning the variable of the step at which you want to calculate the jvps and influence
        with tf.control_dependencies(self.saver.assign_var_list):
            self.jvps = self.get_jacobian_vector_prod()
            self.influence = self.get_influence()

    def cal_asgd_influence(self, dataset_train, dataset_valid, metric, converge_check=False, target_indices=None,
                           damping=0.):
        '''
        calculate influence on metric

        Args:
            dataset_train (modules.dataset.MyDataset): training dataset
            dataset_valid (modules.dataset.MyDataset): validation dataset
            metric (str): metric operation name for which influence is computed
            converge_check: when True, it does not influence but Jacobian vector products
            target_indices: indices of instances in the training dataset of which influence on metric is computed. if None it calculates influece of all the training indices.
            damping: not used.

        Returns: array of values of influence on metrics

        '''
        influences = [0.] * dataset_train.sample_size  # initialize influence on the metric
        if target_indices is None:
            target_indices_ = np.arange(dataset_train.sample_size)
        else:
            if len(target_indices) > 0:
                target_indices_ = target_indices
            else:
                return influences  # when target_indices are empty it return zeros influences

        assert len(self.stored_info['step']) == \
               len(self.stored_info['params']) == \
               len(self.stored_info['indices']) == \
               len(self.stored_info['lrs']) == \
               len(self.stored_info['scopes']) == \
               len(self.stored_info['z'])

        # gradient of metrics with respect to parameters. note that parameters are contained in the scope-key dictionary
        gradient_ops_maybe_with_nones = OrderedDict(
            (scope, tf.gradients(metric, var)) for scope, var in self.vars.items())
        # when metric is not differentiable with respect to some parameters, it has None gradient. so replace None with zeros array
        gradient_ops = OrderedDict(
            (scope, [g if g is not None else tf.zeros_like(w) for g, w in zip(gs, var)]) for scope, gs, var in
            zip(self.scopes, gradient_ops_maybe_with_nones.values(), self.vars.values()))
        # compute gradient of metrics to initialize u
        u = OrderedDict((scope, self.run_with_batches(op, dataset_valid, self.batch_size_eval)) for scope, op in
                        gradient_ops.items())

        step_indices = np.arange(len(self.stored_info['step']))
        # traces back the steps from the latest step
        for step_index in tqdm(step_indices[::-1], ascii=True):
            step = self.stored_info['step'][step_index]
            indices = self.stored_info['indices'][step_index]
            feed_dict_batch = dataset_train[indices]

            if self.z is not None:  # if GAN
                assert self.z not in feed_dict_batch
                stored_z = self.stored_info['z'][step_index]
                feed_dict_batch.update({self.z: stored_z})

            u_copied = deepcopy(u)  # just in case u is overwritten before update of u
            feed_dict_u = {}
            for u_placeholder, u_ in zip(self.u_phs.values(), u_copied.values()):
                feed_dict_u.update({key: value for key, value in zip(u_placeholder, u_)})

            # feed dict for restoring the parameters
            params = self.stored_info['params'][step_index]
            feed_dict_weight_assign = self.saver.get_feed_dict(params)

            lrs = self.stored_info['lrs'][step_index]
            scopes = self.stored_info['scopes'][step_index]

            # make sure the number of trained param scoes is 2 for simultaneous training and 1 for alternate training.
            if hasattr(self, 'sim'):
                assert (len(scopes) == 1 and not self.hypparams['sim']) or (len(scopes) == 2 and self.hypparams['sim'])

            if not converge_check and 'discriminator' in scopes:
                # influence update
                for j in indices:
                    if j in target_indices_:
                        index_in_batch = np.where(indices == j)[0]
                        # update influence of all variables
                        # feed only jth sample by masking jth z
                        feed_dict_of_j = {self.x: feed_dict_batch[self.x][index_in_batch]}
                        influence_tmp = self.sess.run(self.influence,
                                                      feed_dict=merge_dicts((feed_dict_of_j,
                                                                             feed_dict_weight_assign,
                                                                             feed_dict_u)))
                        # divide by batch size as 1 sample loss lacks 1/|St|
                        influence = influence_tmp / self.hypparams['batch_size']
                        assert not np.isnan(
                            influence), f'Met NaN influence: t={step}, j={j}, influence_diff={influence}'
                        influences[j] += influence * lrs['discriminator']

            # cal Jacobian vector product
            jvps = self.sess.run({scope: self.jvps[scope] for scope in scopes},
                                 feed_dict=merge_dicts((feed_dict_batch,
                                                        feed_dict_weight_assign,
                                                        feed_dict_u)))

            assert len(scopes) == len(jvps)
            for scope, jvp in jvps.items():
                for i, jvp_ in enumerate(jvp):
                    u[scope][i] = (1 - damping) * u[scope][i] - jvp_ * lrs[scope]

                logging.debug(f't={step:<5}, scope={scope:<15}, u_diff_mean={np.mean([np.mean(x) for x in jvp]):.05}')
                if converge_check:
                    print(f't={step}, opt_index={scope}, u_abs_sum={np.mean([np.mean(np.abs(uu)) for uu in u[scope]])}')

        return influences
