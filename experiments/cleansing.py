import numpy as np

def run_experiment(_run, option, nepochs, weight_dir, model, dataset_train, dataset_valid, dataset_test, train_kwargs, fetches,
                   metric_kwargs, approx_diffs, train_dataset_dir):

    if option['metric'] in ['inception_score', 'log_inception_score', 'log_likelihood', 'log_likelihood_kde', 'random']:
        larger_is_more_harmful = True
    elif option['metric'] in ['fid', 'loss_d', 'if', 'if_data']:
        larger_is_more_harmful = False
    else:
        raise ValueError('invalid gan evaluation metric: {}'.format(option['metric']))

    n_tr = dataset_train.sample_size
    remove_size = int(n_tr*option['removal_rate'])

    ## asgd
    scores = np.asarray(approx_diffs)
    if larger_is_more_harmful:
        remove_indices = np.argsort(-scores)[:remove_size]
        harmful_scores = scores
    else:
        remove_indices = np.argsort(scores)[:remove_size]
        harmful_scores = -scores

    model.sgd_train(dataset_train,
                    nepochs=nepochs,
                    mode='counterfactual',
                    quiet=True,
                    j=remove_indices,
                    **train_kwargs,
                    metric_kwargs=metric_kwargs)
    model.dump_harmers(dataset=dataset_train, harmers_idxs=remove_indices, harmful_scores=harmful_scores)

    return

