import numpy as np

def run_experiment(_run, option, nepochs, weight_dir, model, dataset_train, dataset_valid, train_kwargs, fetches,
                   metric_no_removal, metric_kwargs):
    if option['ncfsgd'] > 0:
        selected_indices = np.arange(dataset_train.sample_size)[:option['ncfsgd']]
    else:
        selected_indices = np.arange(dataset_train.sample_size)

    # actual retraining
    print('Start actual counterfactual SGD ... ')
    actual_diffs_selected = []
    for i, j in zip(range(len(selected_indices)), selected_indices):
        ## train
        model.sgd_train(dataset_train,
                        nepochs=nepochs,
                        mode='counterfactual',
                        quiet=True,
                        j=j,
                        **train_kwargs,
                        metric_kwargs=metric_kwargs)
        metric_cf = model.eval_metric(option['metric'], dataset_valid[:], **metric_kwargs)
        metric_diff = metric_cf - metric_no_removal
        actual_diffs_selected.append(metric_diff)
        print('[{}/{}] j: {}, actual_diff: {}'.format(i + 1,
                                                      len(selected_indices),
                                                      j,
                                                      metric_diff))
    actual_diffs_selected = np.asarray(actual_diffs_selected)

    return actual_diffs_selected
