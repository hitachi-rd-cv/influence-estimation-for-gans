import numpy as np

def run_experiment(_run, option, nepochs, weight_dir, model, dataset_train, dataset_valid, train_kwargs, fetches,
                   metric_kwargs):
    # influence approximation
    if option['ncfsgd'] >= 0:
        selected_indices = np.arange(dataset_train.sample_size)[:option['ncfsgd']]
    else:
        selected_indices = np.arange(dataset_train.sample_size)

    print('Constructing Linear Influence operations ... ')
    model.add_asgd_influence_ops()

    print('Start approximating influences ... ')
    metric_tensor = model.get_metric(option['metric'], **metric_kwargs, n=dataset_valid.sample_size)
    approx_diffs = model.cal_asgd_influence(dataset_train, dataset_valid, metric_tensor, target_indices=selected_indices, **option['infl_args'])

    approx_diffs_selected = np.asarray(approx_diffs)[selected_indices]

    return approx_diffs_selected
