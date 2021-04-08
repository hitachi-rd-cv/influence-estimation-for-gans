import numpy as np
from sklearn.ensemble import IsolationForest

def cal_isolation_forest_scores(X):
    clf = IsolationForest(random_state=0)
    clf.fit(X)
    return clf.score_samples(X)

def run_experiment(_run, option, model, dataset_train, metric_kwargs, train_dataset_dir):
    metric = option['metric']

    if metric == 'random':
        # random
        scores = np.random.rand(dataset_train.sample_size)

    elif metric == 'if':
        # isolation forest (The lower, the more abnormal.)
        _, _, features_real = model.get_classifier_ops(model.x, **metric_kwargs)
        features_train = model.run_with_batches(features_real, dataset_train, batch_size=model.batch_size_eval)
        scores = cal_isolation_forest_scores(features_train)

    elif metric == 'if_data':
        # isolation forest (The lower, the more abnormal.)
        x_flat = dataset_train[:][model.x].reshape([dataset_train.sample_size, -1])
        scores = cal_isolation_forest_scores(x_flat)

    else:
        raise ValueError(metric)

    return scores
