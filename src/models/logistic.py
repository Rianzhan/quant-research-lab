"""Logistic classification with training-only standardization."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic(config: dict) -> Pipeline:
    """Do not set deprecated penalty/n_jobs arguments in sklearn1.9."""
    estimator=LogisticRegression(**config['models']['logistic'],random_state=config['experiment']['seed'])
    return Pipeline([('scaler',StandardScaler()),('model',estimator)])
