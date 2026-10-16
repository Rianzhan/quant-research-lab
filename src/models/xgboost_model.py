"""Fixed CPU XGBoost baselines, no eval_set or early stopping."""
from xgboost import XGBClassifier, XGBRegressor


def make_xgboost(task: str, config: dict) -> XGBClassifier | XGBRegressor:
    """Explicit objectives and fixed rounds, never a test-driven stopping rule."""
    model=XGBClassifier if task=='classification' else XGBRegressor
    objective='binary:logistic' if task=='classification' else 'reg:squarederror'
    return model(**config['models']['xgboost'],random_state=config['experiment']['seed'],
                 n_jobs=config['models']['n_jobs'],objective=objective,verbosity=0)
