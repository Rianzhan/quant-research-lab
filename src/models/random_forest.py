"""Small fixed random-forest baselines for both supervised tasks."""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def make_random_forest(task: str, config: dict) -> RandomForestClassifier | RandomForestRegressor:
    """Use registered complexity, deterministic seed and one worker."""
    model=RandomForestClassifier if task=='classification' else RandomForestRegressor
    return model(**config['models']['random_forest'],random_state=config['experiment']['seed'],n_jobs=config['models']['n_jobs'])
