"""One construction path for all registered model/task combinations."""
from src.data.research_policy import require
from src.models.base import ResearchModel
from src.models.logistic import make_logistic
from src.models.random_forest import make_random_forest
from src.models.xgboost_model import make_xgboost
from src.utils.config import validate_config

MODEL_TASKS=[{'model':'logistic','task':'classification'},
             {'model':'random_forest','task':'classification'},{'model':'random_forest','task':'regression'},
             {'model':'xgboost','task':'classification'},{'model':'xgboost','task':'regression'}]


def create_model(family: str, task: str, config: dict, feature_names: list[str]) -> ResearchModel:
    """Reject unsupported combinations and bind the model's exact input order."""
    validate_config(config)
    require({'model':family,'task':task} in MODEL_TASKS,'Unsupported model/task combination')
    require(config['models']['n_jobs']==1,'Phase5 requires one model worker')
    if family=='logistic': estimator=make_logistic(config)
    elif family=='random_forest': estimator=make_random_forest(task,config)
    else: estimator=make_xgboost(task,config)
    return ResearchModel(estimator,family,task,feature_names)
