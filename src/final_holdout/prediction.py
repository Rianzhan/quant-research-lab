"""Label-independent point-in-time inference: exact numeric predictors only."""
import numpy as np
import pandas as pd
from src.backtest.contracts import require
from src.final_holdout.training import FEATURES
from src.models.base import ResearchModel


def predict_features(model: ResearchModel | None, features: pd.DataFrame) -> pd.DataFrame:
    """Return a score or explicit missing reason; no target, date or fallback API.

    The caller routes an asset to its frozen model and supplies only features
    calculated up to its information cutoff. Missing labels are not an input.
    """
    require(isinstance(features,pd.DataFrame) and tuple(features.columns)==FEATURES,
        'Exactly the ordered 17 features required; labels/future metadata forbidden')
    require(features.index.is_unique,'Duplicate inference index')
    require(all(pd.api.types.is_numeric_dtype(d) for d in features.dtypes),'Nonnumeric feature')
    values=features.to_numpy(dtype=float,na_value=np.nan)
    require(not np.isinf(values).any(),'Infinite inference feature')
    result=pd.DataFrame({'prediction':np.nan,'reason':'model_unavailable'},index=features.index)
    if model is None:return result
    require(model.fitted and model.feature_names==FEATURES,'Unfitted or wrong-feature model')
    require((model.family,model.task) in {('logistic','classification'),('random_forest','regression')},'Unregistered model')
    valid=np.isfinite(values).all(axis=1);result['reason']='feature_incomplete'
    result.loc[valid,'prediction']=model.predict(features.loc[valid])
    result.loc[valid,'reason']='available'
    return result
