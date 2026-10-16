"""Strict, whitelist-bound model interface without data splitting or target access."""
import math
import warnings
from typing import Any
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data.research_policy import require

FORBIDDEN={'date','exchange','asset_code','price_segment','label_end_date','label_valid',
           'classification_target','regression_target','future_return_5d','supervised_eligible'}


def json_parameters(values: dict) -> dict:
    """Represent nonfinite estimator defaults explicitly, not as invalid JSON."""
    def convert(value):
        if isinstance(value,np.generic): value=value.item()
        if isinstance(value,float) and not math.isfinite(value): return str(value)
        if isinstance(value,(list,tuple)): return [convert(item) for item in value]
        if isinstance(value,dict): return {key:convert(item) for key,item in value.items()}
        require(value is None or isinstance(value,(str,bool,int,float)),'Unsupported estimator parameter type')
        return value
    return convert(values)


class ResearchModel:
    """One freshly constructed estimator per fold, scope and task.

    fit sees only aligned training X/y. predict has no label or fit option.
    Training warnings fail the run rather than silently publishing a bad fit.
    """

    def __init__(self, estimator: Any, family: str, task: str, feature_names: list[str]):
        require(len(feature_names)==17 and len(set(feature_names))==17
                and not set(feature_names)&FORBIDDEN,'Expected 17 distinct predictor columns, no targets/identity')
        self.estimator=estimator
        self.family=family
        self.task=task
        self.feature_names=tuple(feature_names)
        self.fitted=False

    def _validate_X(self, X: pd.DataFrame, *, allow_empty: bool=False) -> None:
        require(isinstance(X,pd.DataFrame) and list(X.columns)==list(self.feature_names),
                'Predictor names and order must exactly match the registered whitelist')
        require(X.index.is_unique and (allow_empty or len(X)>0),'Missing rows or duplicate row index')
        require(all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes),'Predictors must have numeric types')
        require(np.isfinite(X.to_numpy(dtype=float,na_value=np.nan)).all(),'Predictors must be finite; no imputation')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ResearchModel':
        """Fit once on explicitly supplied training rows; never infer a split."""
        require(not self.fitted,'Create a fresh model for each fit')
        self._validate_X(X)
        require(isinstance(y,pd.Series) and y.index.equals(X.index),'Training target index must match X')
        require(pd.api.types.is_numeric_dtype(y.dtype),'Targets must have numeric types')
        values=y.to_numpy(dtype=float,na_value=np.nan)
        require(np.isfinite(values).all(),'Training targets must be finite')
        if self.task=='classification':
            require(set(values)=={0.,1.},'Classification training requires both classes 0 and 1')
            values=values.astype(int)
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter('error')
            self.estimator.fit(X,values)
        self.fitted=True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class-1 probabilities or continuous returns without thresholds."""
        require(self.fitted,'Model must be fitted before prediction')
        self._validate_X(X,allow_empty=True)
        if X.empty: return np.empty(0,dtype=float)
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter('error')
            if self.task=='classification':
                classes=list(self.estimator.classes_)
                result=self.estimator.predict_proba(X)[:,classes.index(1)]
            else:
                result=self.estimator.predict(X)
        result=np.asarray(result,dtype=float)
        require(result.shape==(len(X),) and np.isfinite(result).all(),'Invalid prediction shape/value')
        if self.task=='classification':
            require(((result>=0)&(result<=1)).all(),'Classification probability outside [0,1]')
        return result

    def metadata(self) -> dict:
        """Expose effective parameters and train-fitted transform statistics."""
        estimator=self.estimator.named_steps['model'] if self.family=='logistic' else self.estimator
        result={'model':self.family,'task':self.task,'feature_names':list(self.feature_names),
                'parameters':json_parameters(estimator.get_params(deep=False)),
                'preprocessing':'train_only_standard_scaler' if self.family=='logistic' else 'none'}
        if self.fitted and self.family=='logistic':
            scaler=self.estimator.named_steps['scaler']
            result['scaler']={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                              'variance':scaler.var_.tolist(),'samples_seen':int(scaler.n_samples_seen_)}
        return result
