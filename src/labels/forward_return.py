"""Calendar-anchored future targets; entirely separate from predictor construction."""
import numpy as np
import pandas as pd
from src.data.research_policy import require
from src.utils.research_rows import ASSET, KEYS, prepare_rows


def build_labels(panel: pd.DataFrame, config: dict, calendar: list[str]) -> pd.DataFrame:
    """Price return at the h-th future domestic session, within one action segment.

    A missing quote at the exact target session stays invalid, not rolled forward.
    The future endpoint and validity metadata must never be used as predictors.
    """
    rows=prepare_rows(panel,config,calendar)
    horizon=config['timing']['prediction_horizon_days']
    column=f'future_return_{horizon}d'
    endpoints={day: calendar[i+horizon] if i+horizon<len(calendar) else None for i,day in enumerate(calendar)}
    result=rows[KEYS].copy()
    result['label_end_date']=rows.date.map(endpoints).astype('string')
    result[column]=np.nan
    result['label_valid']=False
    result['label_invalid_reason']=''
    for _,group in rows.groupby(ASSET,sort=False):
        end=result.loc[group.index,'label_end_date']
        observations=group.set_index('date')
        target_close=end.map(observations.close)
        target_segment=end.map(observations.price_segment)
        valid=end.notna() & target_close.notna() & target_segment.eq(group.price_segment)
        result.loc[group.index,'label_invalid_reason']=np.select(
            [end.isna().to_numpy(),target_close.isna().to_numpy(),target_segment.ne(group.price_segment).to_numpy()],
            ['horizon_outside_development_calendar','target_quote_missing','corporate_action_boundary'],default='')
        result.loc[group.index,'label_valid']=valid
        result.loc[group.index,column]=(target_close/group.close-1).where(valid)
    require(not np.isinf(result[column].to_numpy()).any(),'Nonfinite label overflow')
    result['regression_target']=result[column]
    result['classification_target']=pd.Series(pd.NA,index=result.index,dtype='Int8')
    valid=result.label_valid
    result.loc[valid,'classification_target']=(result.loc[valid,column]>config['labels']['classification_return_threshold']).astype('Int8')
    return result
