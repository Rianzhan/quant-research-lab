"""Read only bounded session dates from the already accepted calendar snapshot."""
from pathlib import Path
import pandas as pd
from src.data.cleaner import date_strings
from src.data.loader import digest
from src.data.pipeline import read_audit
from src.data.research_dataset import _inside
from src.data.research_policy import require
from src.utils.config import validate_config
from src.utils.research_rows import validate_calendar


def load_development_calendar(config: dict, root: Path) -> list[str]:
    """No network or market-price input; return domestic sessions in development."""
    validate_config(config); root=Path(root).resolve()
    source=read_audit(_inside(root,config['governance']['phase2_manifest']))
    entry=source['calendar_input']
    path=_inside(root,entry['path']+'/manifest.json')
    require(digest(path.read_bytes())==entry['manifest_sha256'],'Calendar manifest hash mismatch')
    metadata=read_audit(path)
    require(metadata['status']=='complete' and metadata['request']['source']=='akshare_calendar',
            'Expected a complete domestic calendar snapshot')
    raw=_inside(root,entry['path']+'/raw.csv')
    require(digest(raw.read_bytes())==metadata['raw_sha256'],'Calendar data hash mismatch')
    frame=pd.read_csv(raw,dtype=str,keep_default_na=False)
    require(list(frame.columns)==['trade_date'] and len(frame)==metadata['row_count'],'Invalid calendar schema/rows')
    dates=date_strings(frame.trade_date)
    require(dates.is_unique and dates.is_monotonic_increasing,'Invalid calendar ordering')
    bounded=dates.loc[dates.between(config['data']['start_date'],config['data']['end_date'])].tolist()
    return validate_calendar(bounded,config)
