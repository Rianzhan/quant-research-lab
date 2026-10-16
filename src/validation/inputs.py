"""Read accepted development artifacts without opening any sealed holdout prices."""
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import pandas as pd
from src.data.development_calendar import load_development_calendar
from src.data.loader import digest, json_bytes
from src.data.pipeline import read_audit
from src.data.research_dataset import _inside
from src.data.research_policy import require
from src.features.builder import feature_specs
from src.labels.forward_return import build_labels
from src.utils.config import config_digest
from src.utils.research_rows import KEYS, prepare_rows
from src.validation.config import validate_validation


def _frozen_code_hashes(commit: str) -> dict[str,str]:
    """Read registered Git objects, never execute old code or change HEAD/config."""
    root=Path(__file__).resolve().parents[2]
    git=shutil.which('git') or 'C:/Program Files/Git/cmd/git.exe'
    try:
        result=subprocess.run([git,'-C',str(root),'archive','--format=tar',commit,'src'],
                              capture_output=True,check=True)
    except (OSError,subprocess.CalledProcessError) as exc:
        raise ValueError('Cannot verify registered feature Git revision') from exc
    with tarfile.open(fileobj=BytesIO(result.stdout)) as archive:
        return {item.name:digest(archive.extractfile(item).read().replace(b'\r\n',b'\n'))
                for item in archive.getmembers() if item.isfile() and item.name.endswith('.py')}


def _read(path: Path, labels: bool=False) -> pd.DataFrame:
    """Parse only explicitly selected development files with round-trip floats."""
    dtype={'date':str,'asset_code':str,'exchange':str}
    if labels: dtype.update(label_end_date='string',classification_target='Int8',label_invalid_reason='string')
    return pd.read_csv(path,dtype=dtype,float_precision='round_trip',keep_default_na=False,na_values=[''])


def load_inputs(config: dict, policy: dict, root: Path) -> tuple[pd.DataFrame,list[str],dict]:
    """Validate the accepted Phase3 development chain; no mixed-price/raw replay."""
    validate_validation(policy,config);root=Path(root).resolve();source=policy['source']
    path=_inside(root,source['feature_manifest'])
    require(digest(path.read_bytes())==source['feature_manifest_sha256'],'Frozen feature manifest hash changed')
    meta=read_audit(path);binding=meta['bindings']
    require(meta['kind']=='development_features_labels' and meta['status']=='complete' and meta['formula_version']=='phase3-v1'
            and meta['scope']=='development_only' and meta['backtest_total_return_ready'] is False
            and meta['model_training_performed'] is False,'Unapproved Phase3 publication')
    require(meta['feature_parameters']==config['features'] and meta['prediction_horizon_days']==config['timing']['prediction_horizon_days']
            and meta['classification_return_threshold']==config['labels']['classification_return_threshold'],'Feature/label parameter mismatch')
    require(binding['config_sha256']==config_digest(config),'Frozen data config changed')
    require(binding['code_sha256']==_frozen_code_hashes(source['feature_code_commit']),'Historical feature code mismatch')
    require(set(meta['files'])=={'features.csv','labels.csv','sample_eligibility.csv','features_head20.csv','audit.json'},'Invalid Phase3 file set')
    for name,sha in meta['files'].items():
        require(digest((path.parent/name).read_bytes())==sha,'Frozen feature artifact hash changed: '+name)
    key=digest(json_bytes({'version':'phase3-v1','bindings':binding,'files':meta['files']}))
    require(key==meta['build_key'] and path.parent.name=='phase3-v1-'+key[:16],'Frozen feature build key mismatch')
    require(binding['research_manifest']==config['governance']['dataset_manifest'],'Development manifest path mismatch')
    research=_inside(root,binding['research_manifest'])
    require(digest(research.read_bytes())==binding['research_manifest_sha256'],'Development manifest hash changed')
    research_meta=read_audit(research)
    require(research_meta['kind']=='research_policy_dataset' and research_meta['status']=='complete'
            and research_meta['backtest_total_return_ready'] is False,'Unapproved development dataset')
    require(research_meta['config_sha256']==binding['config_sha256'] and research_meta['policy_sha256']==binding['policy_sha256']
            and research_meta['universe_sha256']==config['universe']['freeze_sha256'],'Frozen development policy/config mismatch')
    development=research.parent/'development.csv'
    require(digest(development.read_bytes())==binding['development_sha256']==research_meta['files']['development.csv'],
            'Development prices hash changed')
    policy_path=_inside(root,config['governance']['policy_file'])
    require(digest(policy_path.read_bytes())==binding['policy_sha256']==config['governance']['policy_sha256'],'Research policy hash changed')
    phase2=_inside(root,config['governance']['phase2_manifest'])
    require(digest(phase2.read_bytes())==research_meta['phase2_manifest_sha256'],'Calendar source manifest changed')
    phase2_meta=read_audit(phase2)
    require(phase2_meta['calendar_input']['manifest_sha256']==binding['calendar_manifest_sha256'],'Calendar binding changed')
    calendar=load_development_calendar(config,root)
    observed=prepare_rows(_read(development),config,calendar)
    require(len(observed)==research_meta['rows']['development'],'Development row count changed')
    features=_read(path.parent/'features.csv');labels=_read(path.parent/'labels.csv',labels=True)
    names=[s['name'] for s in feature_specs(config)]
    require(meta['predictors']==names and list(features.columns)==KEYS+names,'Feature whitelist mismatch')
    require(features[KEYS].equals(observed[KEYS]) and labels[KEYS].equals(observed[KEYS]),'Feature/label/development keys misaligned')
    expected=build_labels(observed,config,calendar)
    labels['label_invalid_reason']=labels.label_invalid_reason.fillna('')
    try:
        pd.testing.assert_frame_equal(labels,expected,check_dtype=False,check_exact=True)
    except AssertionError as exc:
        raise ValueError('Frozen label semantics inconsistent with development calendar/prices') from exc
    for period in (5,20):
        require(features[f'return_{period}d'].equals(features[f'momentum_{period}d']),'Registered duplicate features differ')
    frame=features.copy();frame['research_close']=observed.research_close
    for column in labels.columns:
        if column not in KEYS: frame[column]=labels[column]
    current_binding={'feature_manifest':source['feature_manifest'],'feature_manifest_sha256':source['feature_manifest_sha256'],
        'feature_code_commit':source['feature_code_commit'],'development_sha256':binding['development_sha256'],
        'policy_sha256':binding['policy_sha256'],'calendar_manifest_sha256':binding['calendar_manifest_sha256'],
        'config_sha256':binding['config_sha256'],'holdout_files_opened':False}
    return frame,calendar,current_binding
