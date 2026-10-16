"""Exact pinned development inputs; no raw calendar or mixed market file opens."""
from contextlib import contextmanager
from dataclasses import fields
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import json
from pathlib import Path
import pandas as pd
import yaml
from src.backtest.contracts import Action, Asset, Quote, day, require
from src.backtest.inputs import read_policy
from src.governance import final_holdout as governance
from src.portfolio.rules import project_predictions
from src.utils.config import _UniqueSafeLoader

SOURCE_SHA='86890e0bd036f476e5c0bafe2c7bc7f919702ec65ff564e07a5b66d915c093fa'
SOURCE='config/phase9b_sources.json'
P3='data/processed/features/phase3-v1-63d97b331713ad2d/'
P45='data/processed/features/phase45-v1-0a9eb107fb50d0b5/'
P7='results/reports/phase7-v1-34d765af3b0494e9/'
P65='results/reports/phase6a5-v1-0d29f2d0ea092448/'
ROLES={'panel':'data/processed/research/phase25-v1/development.csv',
    'features':P45+'features.csv','raw_features':P3+'features.csv','labels':P3+'labels.csv',
    'eligibility':P45+'sample_eligibility.csv',
    'calendar':P65+'cases/full_development_manual_accounting/daily_ledger.csv',
    'historical_calendar':P7+'accounts/C_fee_0/daily_ledger.csv',
    'actions':P65+'corporate_actions.json','accounting_gates':P65+'gates.json',
    'predictions':'data/processed/models/phase5-v1-5f899c5bc2f44f86/predictions.csv',
    'account_hashes':P7+'account_hashes.json'}
SAFE_CONFIG={'config/'+n for n in ['config.yaml','execution.yaml','phase7.json','scale_free.yaml',
    'strategy_sources.json','strategy_spec.json','validation.yaml']}
SAFE_MANIFESTS={prefix+'manifest.json' for prefix in [P3,P45,P65,P7,
    'data/processed/models/phase5-v1-5f899c5bc2f44f86/','data/processed/research/phase25-v1/']}
ACCOUNT_NAMES={'cancelled_unfilled_orders.csv','corporate_action_journal.csv','costs.csv',
    'daily_ledger.csv','decisions.csv','dividend_receivable_ledger.csv','event_journal.csv','fills.csv',
    'holding_count_distribution.csv','holdings_ledger.csv','order_provenance.json','orders.csv',
    'signals.csv','summary.json','turnover.csv'}
SAFE_INPUTS=set(ROLES.values())|SAFE_CONFIG|SAFE_MANIFESTS|{
    P7+'accounts/'+s+'_fee_'+fee+'/'+name for s in 'ABC'
    for fee in ['0','0p0003','0p0005','0p001'] for name in ACCOUNT_NAMES}


class PinnedReader(governance.PinnedReader):
    """A caller cannot extend the accepted payload set by adding a new hash."""
    def read(self,name: str) -> bytes:
        require(name in SAFE_INPUTS,'Phase9B access denied: forbidden input')
        return super().read(name)


@contextmanager
def access_guard(root: Path, reads: set[str], writes: set[str]):
    """Enforce this phase's narrower allowlist before delegating the open hook."""
    require(reads<=SAFE_INPUTS,'Phase9B access denied: forbidden input')
    with governance.access_guard(root,reads,writes):yield


def validate_source_lock(lock: dict) -> None:
    """Bind the whole source map, not caller-redeclared individual hashes."""
    # Original registration uses sorted, indented JSON with a terminal newline.
    payload=(json.dumps(lock,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    require(sha256(payload).hexdigest()==SOURCE_SHA,'Phase9B source lock/pins changed')
    require(lock['roles']==ROLES and set(lock['files_sha256'])==SAFE_INPUTS,'Changed source roles/pins')


def source_lock(root: Path) -> dict:
    """Read only the immutable bootstrap metadata before opening any data."""
    reader=governance.PinnedReader(root,{SOURCE:SOURCE_SHA})
    with governance.access_guard(root,{SOURCE},set()):lock=json.loads(reader.read(SOURCE))
    validate_source_lock(lock);return lock


def validate_development(frame: pd.DataFrame) -> None:
    """Reject non-development rows and label end dates, including invalid ISO."""
    for column in ['date','label_end_date']:
        if column not in frame:continue
        for value in frame[column]:
            if column=='label_end_date' and (pd.isna(value) or value==''):continue
            try:valid=isinstance(value,str) and '2018-01-01'<=day(value)<='2025-12-31'
            except ValueError as exc:raise ValueError('Invalid development date') from exc
            require(valid,'Non-development date forbidden')


def load(root: Path) -> dict:
    """Verify old publications, then parse only development-specific payloads."""
    root=Path(root).resolve();lock=source_lock(root)
    prior=governance.build(root)
    require(prior['build_key']==lock['phase9a_build_key'],'Changed Phase9A protocol')
    require(prior['code_sha256']==lock['phase9a_code_sha256'],'Changed Phase9A code')
    reader=PinnedReader(root,lock['files_sha256'])
    with access_guard(root,SAFE_INPUTS,set()):
        content={name:reader.read(name) for name in sorted(SAFE_INPUTS)}
    def yaml_doc(name):return yaml.load(content[name].decode(),Loader=_UniqueSafeLoader)
    def frame(role):
        out=pd.read_csv(BytesIO(content[ROLES[role]]),dtype={'date':str,'asset_code':str,'exchange':str,'label_end_date':'string'},
            float_precision='round_trip',keep_default_na=False,na_values=[''])
        validate_development(out);return out
    config=yaml_doc('config/config.yaml');validation=yaml_doc('config/validation.yaml');registration=yaml_doc('config/scale_free.yaml')
    panel=frame('panel');features=frame('features');labels=frame('labels');raw_features=frame('raw_features');eligibility=frame('eligibility')
    calendar=frame('calendar').date.tolist();historical=frame('historical_calendar').date.tolist()
    require(calendar==sorted(set(calendar)) and historical==[d for d in calendar if d>='2022-01-04']
        and len(historical)==969,'Frozen development calendar/session identity changed')
    require(set(panel.date)<=set(calendar),'Development quote outside frozen sessions')
    for f in [panel,features,labels,raw_features,eligibility]:
        require(not f.duplicated(['date','exchange','asset_code']).any(),'Duplicate asset/date')
    raw,policy=read_policy(root/'config/execution.yaml')
    # Execution quotes retain the exact decimal strings used in Phase7.
    market=list(csv.DictReader(StringIO(content[ROLES['panel']].decode())))
    assets=[Asset(a['code'],panel.loc[panel.asset_code.eq(a['code']),'date'].min(),lot_size=raw['research_lot_size'])
        for a in config['universe']['assets']]
    quotes=[Quote(r['date'],r['asset_code'],r['open'] or None,r['close'] or None,
        high=r['high'] or None,low=r['low'] or None) for r in market if r['date'] in set(historical)]
    require(all(r['price_basis']=='none' for r in market),'Adjusted execution quote forbidden')
    names={f.name for f in fields(Action)}
    actions=[Action(**{k:tuple(v) if k=='suspended_dates' else v for k,v in r.items() if k in names})
        for r in json.loads(content[ROLES['actions']])]
    gates=json.loads(content[ROLES['accounting_gates']])
    require(gates['historical_research_execution_ready'] and gates['historical_total_return_research_ready'],'Accounting not ready')
    predrows=list(csv.DictReader(StringIO(content[ROLES['predictions']].decode())))
    predictions=project_predictions(predrows)
    binding={'source_lock_sha256':SOURCE_SHA,'inputs_sha256':reader.opened,
        'phase9a_build_key':prior['build_key'],'phase8_build_key':prior['phase8_build_key'],
        'rules_hash':prior['rules_hash'],'strategy_spec_hash':prior['strategy_spec_hash'],
        'holdout_files_opened':False,'raw_calendar_opened':False,'historical_predictions_role':'account_equivalence_only'}
    return dict(config=config,validation=validation,registration=registration,panel=panel,features=features,
        raw_features=raw_features,labels=labels,eligibility=eligibility,calendar=calendar,historical_sessions=historical,
        predictions=predictions,actions=actions,assets=assets,policy=policy,quotes=quotes,
        exchanges={r['asset_code']:r['exchange'] for r in market},
        segments={(r['date'],r['asset_code']):int(r['price_segment']) for r in market},
        phase7_manifest=json.loads(content[P7+'manifest.json']),account_hashes=json.loads(content[ROLES['account_hashes']]),
        reader=reader,source_lock=lock,binding=binding)
