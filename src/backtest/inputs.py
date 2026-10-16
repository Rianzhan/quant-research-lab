"""Explicit safe source allowlist: never call legacy mixed-price replay loaders."""
from dataclasses import fields
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import pandas as pd
import yaml
from src.backtest.actions import applicability_issues, register_actions
from src.backtest.contracts import Asset, Policy, Quote, day, require
from src.utils.config import _UniqueSafeLoader

DEVELOPMENT='data/processed/research/phase25-v1/development.csv'


class FrozenReader:
    """Audit all opens, reject sealed or unlisted files before filesystem access."""

    def __init__(self, root: Path, files: dict[str,str], calendar_raw: str=''):
        self.root=Path(root).resolve();self.files=files;self.calendar_raw=calendar_raw
        self.opened_paths: list[str]=[]

    def path(self, relative: str) -> Path:
        """Resolve containment and structural exclusions before touching file bytes."""
        name=relative.replace('\\','/');parts=PurePosixPath(name)
        require(not parts.is_absolute() and '..' not in parts.parts and ':' not in name, 'Path escapes repository')
        require(not any(x.startswith('.pending-') for x in parts.parts), 'Incomplete source forbidden')
        lowered=name.lower()
        require('holdout' not in lowered and 'coverage_prices' not in lowered, 'Sealed/mixed file forbidden')
        require(not (lowered.startswith('data/raw/') and lowered.endswith('.csv') and name!=self.calendar_raw), 'Mixed raw prices forbidden')
        allowed=(name=='development.csv' or name=='requirements.lock.txt' or name.startswith('config/')
                 or name.startswith('data/processed/') or name.startswith('results/reports/phase55')
                 or (name.startswith('data/raw/') and (name.endswith('/manifest.json') or name==self.calendar_raw
                     or (name.startswith('data/raw/policy_evidence/') and name.endswith('.pdf')))))
        require(allowed and name in self.files, 'Source not in explicit allowlist')
        path=(self.root/name).resolve()
        require(path.is_relative_to(self.root), 'Path escapes repository')
        # Resolve aliases as well: a harmless lexical name must not point at a sealed file.
        resolved=path.relative_to(self.root).as_posix()
        require(resolved==name, 'Source aliases forbidden')
        return path

    def read(self, relative: str) -> bytes:
        path=self.path(relative);payload=path.read_bytes();self.opened_paths.append(relative)
        require(sha256(payload).hexdigest()==self.files[relative], 'Frozen file hash changed: '+relative)
        return payload

    def verify_all(self) -> None:
        """Opaque hashes only: model binaries/predictions are never interpreted."""
        for name in sorted(self.files):
            self.read(name)

    def csv(self, relative: str) -> pd.DataFrame:
        """Only the development panel and exact registered calendar may be parsed."""
        require(relative in {DEVELOPMENT,self.calendar_raw} and relative, 'CSV parse not permitted')
        frame=pd.read_csv(BytesIO(self.read(relative)),dtype=str,keep_default_na=False)
        if relative==DEVELOPMENT:
            require('date' in frame and frame.date.map(lambda x:'2018-01-01'<=day(x)<='2025-12-31').all(),
                    'Non-development rows forbidden')
        return frame


def read_policy(path: Path) -> tuple[dict,Policy]:
    """Strict execution-only configuration; previous phase configs stay frozen."""
    raw=yaml.load(Path(path).read_text(encoding='utf-8'),Loader=_UniqueSafeLoader)
    require(raw['schema_version']==1 and raw['version']=='phase6a-v1', 'Unsupported execution policy')
    for name in ['model_signals_used','strategy_backtest_performed','holdout_access','live_execution_ready']:
        require(raw[name] is False, 'Execution scope flag must stay false: '+name)
    expected={'signal_time':'t_close_after','execution_time':'t_plus_1_open',
        'dividend_payment_availability':'payment_date_after_close','same_day_order_priority':'sell_then_buy_asset_code_order_id',
        'missing_open_policy':'terminate_at_next_session','event_crossing_policy':'cancel',
        'stale_policy':'last_own_close_same_share_basis_valuation_only',
        'turnover_definition':'effective_traded_notional_over_open_pre_trade_nav',
        'incomplete_event_policy':'reject_affected_window','orders_source':'frozen_manual_fixture',
        'lot_size_basis':'unverified_research_unit_not_broker_rule','decimal_precision':40}
    require(all(raw.get(k)==v for k,v in expected.items()), 'Unsupported accounting policy')
    require(raw['fee_rate_sensitivity']==['0','0.0003','0.0005','0.001'], 'Frozen fee sensitivity changed')
    supported=set(expected)|{'schema_version','version','development','initial_cash','fee_rate','fee_rate_sensitivity',
        'slippage','epsilon','research_lot_size','sale_proceeds_reuse_policy','model_signals_used',
        'strategy_backtest_performed','holdout_access','live_execution_ready','share_event_details','source_lock','source_lock_sha256'}
    require(set(raw)==supported, 'Unexpected execution configuration keys')
    policy=Policy(**{f.name:raw[f.name] for f in fields(Policy) if f.name in raw},**raw['development'])
    policy.validate()
    return raw,policy


def load_inputs(root: Path, raw: dict) -> dict:
    """Verify old artifacts, then load raw development quotes/calendar and events only."""
    root=Path(root).resolve()
    relative=raw['source_lock'];path=(root/relative).resolve()
    require(path.is_relative_to(root) and relative=='config/execution_sources.json', 'Invalid source lock path')
    payload=path.read_bytes()
    require(sha256(payload).hexdigest()==raw['source_lock_sha256'], 'Source lock hash changed')
    lock=json.loads(payload);reader=FrozenReader(root,lock['files_sha256'],lock['calendar_raw'])
    reader.verify_all()
    verify_preserved_code(root,lock['preserved_code_sha256'])
    frame=reader.csv(DEVELOPMENT)
    required={'date','asset_code','exchange','open','high','low','close','price_basis'}
    require(required<=set(frame.columns) and not frame.empty, 'Invalid development OHLC schema')
    require((frame.price_basis=='none').all(), 'Execution requires unadjusted prices')
    require(not frame.duplicated(['date','asset_code']).any(), 'Duplicate development asset/date')
    calendar=reader.csv(lock['calendar_raw'])
    require(list(calendar.columns)==['trade_date'] and calendar.trade_date.is_unique and calendar.trade_date.is_monotonic_increasing,
            'Invalid domestic calendar')
    sessions=[day(x) for x in calendar.trade_date if '2018-01-01'<=x<='2025-12-31']
    require(set(frame.date)<=set(sessions), 'Quote outside domestic sessions')
    config=yaml.load(reader.read('config/config.yaml').decode('utf-8'),Loader=_UniqueSafeLoader)
    require(set(frame.asset_code)=={a['code'] for a in config['universe']['assets']}, 'Frozen asset universe mismatch')
    require(config['universe']['registration_status']=='frozen', 'Unregistered asset pool')
    assets=[Asset(a['code'],frame.loc[frame.asset_code==a['code'],'date'].min(),lot_size=raw['research_lot_size'])
            for a in config['universe']['assets']]
    research=json.loads(reader.read(lock['research_policy']))
    actions,table=register_actions(research,raw)
    for a in actions:
        require(not applicability_issues(a,assets),'Invalid event applicability: '+a.event_id)
    quotes=[Quote(x.date,x.asset_code,x.open or None,x.close or None,high=x.high or None,low=x.low or None)
            for x in frame.itertuples()]
    return {'assets':assets,'calendar':sessions,'quotes':quotes,'actions':actions,'action_table':table,
            'reader':reader,'source_lock':lock,'source_lock_sha256':raw['source_lock_sha256']}


def verify_preserved_code(root: Path, expected_files: dict[str,str]) -> None:
    """Check old stage source bytes both before replay and immediately before publish."""
    for relative,expected in expected_files.items():
        p=(root/relative).resolve()
        require(p.is_relative_to(root) and relative.startswith('src/') and relative.endswith('.py'), 'Unsafe code binding')
        require(sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected, 'Previously frozen code changed: '+relative)
