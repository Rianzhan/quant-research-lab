"""Pinned safe inputs: execution data and post-replay diagnostics are separated."""
import csv,json
from dataclasses import fields
from hashlib import sha256
from io import StringIO
from pathlib import Path
from src.backtest.contracts import Action,day,require,stable_bytes
from src.backtest.inputs import DEVELOPMENT,load_inputs,read_policy
from src.portfolio.inputs import SpecReader,read_spec,verify_code
from src.portfolio.rules import project_predictions,rules_hash

SPEC_HASH='cc65661cd7847d56eae23cbbba82338e177321090a570168f82737c64d1dc8e9'
RAW_SPEC_SHA='764d367a875aea457cd6f4b216704514d065bf424923659f1f1a0e461cb7601f'
RULES_HASH='384d95f2a5ba0faa51100ec73fb399e52ca052acff20480b9d1d6a843404b1fc'
ACCEPTED_MANIFEST='results/reports/phase6b-v1-0ae0b62b57503175/manifest.json'
MANIFEST_SHA='5e733399c1863131c5647ee8f927901d8dbc5dc5f684e0598a28cb080198538c'

def checked_bytes(reader: SpecReader,name: str) -> bytes:
    """Use the accepted before-open exclusion and exact-byte hash check."""
    return reader.read(name)

def validate_spec(spec: dict):
    """Reject changes to the accepted strategy even if a caller edits new config."""
    require(sha256(stable_bytes(spec)).hexdigest()==SPEC_HASH,'Accepted Phase6B specification changed')
    rules=read_spec(spec)
    require(rules_hash(rules)==RULES_HASH,'Accepted Phase6B rule hash changed')
    return rules

def validate_pins(config: dict,lock: dict,manifest: dict) -> None:
    """New configuration cannot bless changed old source/code hashes."""
    require(all(config['files_sha256'].get(k)==v for k,v in lock['files_sha256'].items()),'Accepted source hashes redeclared')
    prefix=Path(ACCEPTED_MANIFEST).parent.as_posix()+'/'
    published={prefix+k:v for k,v in manifest['files'].items()}
    require(all(config['files_sha256'].get(k)==v for k,v in published.items()),'Accepted publication hashes redeclared')
    require(config['files_sha256'].get('config/strategy_spec.json')==RAW_SPEC_SHA,'Accepted raw specification hash redeclared')
    allowed=set(lock['files_sha256'])|set(published)|{ACCEPTED_MANIFEST,'config/strategy_spec.json','config/strategy_sources.json'}
    require(set(config['files_sha256'])==allowed,'Accepted input allowlist changed')
    code={**lock['preserved_code_sha256'],**manifest['bindings']['code_sha256']}
    require(config['preserved_code_sha256']==code,'Accepted source code hashes redeclared')
    for k in ['predictions','calendar_raw']:require(config[k]==lock[k],'Changed accepted source role')
    parent=str(Path(lock['accounting_manifest']).parent).replace('\\','/')
    require(config['corporate_actions']==parent+'/corporate_actions.json' and config['accounting_gates']==parent+'/gates.json','Changed accounting source role')
    feature='data/processed/features/phase45-v1-0a9eb107fb50d0b5/'
    require(config['development']==DEVELOPMENT and config['eligibility']==feature+'sample_eligibility.csv'
            and config['membership']==feature+'fold_membership.csv','Changed development/diagnostic source role')

def rows(reader: SpecReader,name: str) -> list[dict]:
    """Read only an already registered CSV, rejecting any non-development date."""
    out=list(csv.DictReader(StringIO(checked_bytes(reader,name).decode('utf-8'))))
    for row in out:
        if 'date' in row:require('2018-01-01'<=day(row['date'])<='2025-12-31','Non-development CSV row')
    return out

def diagnostic_rows(values: list[dict]) -> dict:
    """Only availability flags, never realized target magnitudes; called outside trading."""
    result={}
    for row in values:
        date=day(row['date']);require('2018-01-01'<=date<='2025-12-31','Non-development diagnostics')
        key=(date,row['asset_code']);require(key not in result,'Duplicate diagnostic key')
        require(row['features_complete'] in {'True','False'} and row['label_valid'] in {'True','False'},'Invalid availability flag')
        result[key]={'exchange':row['exchange'],'price_segment':int(row['price_segment']),
                     'feature_available':row['features_complete']=='True','label_valid':row['label_valid']=='True'}
    return result

def contract(root: Path) -> tuple[dict,SpecReader,object,object]:
    """Bind old sources/code before market parsing, then provide a repeat checker."""
    root=root.resolve();path=root/'config/phase7.json';payload=path.read_bytes();config=json.loads(payload)
    require(config['version']=='phase7-v1' and config['strategy_spec_hash']==SPEC_HASH and config['rules_hash']==RULES_HASH,'Changed Phase7 contract')
    require(config['holdout_files_opened'] is False and config['live_execution_ready'] is False,'Forbidden scope')
    reader=SpecReader(root,config['files_sha256'],config['calendar_raw'])
    spec=json.loads(reader.read('config/strategy_spec.json'));rules=validate_spec(spec)
    lock_bytes=reader.read(spec['source_lock'])
    require(sha256(lock_bytes).hexdigest()==spec['source_lock_sha256'],'Accepted source lock changed')
    require(config['phase6b_manifest']==ACCEPTED_MANIFEST,'Changed accepted publication')
    accepted=reader.read(ACCEPTED_MANIFEST)
    require(sha256(accepted).hexdigest()==MANIFEST_SHA,'Accepted publication changed')
    validate_pins(config,json.loads(lock_bytes),json.loads(accepted))
    def recheck():
        require(path.read_bytes()==payload,'Phase7 configuration changed during run')
        reader.verify_all();verify_code(root,config['preserved_code_sha256'])
    recheck()
    return config,reader,rules,recheck

def load(root: Path,config: dict,reader: SpecReader) -> tuple[dict,dict]:
    """Execution arguments contain no feature-completeness or label-valid fields."""
    raw,policy=read_policy(root/'config/execution.yaml');loaded=load_inputs(root,raw)
    gates=json.loads(reader.read(config['accounting_gates']))
    require(gates['historical_research_execution_ready'] and gates['historical_total_return_research_ready'],'Research accounting gate closed')
    registry=json.loads(reader.read(config['corporate_actions']));names={f.name for f in fields(Action)}
    actions=[Action(**{k:tuple(v) if k=='suspended_dates' else v for k,v in r.items() if k in names}) for r in registry]
    market=rows(reader,DEVELOPMENT)
    segments={(r['date'],r['asset_code']):int(r['price_segment']) for r in market}
    exchanges={r['asset_code']:r['exchange'] for r in market}
    pred=project_predictions(rows(reader,config['predictions']))
    dates=[d for d in loaded['calendar'] if '2022-01-04'<=d<='2025-12-31']
    execution={'policy':policy,'predictions':pred,'assets':loaded['assets'],'sessions':dates,
               'quotes':[q for q in loaded['quotes'] if q.date in set(dates)],'exchanges':exchanges,'segments':segments,'actions':actions}
    diagnostic=diagnostic_rows(rows(reader,config['eligibility']))
    membership=rows(reader,config['membership']);statuses={}
    for r in membership:
        key=(r['fold_id'],r['asset_code']);value=r['asset_status']
        require(key not in statuses or statuses[key]==value,'Conflicting fold asset availability')
        statuses[key]=value
    observed={code:sorted(r['date'] for r in market if r['asset_code']==code) for code in exchanges}
    diagnostics={'availability':diagnostic,'asset_fold_status':statuses,'observed_dates':observed}
    return execution,diagnostics
