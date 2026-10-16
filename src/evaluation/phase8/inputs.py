"""Independently anchored account files, never raw prices or prediction recomputation."""
import csv,json,sys
from io import StringIO
from pathlib import Path,PurePosixPath
from .common import STRATEGIES,FEES,account_id,digest,require,stable_bytes

MANIFEST='results/reports/phase7-v1-34d765af3b0494e9/manifest.json'
MANIFEST_SHA='641dbde1c824c81809c30a2629c5d0127f27435bc49b8896889a4a35795d0e0e'
BUILD_KEY='34d765af3b0494e98e49afbc53fd4a15e798b5a6842907506ad81688b13baab7'
SPEC_HASH='cc65661cd7847d56eae23cbbba82338e177321090a570168f82737c64d1dc8e9'
RAW_SPEC_SHA='764d367a875aea457cd6f4b216704514d065bf424923659f1f1a0e461cb7601f'
RULES_HASH='384d95f2a5ba0faa51100ec73fb399e52ca052acff20480b9d1d6a843404b1fc'
ACCOUNT_NAMES={account_id(s,f) for s in STRATEGIES for f in FEES}

class PinnedReader:
    """Exact allowlist and byte hashes; exclusions are checked before open."""
    def __init__(self,root: Path,files: dict[str,str]):
        self.root=Path(root).resolve();self.files=files;self.opened=[]

    def read(self,name: str) -> bytes:
        require(isinstance(name,str) and '\\' not in name and ':' not in name,'Invalid path')
        p=PurePosixPath(name);low=name.lower()
        require(not p.is_absolute() and '..' not in p.parts,'Path escape')
        require(not low.startswith('data/') and 'holdout' not in low and 'coverage_prices' not in low,'Market data prohibited in Phase8')
        require(name in self.files,'Input not allowlisted')
        path=(self.root/name).resolve()
        require(path.is_relative_to(self.root) and path.relative_to(self.root).as_posix()==name,'Input alias/escape')
        payload=path.read_bytes();self.opened.append(name)
        require(digest(payload)==self.files[name],'Frozen input changed: '+name)
        return payload

    def verify_all(self) -> None:
        for name in sorted(self.files):self.read(name)

def verify_manifest(payload: bytes) -> dict:
    """Pin raw manifest bytes and its canonical content-addressed build key."""
    require(digest(payload)==MANIFEST_SHA,'Accepted Phase7 manifest changed')
    meta=json.loads(payload)
    key=digest(stable_bytes({k:meta[k] for k in ['version','bindings','files']}))
    require(meta['build_key']==key==BUILD_KEY and meta['status']=='complete','Phase7 build changed')
    require(meta['bindings']['strategy_spec_hash']==SPEC_HASH and meta['bindings']['rules_hash']==RULES_HASH,'Phase7 rule binding changed')
    require(len(meta['files'])==189,'Incomplete Phase7 publication')
    return meta

def verify_spec(payload: bytes) -> dict:
    require(digest(payload)==RAW_SPEC_SHA,'Raw strategy specification changed')
    spec=json.loads(payload);require(digest(stable_bytes(spec))==SPEC_HASH,'Canonical specification changed')
    return spec

def validate_config(config: dict) -> None:
    """Evaluation conventions are fixed before viewing risk/annual metrics."""
    expected={'version':'phase8-v1','phase7_commit':'b77fadbf881f9cd2337ebaad63a7f748e28859d3',
        'phase7_manifest':MANIFEST,'phase7_manifest_sha256':MANIFEST_SHA,'phase7_build_key':BUILD_KEY,
        'strategy_spec_hash':SPEC_HASH,'rules_hash':RULES_HASH,'annualization_sessions':252,'risk_free_rate':0,
        'maximum_drawdown_sign':'positive_loss_magnitude','return_interval':'each_frozen_session_including_initial_session',
        'holdout_files_opened':False,'strategy_optimization_performed':False}
    require(config==expected,'Phase8 evaluation contract changed')

def validate_account_hashes(hashes: dict,summaries: dict) -> None:
    """Compare all twelve identities to summaries under the independently pinned manifest."""
    require(set(hashes)==set(summaries)==ACCOUNT_NAMES,'Exactly twelve accepted accounts required')
    for name,entry in hashes.items():
        require(entry['account_hash']==summaries[name]['account_hash'],'Account identity changed: '+name)
        require(entry['rules_hash']==summaries[name]['rule_hash']==RULES_HASH,'Account rules changed')
        require(len(entry['account_hash'])==64,'Invalid account digest')

def csv_rows(payload: bytes) -> list[dict]:
    return list(csv.DictReader(StringIO(payload.decode('utf-8'))))

def load(root: Path) -> tuple[dict,object]:
    """Hash all Phase7 members; parse only frozen accounts, coverage and specification."""
    root=Path(root).resolve();config_payload=(root/'config/phase8.json').read_bytes()
    config=json.loads(config_payload);validate_config(config)
    reader=PinnedReader(root,{MANIFEST:MANIFEST_SHA,'config/strategy_spec.json':RAW_SPEC_SHA})
    meta=verify_manifest(reader.read(MANIFEST));prefix=Path(MANIFEST).parent.as_posix()+'/'
    reader.files.update({prefix+k:v for k,v in meta['files'].items()});reader.verify_all()
    spec=verify_spec(reader.read('config/strategy_spec.json'))
    hashes=json.loads(reader.read(prefix+'account_hashes.json'));accounts={};summaries={}
    for name in sorted(ACCOUNT_NAMES):
        base=prefix+'accounts/'+name+'/'
        summaries[name]=json.loads(reader.read(base+'summary.json'))
        accounts[name]={k:csv_rows(reader.read(base+v+'.csv')) for k,v in
            {'daily':'daily_ledger','holdings':'holdings_ledger','costs':'costs','turnover':'turnover','fills':'fills','orders':'orders'}.items()}
        accounts[name]['summary']=summaries[name]
    validate_account_hashes(hashes,summaries)
    bundle={'accounts':accounts,'coverage':csv_rows(reader.read(prefix+'coverage_audit.csv')),'spec':spec,
        'manifest':meta,'account_hashes':hashes,'config':config,
        'input_audit':{'phase7_build_key':BUILD_KEY,'manifest_sha256':MANIFEST_SHA,'account_hashes':hashes,
            'files_verified':len(reader.files),'opened_files':sorted(set(reader.opened)),
            'account_hash_verification':'pinned manifest member hashes plus account_hash/summary identity; no strategy replay',
            'holdout_files_opened':False,'market_payloads_opened':False,'strategy_replayed':False}}
    def recheck():
        require((root/'config/phase8.json').read_bytes()==config_payload,'Evaluation config changed during run')
        reader.verify_all()
    return bundle,recheck

def install_guard(root: Path) -> dict:
    """CLI-wide guard denies all market/feature/prediction datasets before opening."""
    root=Path(root).resolve();audit={'denied':[],'holdout_files_opened':False}
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if not path.is_relative_to(root):return
        rel=path.relative_to(root).as_posix().lower()
        if rel.startswith('data/') or 'holdout_candidate' in rel or 'coverage_prices' in rel:
            audit['denied'].append(rel);raise ValueError('Phase8 data open forbidden: '+rel)
    sys.addaudithook(hook)
    return audit
