"""Immutable specification publication; the output set excludes strategy P&L."""
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4
from typing import Callable
from src.backtest.contracts import require,stable_bytes
from src.backtest.pipeline import write_file

VERSION='phase6b-v1'
FILES={'strategy_spec.json','strategy_spec.sha256','gates.json','history_audit.json','prediction_audit.json',
       'rebalance_calendar.csv','examples.json','access_audit.json','protocol.md','source_bindings.json'}

def check_outputs(files: dict[str,bytes]) -> None:
    """Only specified contracts, synthetic examples and structural audits may publish."""
    require(set(files)==FILES,'Unsupported outputs: no strategy performance/NAV allowed')
    gates=json.loads(files['gates.json'])
    for k in ['strategy_spec_ready','signal_to_order_mapping_ready','strategy_rules_frozen']:
        require(gates.get(k) is True,'Incomplete strategy gate: '+k)
    for k in ['strategy_backtest_performed','holdout_files_opened']:
        require(gates.get(k) is False,'Forbidden strategy gate: '+k)

def verify(manifest: Path) -> dict:
    """Read and check a complete content-addressed protocol without repairs."""
    manifest=Path(manifest);meta=json.loads(manifest.read_bytes())
    require(manifest.name=='manifest.json' and meta['status']=='complete','Incomplete publication')
    require(set(meta['files'])==FILES,'Unexpected publication output set')
    files={}
    for name,h in meta['files'].items():
        p=(manifest.parent/name).resolve();require(p.is_relative_to(manifest.parent.resolve()),'Publication escape')
        files[name]=p.read_bytes();require(sha256(files[name]).hexdigest()==h,'Publication file changed: '+name)
    check_outputs(files)
    key=sha256(stable_bytes({'version':VERSION,'bindings':meta['bindings'],'files':meta['files']})).hexdigest()
    require(meta['build_key']==key and manifest.parent.name==VERSION+'-'+key[:16],'Publication identity changed')
    return meta

def publish(parent: Path,files: dict[str,bytes],bindings: dict,recheck: Callable[[],None]) -> Path:
    """Write/readback/source-recheck/rename; failed directories never look complete."""
    check_outputs(files);recheck()
    hashes={n:sha256(b).hexdigest() for n,b in files.items()}
    key=sha256(stable_bytes({'version':VERSION,'bindings':bindings,'files':hashes})).hexdigest()
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True);target=parent/(VERSION+'-'+key[:16])
    if target.exists():verify(target/'manifest.json');return target/'manifest.json'
    pending=parent/('.pending-'+uuid4().hex);pending.mkdir()
    for name,b in sorted(files.items()):write_file(pending/name,b)
    for name,h in hashes.items():require(sha256((pending/name).read_bytes()).hexdigest()==h,'Pending write corrupted')
    meta={'status':'complete','version':VERSION,'build_key':key,'bindings':bindings,'files':hashes}
    write_file(pending/'manifest.json',stable_bytes(meta))
    require((pending/'manifest.json').read_bytes()==stable_bytes(meta),'Pending manifest corrupted')
    recheck();pending.rename(target)
    return target/'manifest.json'
