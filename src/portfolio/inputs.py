"""Allowlisted, hashed inputs; model binaries/labels are never interpreted."""
import csv
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path,PurePosixPath
import sys
from src.backtest.contracts import require
from src.portfolio.rules import Rules

class SpecReader:
    """Reject sealed files and aliases before opening any supplied path."""
    def __init__(self,root: Path,files: dict[str,str],calendar: str):
        self.root=Path(root).resolve();self.files=files;self.calendar=calendar;self.opened=[]

    def path(self,name: str) -> Path:
        """Resolve an exact allowlisted relative name, never an alternate alias."""
        require(isinstance(name,str) and '\\' not in name and ':' not in name,'Invalid relative name')
        parts=PurePosixPath(name)
        require(not parts.is_absolute() and '..' not in parts.parts,'Path escape')
        low=name.lower()
        require('holdout' not in low and 'coverage_prices' not in low,'Sealed market access forbidden')
        require(not (low.startswith('data/raw/') and low.endswith('.csv') and name!=self.calendar),'Mixed raw market forbidden')
        require(name in self.files and not any(p.startswith('.pending-') for p in parts.parts),'Unlisted/incomplete source')
        p=(self.root/name).resolve()
        require(p.is_relative_to(self.root) and p.relative_to(self.root).as_posix()==name,'Source alias/escape forbidden')
        return p

    def read(self,name: str) -> bytes:
        """Return bytes only after checking the immutable identity."""
        data=self.path(name).read_bytes();self.opened.append(name)
        require(sha256(data).hexdigest()==self.files[name],'Frozen input changed: '+name)
        return data

    def verify_all(self) -> None:
        """Opaque SHA checks include frozen predictions/models, never deserialization."""
        for name in sorted(self.files):self.read(name)

def read_spec(spec: dict) -> Rules:
    """Phase6B v1 admits one registered rule set, never a parameter grid."""
    require(spec['registration_status']=='frozen','Strategy rules not frozen')
    for k in ['strategy_backtest_performed','holdout_files_opened','model_training_performed','historical_signals_generated']:
        require(spec[k] is False,'Forbidden Phase6B scope flag: '+k)
    require(spec['governance']['selection_status']=='development_stage_model_family_selection','Selection evidence mislabeled')
    require(spec['governance']['holdout_status']=='sealed_candidate','Holdout must remain sealed')
    values=dict(spec['rules']);values['fee_scenarios']=tuple(values['fee_scenarios'])
    rules=Rules(**values)
    require(asdict(rules)==asdict(Rules()),'Frozen v1 rules changed; a new authorized protocol is required')
    return rules

def verify_code(root: Path,expected: dict[str,str]) -> None:
    """Normalized source hashes preserve all earlier phases without rewriting them."""
    for name,h in expected.items():
        p=(root/name).resolve()
        require(p.is_relative_to(root) and name.startswith('src/') and name.endswith('.py'),'Invalid source-code path')
        require(sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,'Frozen source code changed: '+name)

def csv_rows(payload: bytes,columns: list[str]) -> list[dict]:
    """Parse only a specifically approved schema; no automatic target joins."""
    reader=csv.DictReader(StringIO(payload.decode('utf-8')))
    require(reader.fieldnames==columns,'Unexpected approved CSV schema')
    return list(reader)

def install_guard(reader: SpecReader) -> dict:
    """Trace data-file opens by the CLI and fail before sealed payload access."""
    audit={'holdout_files_opened':False,'denied':[],'market_paths_opened':[]}
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if not p.is_relative_to(reader.root):return
        rel=p.relative_to(reader.root).as_posix()
        if rel.startswith(('data/raw/','data/processed/')):
            try:reader.path(rel)
            except ValueError:audit['denied'].append(rel);raise
            audit['market_paths_opened'].append(rel)
    sys.addaudithook(hook)
    return audit
