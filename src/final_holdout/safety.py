"""Before-open guard for every project market payload, including legacy loaders."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import sys
from src.final_holdout.inputs import SAFE_INPUTS

_ACTIVE=ContextVar('phase9b_market_guard',default=None)


def _audit(event: str, args: tuple) -> None:
    state=_ACTIVE.get()
    if event!='open' or state is None:return
    root,record=state;filename,mode,flags=args
    if not isinstance(filename,(str,bytes,os.PathLike)):return
    path=Path(os.fsdecode(filename)).resolve()
    try:name=path.relative_to(root).as_posix()
    except ValueError:return
    writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    # Temporary synthetic fixtures are allowed; actual market data is fail closed.
    lower=name.casefold()
    if lower.startswith('data/') and (writing or name not in SAFE_INPUTS):
        record['denied'].append(name)
        raise ValueError('Phase9B forbidden market payload before open: '+name)
    if writing and (name in SAFE_INPUTS or lower.startswith('results/reports/phase8-v1-')
                    or lower.startswith('results/reports/phase9a-v1-')):
        raise ValueError('Phase9B cannot modify frozen publication/input: '+name)
    if not writing and lower.startswith('data/'):
        record['opened'].add(name)


sys.addaudithook(_audit)


@contextmanager
def sealed_guard(root: Path):
    """Leave unrelated runtime modules accessible; project market reads pinned."""
    record={'opened':set(),'denied':[]};token=_ACTIVE.set((root.resolve(),record))
    try:yield record
    finally:_ACTIVE.reset(token)
