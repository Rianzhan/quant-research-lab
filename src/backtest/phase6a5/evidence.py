"""Reconstruct advance terms independently of the later confirmation notice."""
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP, localcontext
from src.backtest.contracts import Action, day, number, require


def calculate_ratio(notice: dict) -> Decimal:
    """Use the advance notice's X/Y/(I/1000), rounded half up to five places."""
    with localcontext() as ctx:
        ctx.prec=40
        x,y,i=(number(notice[k]) for k in ('net_assets','fund_shares','index_close'))
        require(min(x,y,i)>0,'Invalid advance formula inputs')
        return (x/y/(i/1000)).quantize(Decimal('.00001'),rounding=ROUND_HALF_UP)


def correct_event(original: Action, notice: dict) -> Action:
    """Explicit dated erratum, never overwrite the original registered Action."""
    require(original.event_id==notice['event_id']=='510500-split-2022-08-26','Wrong correction identity')
    require(notice['time_precision']=='date','No evidenced intraday publication timestamp')
    require(notice['known_at']==day(notice['published_date'])+'T23:59:59+08:00','Date precision requires end-of-day bound')
    require(notice['published_date']<original.effective_date,'Advance notice must precede effective date')
    require(calculate_ratio(notice)==number(notice['announced_ratio']),'Advance formula disagrees with stated ratio')
    corrected=replace(original,known_at=notice['known_at'],share_ratio=notice['announced_ratio'],
        evidence_sha256=notice['evidence_sha256'],
        limitation='Ratio/rounding disclosed in 2022-08-23 advance notice; date precision, conservative day-end bound.')
    require(not corrected.issues(),'Incomplete corrected advance terms: '+','.join(corrected.issues()))
    return corrected
