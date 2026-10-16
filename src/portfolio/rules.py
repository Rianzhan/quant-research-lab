"""Pure decision-time inputs, fixed strategy families and capped equal weights."""
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN, localcontext
from hashlib import sha256
from typing import Iterable, Mapping, Any
from src.backtest.contracts import day, number, require, stable_bytes

FAMILIES={'A':('per_asset','logistic','classification'),
          'B':('pooled','random_forest','regression'), 'C':None}

@dataclass(frozen=True)
class Rules:
    """All numeric rules precede strategy P&L; costs are single-side rates."""
    version: str='phase6b-v1'
    prediction_horizon_days: int=5
    rebalance_frequency_days: int=5
    anchor: str='2022-01-04'
    end: str='2025-12-31'
    classification_threshold: str='0.55'
    regression_hurdle: str='0.001'
    max_asset_weight: str='0.25'
    max_total_weight: str='1'
    primary_fee_rate: str='0.0003'
    fee_scenarios: tuple[str,...]=('0','0.0003','0.0005','0.001')
    slippage: str='0.0002'
    signal_time: str='t_close_after'
    execution_time: str='t_plus_1_open'

def rules_hash(rules: Rules) -> str:
    """Canonical rule-only SHA, independent of input row ordering."""
    return sha256(stable_bytes(asdict(rules))).hexdigest()

@dataclass(frozen=True)
class AssetState:
    """t-close fields only. No next open or realized future outcome is representable."""
    asset_code: str
    exchange: str
    first_tradable_date: str
    close: str | None
    shares: str='0'
    price_segment: int=0
    lot_size: str='1'
    tradable_now: bool=True
    basis_valid: bool=True
    pending_action: bool=False
    last_tradable_date: str | None=None

@dataclass(frozen=True)
class DecisionState:
    """Snapshot supplied after the accepted engine has posted t-close events."""
    date: str
    nav: str
    cash_available: str
    assets: tuple[AssetState,...]

@dataclass(frozen=True,order=True)
class Prediction:
    """Whitelisted prediction plus evaluation key, never a label."""
    date: str
    exchange: str
    asset_code: str
    price_segment: int
    fold_id: str
    scope: str
    model: str
    task: str
    prediction: Decimal

def project_predictions(rows: Iterable[Mapping[str,Any]]) -> tuple[Prediction,...]:
    """Project selected families without reading optional labels or future prices."""
    out=[];keys=set()
    for row in rows:
        date=day(row['date'])
        require('2022-01-01'<=date<='2025-12-31','Non-development prediction')
        identity=tuple(row[k] for k in ('scope','model','task'))
        if identity not in (FAMILIES['A'],FAMILIES['B']):
            continue
        require(row['fold_id']=='test_'+date[:4],'Prediction fold/date mismatch')
        segment=int(row['price_segment'])
        require(str(segment)==str(row['price_segment']) and segment>=0,'Invalid segment')
        value=number(row['prediction'])
        require(identity[2]!='classification' or 0<=value<=1,'Invalid probability')
        require(bool(row['exchange']) and bool(row['asset_code']),'Missing asset identity')
        p=Prediction(date,row['exchange'],row['asset_code'],segment,row['fold_id'],*identity,value)
        key=(p.date,p.exchange,p.asset_code,p.price_segment,p.fold_id,*identity)
        require(key not in keys,'Duplicate prediction key');keys.add(key);out.append(p)
    return tuple(sorted(out))

def schedule(calendar: Iterable[str], rules: Rules) -> list[str]:
    """Global domestic-session anchor; no per-asset or annual resets."""
    dates=list(calendar)
    require(len(dates)==len(set(dates)) and dates,'Empty/duplicate session calendar')
    require(all('2018-01-01'<=day(d)<='2025-12-31' for d in dates),'Non-development calendar')
    require(type(rules.rebalance_frequency_days) is int and rules.rebalance_frequency_days>0,'Invalid frequency')
    require(day(rules.anchor) in dates and day(rules.end)<='2025-12-31','Anchor/end not in development')
    active=sorted(d for d in dates if rules.anchor<=d<=rules.end)
    return active[::rules.rebalance_frequency_days]

def validate_state(state: DecisionState) -> None:
    """Reject impossible balances/identities without inventing valuation prices."""
    require('2018-01-01'<=day(state.date)<='2025-12-31','Non-development decision')
    require(number(state.nav)>0 and 0<=number(state.cash_available)<=number(state.nav),'Invalid decision cash/NAV')
    require(len({a.asset_code for a in state.assets})==len(state.assets),'Duplicate asset state')
    for a in state.assets:
        require(a.asset_code and a.exchange,'Incomplete asset identity')
        require(number(a.shares)>=0 and number(a.lot_size)>0,'Invalid shares/lot')
        require(type(a.price_segment) is int and a.price_segment>=0,'Invalid segment')
        require(all(type(x) is bool for x in (a.tradable_now,a.basis_valid,a.pending_action)),'Invalid state flag')
        first=day(a.first_tradable_date)
        require(a.last_tradable_date is None or day(a.last_tradable_date)>=first,'Invalid trading interval')
        require(state.date>=first or number(a.shares)==0,'Prelisting holdings forbidden')

def availability(a: AssetState,date: str) -> str:
    """Only information known at t; t+1 execution eligibility is a later check."""
    if date<a.first_tradable_date:return 'prelisting'
    if a.last_tradable_date is not None and date>a.last_tradable_date:return 'not_tradable'
    if a.pending_action:return 'pending_corporate_action'
    if not a.tradable_now:return 'not_tradable'
    if not a.basis_valid:return 'invalid_price_basis'
    try:
        if a.close is None or number(a.close)<=0:return 'missing_legal_decision_close'
    except ValueError:
        return 'missing_legal_decision_close'
    return 'available'

def targets(strategy: str,predictions: Iterable[Prediction],state: DecisionState,rules: Rules) -> list[dict]:
    """Equal weights after strict fixed eligibility; unavailable inputs are not zeros."""
    require(strategy in FAMILIES,'Unknown strategy family');validate_state(state)
    require(0<number(rules.max_asset_weight)<=number(rules.max_total_weight)<=1,'Invalid exposure limits')
    selected={}
    if strategy!='C':
        for p in predictions:
            if p.date==state.date and (p.scope,p.model,p.task)==FAMILIES[strategy]:
                key=(p.exchange,p.asset_code,p.price_segment)
                require(key not in selected,'Duplicate decision prediction');selected[key]=p
    rows=[]
    for a in sorted(state.assets,key=lambda x:x.asset_code):
        reason=availability(a,state.date);value=None
        if reason=='available' and strategy!='C':
            p=selected.get((a.exchange,a.asset_code,a.price_segment))
            if p is None:reason='missing_prediction'
            else:
                value=p.prediction
                hurdle=number(rules.classification_threshold if strategy=='A' else rules.regression_hurdle)
                reason='eligible' if value>hurdle else 'below_or_equal_hurdle'
        elif reason=='available':reason='eligible'
        rows.append({'asset_code':a.asset_code,'exchange':a.exchange,'price_segment':a.price_segment,
                     'prediction':value,'reason':reason,'weight':Decimal(0)})
    n=sum(r['reason']=='eligible' for r in rows)
    if n:
        with localcontext() as ctx:
            ctx.prec=40
            weight=min(number(rules.max_asset_weight),number(rules.max_total_weight)/n)
            weight=weight.quantize(Decimal('1e-18'),rounding=ROUND_DOWN)
            for row in rows:
                if row['reason']=='eligible':row['weight']=weight
    return rows
