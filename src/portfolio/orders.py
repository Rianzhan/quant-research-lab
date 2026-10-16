"""Freeze quantities from t state; no strategy loop, future prices or P&L."""
from dataclasses import asdict,dataclass,replace
from decimal import Decimal,localcontext
from hashlib import sha256
from typing import Iterable
from src.backtest.contracts import Asset,Action,Policy,Quote,OrderIntent,number,require,stable_bytes,canonical
from src.backtest.costs import execution_cost,floor_lot
from src.backtest.engine import _prepare
from src.backtest.phase6a5.engine import ResearchSimulation,DelayedShareAction
from src.portfolio.rules import Rules,DecisionState,Prediction,availability,schedule,targets,rules_hash,validate_state

@dataclass(frozen=True)
class FrozenOrder:
    """Executable quantity plus immutable strategy and decision provenance."""
    intent: OrderIntent
    execution_date: str
    target_weight: Decimal
    decision_state_hash: str
    rule_hash: str
    decision_batch_hash: str

@dataclass(frozen=True)
class MappingResult:
    """No fills or NAV series; zero targets do not claim a successful liquidation."""
    strategy: str
    decision_date: str
    status: str
    targets: tuple[dict,...]
    orders: tuple[FrozenOrder,...]
    blocked: tuple[dict,...]

def map_orders(strategy: str,predictions: Iterable[Prediction],state: DecisionState,
               rules: Rules,calendar: Iterable[str]) -> MappingResult:
    """Schedule, select targets and freeze lot-rounded quantities before t+1."""
    dates=sorted(calendar);planned=schedule(dates,rules);validate_state(state)
    require(state.date in dates,'Decision is not a domestic session')
    if state.date not in planned:return MappingResult(strategy,state.date,'not_rebalance',(),(),())
    weights=targets(strategy,predictions,state,rules)
    if any(a.pending_action and number(a.shares)>0 for a in state.assets):
        return MappingResult(strategy,state.date,'blocked_pending_portfolio_valuation',tuple(weights),(),())
    i=dates.index(state.date)
    if i==len(dates)-1:return MappingResult(strategy,state.date,'no_next_development_session',tuple(weights),(),())
    next_date=dates[i+1];orders=[];blocked=[]
    normalized=replace(state,assets=tuple(sorted(state.assets,key=lambda a:a.asset_code)))
    state_hash=sha256(stable_bytes(asdict(normalized))).hexdigest();rh=rules_hash(rules)
    batch_hash=sha256(stable_bytes([rh,strategy,state.date,state_hash,weights])).hexdigest()
    by_code={a.asset_code:a for a in state.assets}
    with localcontext() as ctx:
        ctx.prec=40;budget=number(state.cash_available)
        for row in weights:
            a=by_code[row['asset_code']];reason=availability(a,state.date)
            if reason!='available':
                blocked.append({'asset_code':a.asset_code,'reason':reason});continue
            price=number(a.close);lot=number(a.lot_size)
            desired=floor_lot(row['weight']*number(state.nav)/price,lot)
            delta=desired-number(a.shares)
            side='buy' if delta>0 else 'sell';quantity=floor_lot(abs(delta),lot)
            if side=='buy':
                unit=execution_cost(price,Decimal(1),'buy',rules.primary_fee_rate,rules.slippage)
                cost=unit['effective_price']+unit['fee']
                quantity=min(quantity,floor_lot(budget/cost,lot));budget-=quantity*cost
            else:quantity=min(quantity,floor_lot(number(a.shares),lot))
            if quantity<=0:continue
            oid=sha256(stable_bytes([rh,strategy,state.date,a.asset_code,side,quantity,row['weight'],state_hash,batch_hash])).hexdigest()
            intent=OrderIntent('6b-'+oid[:24],state.date,a.asset_code,side,state.date+'T23:59:59+08:00',
                               'strategy_spec:'+strategy,requested_shares=canonical(quantity))
            orders.append(FrozenOrder(intent,next_date,row['weight'],state_hash,rh,batch_hash))
    return MappingResult(strategy,state.date,'mapped',tuple(weights),tuple(orders),tuple(blocked))

class StrategyExecution(ResearchSimulation):
    """Preserve accounting precision and prohibit skipped/replayed execution sessions."""
    def __init__(self,*args):
        super().__init__(*args)
        self._next_session_index=0

    def session(self,date: str,next_date: str | None,previous_nav: Decimal) -> Decimal:
        i=self._next_session_index
        require(i<len(self.sessions) and date==self.sessions[i],'Execution sessions must advance exactly once in order')
        expected=self.sessions[i+1] if i+1<len(self.sessions) else None
        require(next_date==expected,'Cannot override the immediate next execution session')
        with localcontext() as ctx:
            ctx.prec=40
            nav=super().session(date,next_date,previous_nav)
        self._next_session_index+=1
        return nav

def prepare_execution(policy: Policy,assets: list[Asset],sessions: list[str],quotes: list[Quote],
                      orders: Iterable[FrozenOrder],actions: list[Action],rules: Rules=Rules()) -> ResearchSimulation:
    """Adapt provenance validation only; preserve every Phase6A.5 execution rule.

    Phase6A's validator accepts only manual sources. Validate structure using its
    allowed sentinel, then restore actual strategy sources before simulation.
    There is no historical strategy orchestration here; callers must supply orders.
    """
    orders=list(orders);dates=sorted(sessions);original=[];batches={};asset_dates=set()
    require(bool(dates),'Empty execution calendar')
    require(len({o.intent.source for o in orders})<=1,'Each strategy requires a separate account')
    for a in actions:
        if isinstance(a,DelayedShareAction):
            require(not a.issues() and dates[0]<=a.effective_date,'Invalid delayed event window')
            require(sum(b.asset_code==a.asset_code for b in actions)==1,'Delayed event overlap')
    for o in orders:
        intent=o.intent
        require(intent.source in {'strategy_spec:A','strategy_spec:B','strategy_spec:C'},'Invalid strategy source')
        require(intent.requested_shares is not None and intent.target_weight is None,'Quantity must be frozen at t')
        require(intent.decision_date in dates,'Decision outside sessions')
        i=dates.index(intent.decision_date)
        require(i+1<len(dates) and o.execution_date==dates[i+1],'Not the immediate next session')
        require(o.rule_hash==rules_hash(rules) and len(o.decision_state_hash)==64 and len(o.decision_batch_hash)==64,'Changed/missing order provenance')
        batch=(o.decision_state_hash,o.decision_batch_hash)
        require(batches.setdefault(intent.decision_date,batch)==batch,'Mixed snapshots or target sets at one decision date')
        key=(intent.decision_date,intent.asset_code)
        require(key not in asset_dates,'Duplicate asset order at one decision date');asset_dates.add(key)
        require(0<=o.target_weight<=number(rules.max_asset_weight),'Target exceeds frozen exposure cap')
        oid=sha256(stable_bytes([o.rule_hash,intent.source.split(':')[1],intent.decision_date,intent.asset_code,
                                intent.side,number(intent.requested_shares),o.target_weight,o.decision_state_hash,o.decision_batch_hash])).hexdigest()
        require(intent.order_id=='6b-'+oid[:24],'Frozen order identity changed')
        original.append(intent)
    structural=[replace(o,source='frozen_manual_fixture') for o in original]
    prepared=list(_prepare(policy,assets,sessions,quotes,structural,actions))
    prepared[3]=sorted(original,key=lambda o:(o.decision_date,o.asset_code,o.order_id))
    return StrategyExecution(policy,*prepared)
