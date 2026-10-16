"""Append-only execution context; arithmetic copied unchanged or inherited.

The three date-boundary functions below are static copies of frozen Phase6B
state/target/order mapping. Only date membership/scheduling is context-bound.
AccountSimulation's close hook is likewise unchanged except those dispatches.
All fills, fees, dividends, share events and reconciliation reuse Phase6A.5.
"""
from collections import defaultdict
from dataclasses import asdict, replace
from decimal import Decimal, ROUND_DOWN, localcontext
from hashlib import sha256
from typing import Iterable
from src.backtest.contracts import *
from src.backtest.costs import execution_cost, floor_lot
from src.backtest.phase6a5.publication import validate_research
from src.backtest.phase7.engine import AccountRun
from src.portfolio.orders import FrozenOrder, MappingResult, StrategyExecution, prepare_execution
from src.portfolio.rules import Rules, DecisionState, Prediction, AssetState, FAMILIES, rules_hash
from src.final_holdout.context import DateContext

class ContextPolicy(Policy):
    """Validate identical financial mechanics without changing old Policy."""
    def validate(self) -> None:
        Policy(**{**asdict(self),'start':'2018-01-01','end':'2025-12-31'}).validate()
        require('2018-01-01'<=day(self.start)<=day(self.end)<='2026-09-30','Context policy dates')

def validate_state(state: DecisionState) -> None:
    """Reject impossible balances/identities without inventing valuation prices."""
    day(state.date)  # The owning context validates exact membership before dispatch.
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

def map_orders(strategy: str,predictions: Iterable[Prediction],state: DecisionState,
               rules: Rules,calendar: Iterable[str],context: DateContext) -> MappingResult:
    """Schedule, select targets and freeze lot-rounded quantities before t+1."""
    dates=sorted(calendar);planned=context.rebalance_dates;context.validate_dates([state.date]);validate_state(state)
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


class ContextSimulation(StrategyExecution):
    """Override only order generation after the inherited session has posted t close."""

    def __init__(self,base: StrategyExecution,strategy: str,predictions: tuple[Prediction,...],
                 exchanges: dict[str,str],segments: dict[tuple[str,str],int],rules: Rules,context: DateContext):
        super().__init__(base.policy,base.assets,base.sessions,base.quotes,[],base.actions)
        self.strategy=strategy;self.rules=rules;self.exchanges=exchanges;self.segments=segments
        self.context=context;self.planned=set(context.rebalance_dates);self.predictions=defaultdict(list)
        for p in predictions:self.predictions[p.date].append(p)
        self.signals=[];self.decision_rows=[];self.provenance=[]

    def decision_state(self,date: str) -> DecisionState:
        """No diagnostic flags, future opens, future label ends or returns are inputs."""
        assets=[]
        for code,a in sorted(self.assets.items()):
            quote=self.quotes.get((date,code));price=self.price(date,code,'close')
            if price is not None:require((date,code) in self.segments,'Missing same-day segment identity')
            assets.append(AssetState(code,self.exchanges[code],a.first_tradable_date,
                canonical(price),canonical(self.book.shares[code]),self.segments.get((date,code),0),
                a.lot_size,self.tradable(code,date) and (date,code) not in self.halts and not (quote and quote.suspended),
                not bool(self.basis_event.get(code)),code in self.pending_actions,a.last_tradable_date))
        return DecisionState(date,canonical(self.book.nav),canonical(self.book.available),tuple(assets))

    def decisions(self,date: str,next_date: str | None) -> None:
        """Freeze exact Phase6B quantities, then let inherited accounting queue them."""
        self.intents=[]
        if date in self.planned:
            state=self.decision_state(date)
            mapped=map_orders(self.strategy,self.predictions[date],state,self.rules,self.sessions,self.context)
            keys={(p.exchange,p.asset_code,p.price_segment) for p in self.predictions[date]
                  if (p.scope,p.model,p.task)==FAMILIES[self.strategy]}
            self.signals.extend({'strategy':self.strategy,'date':date,'fold_id':'test_'+date[:4],**r,
                'prediction_available':None if self.strategy=='C' else (r['exchange'],r['asset_code'],r['price_segment']) in keys}
                for r in mapped.targets)
            self.decision_rows.append({'date':date,'status':mapped.status,'nav':state.nav,
                'cash_available':state.cash_available,'order_count':len(mapped.orders),'blocked':mapped.blocked,
                'rule_hash':rules_hash(self.rules)})
            self.provenance.extend(asdict(o) for o in mapped.orders)
            self.intents=[o.intent for o in mapped.orders]
        super().decisions(date,next_date)
        self.intents=[]



def run_account_context(strategy: str, policy: Policy, predictions: tuple[Prediction,...], *,
        context: DateContext, assets: list[Asset], sessions: list[str], quotes: list[Quote],
        exchanges: dict[str,str], segments: dict[tuple[str,str],int], actions: list[Action],
        account_start: str | None=None) -> AccountRun:
    """Run development-equivalence or literal synthetic fixtures, never file loads."""
    rules=Rules()
    policy.validate()
    require(strategy in FAMILIES,'Only frozen A/B/C supported')
    require(policy.fee_rate in rules.fee_scenarios and policy.slippage==rules.slippage,'Unregistered costs')
    require(policy.initial_cash=='100000' and policy.sale_proceeds_reuse_policy=='next_session_only','Changed financial policy')
    require(set(exchanges)=={a.asset_code for a in assets},'Asset/exchange mismatch')
    if account_start is not None:
        context.validate_dates([account_start])
        require(context.synthetic and account_start>='2026-01-01','Fresh future account requires synthetic fixture')
    expected=tuple(d for d in context.active_sessions if account_start is None or d>=account_start)
    require(tuple(sorted(sessions))==expected,'Skipped or duplicate context sessions')
    context.validate_dates([q.date for q in quotes]);context.validate_dates([d for d,_ in segments])
    context.validate_dates([p.date for p in predictions])
    for p in predictions:
        require(number(p.prediction).is_finite(),'Invalid prediction')
        require(p.task!='classification' or 0<=p.prediction<=1,'Invalid probability')
    scoped=ContextPolicy(**{**asdict(policy),'end':context.end})
    base=prepare_execution(scoped,list(assets),sorted(sessions),list(quotes),[],list(actions),rules)
    sim=ContextSimulation(base,strategy,tuple(sorted(predictions)),exchanges,segments,rules,context)
    with localcontext() as ctx:
        ctx.prec=40;previous=number(policy.initial_cash)
        for i,date in enumerate(sim.sessions):
            previous=sim.session(date,sim.sessions[i+1] if i+1<len(sim.sessions) else None,previous)
        sim.tables['journal']=sim.book.journal
        validate_research(Result(sim.tables),initial_cash=number(policy.initial_cash))
    return AccountRun(strategy,policy,rules_hash(rules),sim.tables,sim.signals,sim.decision_rows,sim.provenance)
