"""Connect frozen decision rules to accepted accounting through its close hook."""
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import localcontext
from hashlib import sha256
from src.backtest.contracts import Asset,Action,Policy,Quote,Result,canonical,number,require,stable_bytes
from src.backtest.phase6a5.publication import validate_research
from src.portfolio.orders import StrategyExecution,map_orders,prepare_execution
from src.portfolio.rules import AssetState,DecisionState,Prediction,Rules,FAMILIES,schedule,rules_hash


@dataclass
class AccountRun:
    """One independent account plus decision provenance; diagnostics are separate."""
    strategy: str
    policy: Policy
    rule_hash: str
    tables: dict[str,list[dict]]
    signals: list[dict]
    decisions: list[dict]
    order_provenance: list[dict]

    def fingerprint(self) -> str:
        """Economic output identity, independent of input order and wall clock."""
        return sha256(stable_bytes(asdict(self))).hexdigest()


class AccountSimulation(StrategyExecution):
    """Override only order generation after the inherited session has posted t close."""

    def __init__(self,base: StrategyExecution,strategy: str,predictions: tuple[Prediction,...],
                 exchanges: dict[str,str],segments: dict[tuple[str,str],int],rules: Rules):
        super().__init__(base.policy,base.assets,base.sessions,base.quotes,[],base.actions)
        self.strategy=strategy;self.rules=rules;self.exchanges=exchanges;self.segments=segments
        self.planned=set(schedule(self.sessions,rules));self.predictions=defaultdict(list)
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
            mapped=map_orders(self.strategy,self.predictions[date],state,self.rules,self.sessions)
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


def run_account(strategy: str,policy: Policy,predictions: tuple[Prediction,...],*,assets: list[Asset],
                sessions: list[str],quotes: list[Quote],exchanges: dict[str,str],
                segments: dict[tuple[str,str],int],actions: list[Action],rules: Rules=Rules()) -> AccountRun:
    """One frozen account. Changes of fee affect accounting, never the mapping rules."""
    require(strategy in FAMILIES,'Only frozen A/B/C supported')
    require(rules==Rules(),'Frozen Phase6B rules changed')
    require(policy.fee_rate in rules.fee_scenarios and policy.slippage==rules.slippage,'Unregistered cost scenario')
    require(policy.initial_cash=='100000' and policy.sale_proceeds_reuse_policy=='next_session_only',
            'Frozen capital/cash reuse policy changed')
    require(set(exchanges)=={a.asset_code for a in assets},'Exchange/asset identity mismatch')
    dates=sorted(sessions)
    require(dates and dates[0]==rules.anchor and dates[-1]<=rules.end,'Strategy window must begin at frozen anchor')
    require(all('2018-01-01'<=d<='2025-12-31' for d,_ in segments),'Non-development segment metadata')
    base=prepare_execution(policy,list(assets),dates,list(quotes),[],list(actions),rules)
    sim=AccountSimulation(base,strategy,tuple(sorted(predictions)),exchanges,segments,rules)
    with localcontext() as ctx:
        ctx.prec=40;previous=number(policy.initial_cash)
        for i,date in enumerate(sim.sessions):
            previous=sim.session(date,sim.sessions[i+1] if i+1<len(sim.sessions) else None,previous)
        sim.tables['journal']=sim.book.journal
        validate_research(Result(sim.tables),initial_cash=number(policy.initial_cash))
    return AccountRun(strategy,policy,rules_hash(rules),sim.tables,sim.signals,sim.decision_rows,sim.provenance)
