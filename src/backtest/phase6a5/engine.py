"""Research posting extension; delayed information is exercised by synthetic cases only."""
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING, localcontext
from src.backtest.accounting import ZERO
from src.backtest.contracts import Action, Asset, OrderIntent, Policy, Quote, Result, number, require, timestamp
from src.backtest.engine import _Simulation, _prepare


@dataclass(frozen=True)
class DelayedShareAction(Action):
    """Synthetic pending event; empty known_at/None ratio means never resolved.

    The effective event is announced in advance, but the final ratio is not.
    Only exact share conversion is supported for this synthetic late path.
    """
    announced_at: str = ''

    def issues(self) -> list[str]:
        """Validate structural terms without demanding a not-yet-known ratio."""
        try:
            require(self.kind in {'split','conversion'} and self.phase=='after_close','Unsupported delayed event phase/type')
            require(self.rounding=='exact','Delayed rounding terms not supported')
            require(self.record_date==self.effective_date,'Delayed record must equal effective date')
            announced=timestamp(self.announced_at)
            require(announced<timestamp(self.effective_date+'T15:30:00+08:00'),'Event was not announced before effect')
            require(bool(self.known_at)==(self.share_ratio is not None),'Partial resolution terms')
            if self.known_at:
                require(timestamp(self.known_at).date().isoformat()>self.effective_date,'Use ordinary Action for advance/same-day known terms')
            # Base validation checks all other mandatory fields using the advance notice.
            base=Action(**{k:getattr(self,k) for k in Action.__dataclass_fields__})
            return replace(base,known_at=self.announced_at,share_ratio=self.share_ratio or '1').issues()
        except (ValueError,TypeError) as exc:
            return [str(exc)]


@dataclass
class Replay:
    """Standard reconciled ledgers plus event-state audit, kept out of market data."""
    result: Result
    states: list[dict]
    resolved: bool


class ResearchSimulation(_Simulation):
    """Preserve Phase 6A mechanics, separate conversion from issuer rounding value."""
    def __init__(self,*args):
        super().__init__(*args)
        self.pending_actions: dict[str,DelayedShareAction]={}
        self.resolutions: dict[str,dict]={}
        self.states: list[dict]=[]

    def price(self,date: str,code: str,field: str) -> Decimal | None:
        if code in self.pending_actions:
            return None
        return super().price(date,code,field)

    def shares_event(self,date: str,action: Action) -> Decimal:
        if isinstance(action,DelayedShareAction):
            require(action.asset_code not in self.pending_actions,'Overlapping unresolved event')
            require(self.book.shares[action.asset_code]==self.entitlements.get(action.event_id,ZERO),'Pending entitlement mismatch')
            self.pending_actions[action.asset_code]=action
            self.book.record(date,'pending_corporate_action',action.event_id,self.book.snapshot(),ZERO)
            return ZERO
        # First post pure conversion; a separate entry explains any issuer-funded rounding.
        super().shares_event(date,replace(action,rounding='exact'))
        if action.rounding!='ceil_whole_share':
            return ZERO
        b=self.book;code=action.asset_code;exact=b.shares[code]
        rounded=exact.to_integral_value(rounding=ROUND_CEILING)
        before=b.snapshot();b.shares[code]=rounded
        value=(rounded-exact)*b.prices.get(code,ZERO)
        b.record(date,'share_adjustment',action.event_id+':issuer_rounding_compensation',before,value)
        return value

    def resolve(self,date: str,phase: str) -> None:
        cutoff=timestamp(date+('T09:30:00+08:00' if phase=='pre_open' else 'T23:59:59+08:00'))
        for code,a in sorted(list(self.pending_actions.items())):
            if not a.known_at or timestamp(a.known_at)>cutoff:
                continue
            # No ratio arithmetic, changed quantity or disclosure appears before this check.
            base=Action(**{k:getattr(a,k) for k in Action.__dataclass_fields__})
            self.shares_event(date,base)
            self.resolutions[a.event_id]={'date':date,'phase':phase,'ratio':a.share_ratio}
            del self.pending_actions[code]

    def mark(self,date: str,field: str) -> Decimal:
        # Close-after may use today's observed close only after the ratio becomes available.
        self.resolve(date,'pre_open' if field=='open' else 'close_after')
        return super().mark(date,field)

    def fill(self,date: str,order: dict) -> dict | None:
        if order['asset_code'] in self.pending_actions:
            order['status']='cancelled_pending_corporate_action'
            return None
        return super().fill(date,order)

    def decisions(self,date: str,next_date: str | None) -> None:
        start=len(self.tables['orders'])
        held_pending=any(self.book.shares[c]>0 for c in self.pending_actions)
        original=self.intents
        blocked={o.order_id for o in original if o.decision_date==date and
                 (o.asset_code in self.pending_actions or (held_pending and o.target_weight is not None))}
        # Prevent sizing against a provisional portfolio value, even for another asset.
        self.intents=[replace(o,requested_shares='1',target_weight=None) if o.order_id in blocked else o for o in original]
        super().decisions(date,next_date)
        self.intents=original
        lookup={o.order_id:o for o in original}
        for row in self.tables['orders'][start:]:
            if row['order_id'] in blocked:
                o=lookup[row['order_id']]
                row.update(requested_shares=o.requested_shares,target_weight=o.target_weight,frozen_shares=ZERO,
                    status='cancelled_pending_corporate_action' if o.asset_code in self.pending_actions else 'cancelled_pending_portfolio_valuation')
                if next_date in self.pending:self.pending[next_date]=[r for r in self.pending[next_date] if r is not row]
        for a in self.actions:
            if not isinstance(a,DelayedShareAction):continue
            info=self.resolutions.get(a.event_id)
            pending=a.asset_code in self.pending_actions
            self.states.append({'date':date,'event_id':a.event_id,'asset_code':a.asset_code,
                'status':'resolved' if info else ('pending' if pending else 'not_effective'),
                'ratio_visible':info['ratio'] if info else None,'resolved_phase':info['phase'] if info else None,
                'valuation_status':'stale_economic_valuation' if pending else 'ordinary_valuation',
                'valuation_only':True,'model_feature_input':False})


def run_research(policy: Policy,assets: list[Asset],sessions: list[str],quotes: list[Quote],
                 intents: list[OrderIntent],actions: list[Action]=()) -> Replay:
    """Research-only replay; no data mutation, model calls or order generation."""
    with localcontext() as ctx:
        ctx.prec=40
        # Replaying a window that begins mid-pending cannot silently omit the event.
        for a in actions:
            if isinstance(a,DelayedShareAction):
                require(not a.issues(),'Incomplete delayed event: '+','.join(a.issues()))
                require(min(sessions)<=a.effective_date,'Delayed window must include effective date')
                require(sum(b.asset_code==a.asset_code for b in actions)==1,'Delayed event cannot overlap other corporate actions')
        prepared=_prepare(policy,list(assets),list(sessions),list(quotes),list(intents),list(actions))
        sim=ResearchSimulation(policy,*prepared)
        previous=number(policy.initial_cash)
        for i,d in enumerate(sim.sessions):
            previous=sim.session(d,sim.sessions[i+1] if i+1<len(sim.sessions) else None,previous)
        sim.tables['journal']=sim.book.journal
        return Replay(Result(sim.tables),sim.states,not sim.pending_actions)
