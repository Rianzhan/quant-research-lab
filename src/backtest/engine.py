"""Deterministic session execution of frozen manual intents, with no model API."""
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING, localcontext
from src.backtest.accounting import Book, ZERO, reconcile_day
from src.backtest.contracts import Action, Asset, OrderIntent, Policy, Quote, Result, day, number, require, timestamp
from src.backtest.costs import execution_cost, floor_lot


def legal_price(quote: Quote | None, field: str) -> Decimal | None:
    """A missing/invalid observation is not a tradable or fresh valuation price."""
    if quote is None or quote.suspended:
        return None
    try:
        value=number(getattr(quote,field))
        require(value>0, 'Price must be positive')
        if quote.high is not None or quote.low is not None:
            require(0<number(quote.low)<=value<=number(quote.high), 'OHLC bounds invalid')
        return value
    except ValueError:
        return None


def _prepare(policy: Policy, assets: list[Asset], sessions: list[str], quotes: list[Quote],
             intents: list[OrderIntent], actions: list[Action]) -> tuple:
    """Validate the whole supplied window before the first ledger mutation."""
    policy.validate()
    require(sessions and len(sessions)==len(set(sessions)), 'Empty/duplicate calendar sessions')
    sessions=sorted(sessions)
    require(all(policy.start<=day(x)<=policy.end for x in sessions), 'Non-development session')
    asset_map={a.asset_code:a for a in assets}
    require(assets and len(asset_map)==len(assets), 'Empty/duplicate assets')
    for a in assets:
        require(a.asset_code and number(a.lot_size)>0, 'Invalid asset/lot')
        day(a.first_tradable_date)
        require(a.last_tradable_date is None or day(a.last_tradable_date)>=a.first_tradable_date, 'Invalid tradability interval')
    qmap={}
    for q in quotes:
        require(policy.start<=day(q.date)<=policy.end, 'Non-development market row')
        require(q.date in sessions and q.asset_code in asset_map, 'Quote outside supplied calendar/assets')
        require(type(q.suspended) is bool, 'Suspension flag must be Boolean')
        require((q.date,q.asset_code) not in qmap, 'Duplicate asset/date quote')
        qmap[q.date,q.asset_code]=q
    ids=set()
    for o in intents:
        require(o.order_id and o.order_id not in ids, 'Duplicate/empty order id');ids.add(o.order_id)
        require(o.asset_code in asset_map and o.decision_date in sessions, 'Order outside assets/development sessions')
        require(o.side in {'buy','sell'} and o.source in {'manual_test','frozen_manual_fixture'}, 'Only manual research intents supported')
        timestamp(o.information_cutoff)
        require(o.information_cutoff==o.decision_date+'T23:59:59+08:00', 'Decision information cutoff must be same-day close-after')
        require((o.requested_shares is None)!=(o.target_weight is None), 'Specify shares OR target weight')
        if o.requested_shares is not None:
            require(number(o.requested_shares)>0, 'Requested shares must be positive')
        else:
            require(0<=number(o.target_weight)<=1, 'Target weight implies leverage')
    require(len({a.event_id for a in actions})==len(actions), 'Duplicate event id')
    active=[]
    for a in actions:
        require(a.asset_code in asset_map, 'Event asset not registered')
        require(policy.start<=day(a.effective_date)<=policy.end, 'Non-development event')
        if not a.applicable:
            require(a.kind in {'split','conversion'} and a.effective_date<asset_map[a.asset_code].first_tradable_date,
                    'Only prelisting share events may be inapplicable')
            continue
        if sessions[0]<=a.effective_date<=sessions[-1] or (a.payment_date and sessions[0]<=a.payment_date<=sessions[-1]):
            require(not a.issues(), 'incomplete event '+a.event_id+': '+','.join(a.issues()))
            for d in [a.effective_date,a.record_date,a.payment_date]:
                if d is not None and sessions[0]<=d<=sessions[-1]:
                    require(d in sessions, 'Event date is not an included domestic session')
            active.append(a)
    return asset_map,sessions,qmap,sorted(intents,key=lambda o:(o.decision_date,o.asset_code,o.order_id)),sorted(active,key=lambda a:a.event_id)


class _Simulation:
    """Private mutable state; orders and market observations remain immutable."""

    def __init__(self, policy: Policy, assets: dict, sessions: list, quotes: dict,
                 intents: list, actions: list):
        self.policy=policy;self.assets=assets;self.sessions=sessions;self.quotes=quotes
        self.intents=intents;self.actions=actions
        self.book=Book(number(policy.initial_cash),sorted(assets),number(policy.epsilon))
        self.tables={name:[] for name in ['orders','fills','daily','holdings','receivables','journal']}
        self.pending: dict[str,list[dict]]={}
        self.entitlements: dict[str,Decimal]={}
        self.last_close: dict[str,Decimal]={}
        self.last_date: dict[str,str]={}
        self.basis_event: dict[str,str]={}
        self.halts={(d,a.asset_code) for a in actions for d in a.suspended_dates}

    def tradable(self, code: str, date: str) -> bool:
        a=self.assets[code]
        return a.first_tradable_date<=date and (a.last_tradable_date is None or date<=a.last_tradable_date)

    def price(self, date: str, code: str, field: str) -> Decimal | None:
        if (date,code) in self.halts or not self.tradable(code,date):
            return None
        return legal_price(self.quotes.get((date,code)),field)

    def mark(self, date: str, field: str) -> Decimal:
        """Revalue positions at raw observations; stale fallback is close-only."""
        total=ZERO
        for code in sorted(self.assets):
            price=self.price(date,code,field)
            if field=='close':
                if price is not None:
                    self.last_close[code]=price;self.last_date[code]=date;self.basis_event[code]=''
                else:
                    price=self.last_close.get(code)
                    require(price is not None or self.book.shares[code]==0,
                            'Held asset has no last legal close for valuation')
            if price is None:
                require(self.book.shares[code]==0 or code in self.book.prices, 'Missing valuation for held asset')
                continue
            before=self.book.snapshot()
            change=self.book.shares[code]*(price-self.book.prices.get(code,price))
            self.book.prices[code]=price
            if self.book.shares[code]>0:
                self.book.record(date,'market_'+field,code,before,change)
                total+=change
        return total

    def shares_event(self, date: str, action: Action) -> Decimal:
        """Apply verified ratio to quantity and valuation basis, without market writes."""
        b=self.book;code=action.asset_code;old=b.shares[code]
        require(old==self.entitlements.get(action.event_id,ZERO), 'Share-event entitlement mismatch')
        ratio=number(action.share_ratio);exact=old*ratio
        quantity=exact if action.rounding=='exact' else exact.to_integral_value(rounding=ROUND_CEILING)
        before=b.snapshot();b.shares[code]=quantity
        if code in b.prices:
            b.prices[code]/=ratio
        if code in self.last_close:
            self.last_close[code]/=ratio;self.basis_event[code]=action.event_id
        rounding=(quantity-exact)*b.prices.get(code,ZERO)
        b.record(date,'share_adjustment',action.event_id,before,rounding)
        return rounding

    def accrual(self, date: str, action: Action) -> Decimal:
        """Ex-date rights become an asset, but never spendable cash."""
        b=self.book;quantity=self.entitlements.get(action.event_id,ZERO)
        amount=quantity*number(action.cash_per_share);before=b.snapshot()
        b.receivable[action.event_id]=amount
        b.record(date,'dividend_accrual',action.event_id,before,amount)
        self.tables['receivables'].append({'date':date,'event_id':action.event_id,'asset_code':action.asset_code,
            'kind':'accrual','entitled_shares':quantity,'amount':amount,'receivable_after':amount})
        return amount

    def record_rights(self, date: str) -> None:
        """Snapshot record-close holdings; eligibility alone does not add NAV."""
        for a in self.actions:
            if a.record_date!=date:
                continue
            quantity=self.book.shares[a.asset_code];self.entitlements[a.event_id]=quantity
            self.book.record(date,'entitlement',a.event_id,self.book.snapshot(),ZERO)
            if a.kind=='cash_dividend':
                self.tables['receivables'].append({'date':date,'event_id':a.event_id,'asset_code':a.asset_code,
                    'kind':'entitlement','entitled_shares':quantity,'amount':quantity*number(a.cash_per_share),'receivable_after':ZERO})

    def pay(self, date: str) -> None:
        """Payment-date close: receivable to cash, with exactly zero NAV change."""
        for a in self.actions:
            if a.kind!='cash_dividend' or a.payment_date!=date:
                continue
            b=self.book;before=b.snapshot();amount=b.receivable.pop(a.event_id,ZERO);b.cash+=amount
            b.record(date,'dividend_payment',a.event_id,before,ZERO)
            self.tables['receivables'].append({'date':date,'event_id':a.event_id,'asset_code':a.asset_code,
                'kind':'payment','entitled_shares':self.entitlements.get(a.event_id,ZERO),'amount':amount,'receivable_after':ZERO})

    def fill(self, date: str, order: dict) -> dict | None:
        """One execution attempt at the immediately following session only."""
        code=order['asset_code'];b=self.book
        if any(a.asset_code==code and order['decision_date']<a.effective_date<=date for a in self.actions):
            order['status']='cancelled_due_to_corporate_action';return None
        if not self.tradable(code,date):
            order['status']='not_filled_not_tradable';return None
        quote=self.quotes.get((date,code))
        if (date,code) in self.halts or (quote is not None and quote.suspended):
            order['status']='not_filled_suspended';return None
        price=self.price(date,code,'open')
        if price is None:
            order['status']='not_filled_missing_open';return None
        lot=number(self.assets[code].lot_size);quantity=order['frozen_shares']
        unit=execution_cost(price,Decimal(1),order['side'],self.policy.fee_rate,self.policy.slippage)
        capacity=(b.available/(unit['effective_price']+unit['fee']) if order['side']=='buy' else b.shares[code])
        quantity=min(quantity,floor_lot(capacity,lot))
        if quantity<=0:
            order['status']='not_filled_cash_or_shares';return None
        costs=execution_cost(price,quantity,order['side'],self.policy.fee_rate,self.policy.slippage)
        before=b.snapshot();sign=1 if order['side']=='buy' else -1
        b.shares[code]+=sign*quantity
        b.cash-=sign*costs['executed_notional']+costs['fee']
        if sign==-1 and self.policy.sale_proceeds_reuse_policy=='next_session_only':
            b.restricted+=costs['executed_notional']-costs['fee']
        b.record(date,'trade',order['order_id'],before,-costs['fee']-costs['slippage_cost'])
        order['status']='filled' if quantity==order['frozen_shares'] else 'partially_filled_remainder_cancelled'
        result={'fill_id':order['order_id']+'-fill','order_id':order['order_id'],'decision_date':order['decision_date'],
            'date':date,'asset_code':code,'side':order['side'],'shares':quantity,'raw_open':price,
            **costs,'cash_after':b.cash,'cash_available_after':b.available,'shares_after':b.shares[code]}
        self.tables['fills'].append(result)
        return result

    def decisions(self, date: str, next_date: str | None) -> None:
        """Freeze quantity solely from t-close state, before examining t+1 quotes."""
        b=self.book;budget=b.available
        for o in [o for o in self.intents if o.decision_date==date]:
            row={**asdict(o),'execution_date':next_date,'status':'pending','frozen_shares':ZERO,
                 'decision_nav':b.nav,'decision_cash_available':b.available,'decision_shares':b.shares[o.asset_code],
                 'decision_close':(self.price(date,o.asset_code,'close')
                                   if not self.basis_event.get(o.asset_code) else None)}
            lot=number(self.assets[o.asset_code].lot_size)
            if o.requested_shares is not None:
                row['frozen_shares']=floor_lot(number(o.requested_shares),lot)
            else:
                price=row['decision_close']
                if price is None:
                    row['status']='cancelled_missing_decision_close'
                else:
                    target=floor_lot(number(o.target_weight)*b.nav/price,lot)
                    delta=target-b.shares[o.asset_code]
                    require(delta>=0 if o.side=='buy' else delta<=0, 'Target weight conflicts with side')
                    quantity=floor_lot(abs(delta),lot)
                    if o.side=='buy':
                        unit=execution_cost(price,Decimal(1),'buy',self.policy.fee_rate,self.policy.slippage)
                        cost=unit['effective_price']+unit['fee']
                        quantity=min(quantity,floor_lot(budget/cost,lot));budget-=quantity*cost
                    row['frozen_shares']=quantity
            if row['status']=='pending' and next_date is None:
                row['status']='cancelled_no_next_session'
            self.tables['orders'].append(row)
            if row['status']=='pending':
                self.pending.setdefault(next_date,[]).append(row)

    def session(self, date: str, next_date: str | None, previous_nav: Decimal) -> Decimal:
        """Execute the preregistered ten-step session order."""
        b=self.book;before=b.snapshot();b.restricted=ZERO
        b.record(date,'session_start','release_prior_sale_proceeds',before,ZERO)
        accrual=ZERO;rounding=ZERO
        today=[a for a in self.actions if a.effective_date==date]
        for a in today:
            if a.kind=='cash_dividend':
                accrual+=self.accrual(date,a)
            elif a.phase=='pre_open':
                rounding+=self.shares_event(date,a)
        market=self.mark(date,'open');pretrade=b.nav
        fees=ZERO;slip=ZERO;notional=ZERO
        for order in sorted(self.pending.get(date,[]),key=lambda o:(o['side']!='sell',o['asset_code'],o['order_id'])):
            fill=self.fill(date,order)
            if fill:
                fees+=fill['fee'];slip+=fill['slippage_cost'];notional+=fill['executed_notional']
        market+=self.mark(date,'close')
        self.record_rights(date)
        for a in today:
            if a.kind!='cash_dividend' and a.phase=='after_close':
                rounding+=self.shares_event(date,a)
        self.pay(date)
        for code in sorted(self.assets):
            price=b.prices.get(code)
            self.tables['holdings'].append({'date':date,'asset_code':code,'shares':b.shares[code],
                'valuation_price':price,'market_value':b.shares[code]*price if price is not None else ZERO,
                'valuation_stale':self.last_date.get(code)!=date,'last_raw_close_date':self.last_date.get(code),
                'price_basis_adjustment_event':self.basis_event.get(code,''),'valuation_only':True})
        row={'date':date,'previous_nav':previous_nav,'cash':b.cash,'cash_available':b.available,
             'restricted_sale_proceeds':b.restricted,'dividend_receivable':sum(b.receivable.values(),ZERO),
             'gross_asset_value':b.market_value,'nav':b.nav,'pre_trade_nav':pretrade,
             'executed_notional':notional,'turnover':notional/pretrade if notional else ZERO,
             'fees_paid':fees,'slippage_cost':slip,'market_revaluation':market,'dividend_accrual':accrual,
             'share_rounding_value':rounding,
             'unexplained_pnl':b.nav-previous_nav-market-accrual-rounding+fees+slip}
        reconcile_day(row,b.epsilon);row['daily_reconciliation']='passed';self.tables['daily'].append(row)
        self.decisions(date,next_date)
        return b.nav


def run(policy: Policy, assets: list[Asset], sessions: list[str], quotes: list[Quote],
        intents: list[OrderIntent], actions: list[Action]=()) -> Result:
    """Cash-only start; one deterministic replay, without I/O or strategy decisions."""
    with localcontext() as context:
        context.prec=40
        asset_map,dates,qmap,orders,events=_prepare(policy,list(assets),list(sessions),list(quotes),list(intents),list(actions))
        sim=_Simulation(policy,asset_map,dates,qmap,orders,events)
        previous=number(policy.initial_cash)
        for i,date in enumerate(dates):
            previous=sim.session(date,dates[i+1] if i+1<len(dates) else None,previous)
        sim.tables['journal']=sim.book.journal
        return Result(sim.tables)
