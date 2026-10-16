"""Independent balance/flow assertions and an event journal for every mutation."""
from decimal import Decimal
from src.backtest.contracts import require

ZERO=Decimal('0')


def reconcile_day(row: dict, epsilon: Decimal) -> None:
    """Fail on either stock identity or independently accumulated flow mismatch."""
    require(abs(row['nav']-row['cash']-row['dividend_receivable']-row['gross_asset_value'])<=epsilon,
            'Daily balance identity broken')
    expected=(row['market_revaluation']+row['dividend_accrual']+row['share_rounding_value']
              -row['fees_paid']-row['slippage_cost'])
    require(abs(row['nav']-row['previous_nav']-expected)<=epsilon, 'Daily unexplained PnL')
    require(row['cash']>=0 and row['cash_available']>=0 and row['dividend_receivable']>=0,
            'Negative cash/receivable forbidden')
    require(0<=row['gross_asset_value']<=row['nav']+epsilon, 'Leverage forbidden')


class Book:
    """Cash, receivables, positions and valuation references; never market data."""

    def __init__(self, cash: Decimal, assets: list[str], epsilon: Decimal):
        self.cash=cash
        self.restricted=ZERO
        self.receivable: dict[str, Decimal]={}
        self.shares={code:ZERO for code in assets}
        self.prices: dict[str, Decimal]={}
        self.epsilon=epsilon
        self.journal: list[dict]=[]

    @property
    def market_value(self) -> Decimal:
        return sum((n*self.prices.get(code,ZERO) for code,n in self.shares.items()),ZERO)

    @property
    def nav(self) -> Decimal:
        return self.cash+sum(self.receivable.values(),ZERO)+self.market_value

    @property
    def available(self) -> Decimal:
        return self.cash-self.restricted

    def snapshot(self) -> tuple[Decimal, Decimal, Decimal]:
        return self.cash,sum(self.receivable.values(),ZERO),self.market_value

    def check(self) -> None:
        require(self.cash>=0 and self.available>=0, 'Negative cash forbidden')
        require(all(x>=0 for x in self.shares.values()), 'Negative shares forbidden')
        require(all(x>=0 for x in self.receivable.values()), 'Negative receivable forbidden')
        require(all(code in self.prices for code,n in self.shares.items() if n>0), 'Unvalued position')
        require(self.market_value<=self.nav+self.epsilon, 'Leverage forbidden')

    def record(self, date: str, kind: str, reference: str,
               before: tuple[Decimal,Decimal,Decimal], expected: Decimal) -> None:
        """Check each mutation against an independently supplied economic delta."""
        self.check()
        after=self.snapshot()
        actual=sum(after,ZERO)-sum(before,ZERO)
        residual=actual-expected
        require(abs(residual)<=self.epsilon, 'Event unexplained PnL: '+reference)
        self.journal.append({'date':date,'sequence':len(self.journal)+1,'kind':kind,'reference':reference,
            'cash_delta':after[0]-before[0],'receivable_delta':after[1]-before[1],
            'market_value_delta':after[2]-before[2], 'nav_before':sum(before,ZERO),
            'nav_after':sum(after,ZERO),'expected_nav_change':expected,'unexplained_pnl':residual})
