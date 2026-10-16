"""Single-side explicit fees and execution-price slippage, counted once."""
from decimal import Decimal, ROUND_FLOOR
from src.backtest.contracts import number, require


def floor_lot(quantity: Decimal, lot: Decimal) -> Decimal:
    """Round a nonnegative order down to a registered research trading unit."""
    require(quantity>=0 and lot>0, 'Invalid shares or lot')
    return (quantity/lot).to_integral_value(rounding=ROUND_FLOOR)*lot


def execution_cost(open_price: Decimal, shares: Decimal, side: str,
                   fee_rate: str, slippage: str) -> dict[str, Decimal]:
    """Return effective notional, fee and slippage; no round-trip shortcuts."""
    require(side in {'buy','sell'} and open_price>0 and shares>=0, 'Invalid execution')
    effective=open_price*(1+number(slippage)*(1 if side=='buy' else -1))
    notional=shares*effective
    return {'effective_price':effective,'executed_notional':notional,
            'fee':notional*number(fee_rate),'slippage_cost':shares*abs(effective-open_price)}
