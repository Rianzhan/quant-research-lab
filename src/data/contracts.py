"""Explicit requests keep assets, date scopes and providers isolated."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import re


@dataclass(frozen=True)
class DataRequest:
    """Requested observation window, distinct from a vendor's transport scope."""
    asset_code: str
    exchange: str
    source: str
    start_date: str
    end_date: str
    adjustment: str = 'none'
    purpose: str = 'research'

    def __post_init__(self) -> None:
        calendar = self.source == 'akshare_calendar'
        if self.source not in ('akshare_em', 'akshare_sina', 'baostock', 'akshare_calendar'):
            raise ValueError('Choose an explicit supported source; automatic fallback is prohibited')
        if calendar:
            if (self.asset_code, self.exchange) != ('CALENDAR', 'CN'):
                raise ValueError('Calendar identity must be CN:CALENDAR')
        elif not isinstance(self.asset_code, str) or not re.fullmatch(r'[0-9]{6}', self.asset_code) or self.exchange not in ('SSE', 'SZSE'):
            raise ValueError('Invalid domestic ETF identity')
        for value in (self.start_date, self.end_date):
            if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                raise ValueError('Dates must be YYYY-MM-DD')
            date.fromisoformat(value)
        if self.start_date > self.end_date:
            raise ValueError('Reversed date range')
        if self.adjustment not in ('none', 'qfq', 'hfq'):
            raise ValueError('Unknown adjustment')
        if self.adjustment != 'none' and (self.source != 'akshare_em' or self.purpose != 'adjustment_probe'):
            raise ValueError('Adjusted data may only be an explicit Eastmoney adjustment probe')
        if self.purpose not in ('research', 'coverage', 'probe', 'freshness', 'calendar', 'adjustment_probe'):
            raise ValueError('Unknown request purpose')

    @property
    def key(self) -> str:
        """Hash every semantic request field, including source and adjustment."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def symbol(self) -> str:
        """Domestic exchange prefix; cross-border ETFs still use SSE/SZSE."""
        return ('sh' if self.exchange == 'SSE' else 'sz') + self.asset_code

    def interface(self) -> dict:
        """Record exact SDK parameters, including APIs without range support."""
        if self.source == 'akshare_em':
            return {'function': 'akshare.fund_etf_hist_em', 'params': {
                'symbol': self.asset_code, 'period': 'daily',
                'start_date': self.start_date.replace('-', ''), 'end_date': self.end_date.replace('-', ''),
                'adjust': '' if self.adjustment == 'none' else self.adjustment}, 'returns_full_history': False}
        if self.source == 'akshare_sina':
            return {'function': 'akshare.fund_etf_hist_sina', 'params': {'symbol': self.symbol}, 'returns_full_history': True}
        if self.source == 'akshare_calendar':
            return {'function': 'akshare.tool_trade_date_hist_sina', 'params': {}, 'returns_full_history': True}
        return {'function': 'baostock.query_history_k_data_plus', 'params': {
            'code': self.symbol[:2]+'.'+self.asset_code, 'fields': 'date,code,open,high,low,close,volume,amount,adjustflag,tradestatus',
            'start_date': self.start_date, 'end_date': self.end_date, 'frequency': 'd', 'adjustflag': '3'}, 'returns_full_history': False}
