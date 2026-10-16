"""Append-only calendar context. Future dates are synthetic-only in Phase9B."""
from dataclasses import dataclass
from src.backtest.contracts import day, require


@dataclass(frozen=True)
class DateContext:
    """Explicit session identity, separate from the untouched development rules.

    There is deliberately no real-holdout authorization switch in this stage.
    synthetic=True is for in-memory fixtures, never a file-access permission.
    """
    development_sessions: tuple[str,...]
    future_sessions: tuple[str,...]=()
    synthetic: bool=False

    def __post_init__(self) -> None:
        require(type(self.synthetic) is bool,'Synthetic flag must be Boolean')
        for seq in [self.development_sessions,self.future_sessions]:
            require(type(seq) is tuple and list(seq)==sorted(set(seq)),'Calendar must be unique ordered tuple')
            for d in seq:day(d)
        require('2022-01-04' in self.development_sessions,'Missing frozen anchor')
        require(all('2018-01-01'<=d<='2025-12-31' for d in self.development_sessions),'Non-development date')
        require(not self.future_sessions or self.synthetic,'Real future context is not authorized')
        require(all('2026-01-01'<=d<='2026-09-30' for d in self.future_sessions),'Out-of-window future date')

    @property
    def calendar(self) -> tuple[str,...]:return self.development_sessions+self.future_sessions

    @property
    def active_sessions(self) -> tuple[str,...]:return tuple(d for d in self.calendar if d>='2022-01-04')

    @property
    def rebalance_dates(self) -> tuple[str,...]:return self.active_sessions[::5]

    @property
    def start(self) -> str:return self.calendar[0]

    @property
    def end(self) -> str:return self.calendar[-1]

    def validate_dates(self,dates) -> None:
        """Reject every date outside the supplied session context."""
        require(all(day(d) in self.calendar for d in dates),'Date outside explicit context')
