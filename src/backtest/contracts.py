"""Immutable input contracts for a cash-only, long-only daily research engine."""
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Any


def require(condition: bool, message: str) -> None:
    """Fail closed rather than repair an accounting or input violation."""
    if not condition:
        raise ValueError(message)


def number(value: Any) -> Decimal:
    """Use decimal text, reject booleans and non-finite arithmetic."""
    require(not isinstance(value, bool), 'Boolean is not a monetary value')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError('Invalid numeric value') from exc
    require(result.is_finite(), 'Non-finite numeric value')
    return result


def day(value: str) -> str:
    """Require an unambiguous ISO calendar date."""
    require(isinstance(value, str) and date.fromisoformat(value).isoformat()==value, 'Invalid ISO date')
    return value


def timestamp(value: str) -> datetime:
    """Require an explicit China UTC offset; never infer publication times."""
    require(isinstance(value, str) and value.endswith('+08:00'), 'Timestamp requires +08:00')
    result = datetime.fromisoformat(value)
    require(result.utcoffset().total_seconds()==28800, 'Timestamp timezone mismatch')
    return result


def canonical(value: Any) -> Any:
    """Stable Decimal serialization independent of float conversion/context."""
    if isinstance(value, Decimal):
        text = format(value, 'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def stable_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON; NaN is forbidden."""
    return json.dumps(canonical(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class Policy:
    """Research mechanics, explicitly distinct from verified broker rules."""
    initial_cash: str = '100000'
    fee_rate: str = '0.0003'
    slippage: str = '0.0002'
    epsilon: str = '0.00000001'
    sale_proceeds_reuse_policy: str = 'next_session_only'
    start: str = '2018-01-01'
    end: str = '2025-12-31'

    def validate(self) -> None:
        require(number(self.initial_cash)>0, 'Initial cash must be positive')
        require(0<=number(self.fee_rate)<1 and 0<=number(self.slippage)<1, 'Invalid cost rate')
        require(0<number(self.epsilon)<=Decimal('0.00000001'), 'Invalid epsilon')
        require(self.sale_proceeds_reuse_policy in {'same_day_reuse','next_session_only'}, 'Unknown sale proceeds policy')
        require('2018-01-01'<=day(self.start)<=day(self.end)<='2025-12-31', 'Non-development policy')


@dataclass(frozen=True)
class Asset:
    """Observable secondary-market trading interval; lot is a research assumption."""
    asset_code: str
    first_tradable_date: str
    last_tradable_date: str | None = None
    lot_size: str = '1'


@dataclass(frozen=True)
class Quote:
    """Raw unadjusted daily observation, never amended by accounting."""
    date: str
    asset_code: str
    open: str | None
    close: str | None
    suspended: bool = False
    high: str | None = None
    low: str | None = None


@dataclass(frozen=True)
class OrderIntent:
    """Externally frozen instruction, exactly one of shares or target weight."""
    order_id: str
    decision_date: str
    asset_code: str
    side: str
    information_cutoff: str
    source: str
    requested_shares: str | None = None
    target_weight: str | None = None


@dataclass(frozen=True)
class Action:
    """Verified corporate action; incomplete fields are audited, never guessed."""
    event_id: str
    asset_code: str
    kind: str
    known_at: str
    effective_date: str
    phase: str = 'pre_open'
    record_date: str | None = None
    payment_date: str | None = None
    cash_per_share: str | None = None
    share_ratio: str | None = None
    rounding: str | None = None
    evidence_sha256: str = ''
    suspended_dates: tuple[str, ...] = ()
    applicable: bool = True
    limitation: str = ''

    def issues(self) -> list[str]:
        """Return required-field/time failures without changing readiness."""
        issues=[]
        try:
            require(self.event_id and self.asset_code, 'missing_identity')
            require(self.kind in {'cash_dividend','split','conversion'}, 'unknown_event_kind')
            require(self.phase in {'pre_open','after_close'}, 'unknown_effective_phase')
            require(len(self.evidence_sha256)==64 and all(c in '0123456789abcdef' for c in self.evidence_sha256), 'missing_evidence_hash')
            effective=day(self.effective_date)
            known=timestamp(self.known_at)
            cutoff=timestamp(effective+('T09:30:00+08:00' if self.phase=='pre_open' else 'T15:30:00+08:00'))
            require(known<=cutoff, 'terms_not_known_at_effective_time')
            require(self.record_date is not None, 'missing_record_date')
            require(day(self.record_date)<=effective, 'record_after_effective')
            require(known<=timestamp(self.record_date+'T23:59:59+08:00'), 'terms_not_known_by_record_close')
            if self.kind=='cash_dividend':
                require(self.record_date<effective, 'cash_record_must_precede_ex_date')
                require(self.phase=='pre_open', 'cash_accrual_must_be_pre_open')
                require(self.payment_date is not None, 'missing_payment_date')
                require(day(self.payment_date)>=effective, 'payment_before_ex_date')
                require(self.cash_per_share is not None and number(self.cash_per_share)>0, 'missing_positive_cash_per_share')
            else:
                require(self.share_ratio is not None and number(self.share_ratio)>0, 'missing_positive_share_ratio')
                require(self.rounding in {'exact','ceil_whole_share'}, 'missing_share_rounding_rule')
        except (ValueError, TypeError) as exc:
            issues.append(str(exc))
        return issues


@dataclass
class Result:
    """Deterministic audit tables; no performance ranking or strategy metrics."""
    tables: dict[str, list[dict]] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Hash all economic records, excluding wall clock/publication paths."""
        return hashlib.sha256(stable_bytes(self.tables)).hexdigest()
