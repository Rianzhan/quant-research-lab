"""Date adapters are exercised only with literal, synthetic observations."""
from dataclasses import asdict, replace
from decimal import Decimal
import importlib
from pathlib import Path

import pytest
import src

from src.backtest.contracts import Action, Asset, Policy, Quote, stable_bytes
from src.backtest.phase6a5.engine import DelayedShareAction
from src.backtest.phase7.engine import run_account
from src.backtest.pipeline import csv_bytes
from src.portfolio.rules import FAMILIES, Prediction, Rules, rules_hash


DAYS = (
    '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07',
    '2022-01-10', '2022-01-11', '2022-01-12', '2022-01-13',
    '2022-01-14', '2022-01-17', '2022-01-18', '2022-01-19',
)


def api():
    root = Path(src.__file__).resolve().parents[1]
    assert (root / 'src/final_holdout/context.py').exists(), 'DateContext adapter missing'
    assert (root / 'src/final_holdout/execution.py').exists(), 'Execution adapter missing'
    return (importlib.import_module('src.final_holdout.context').DateContext,
            importlib.import_module('src.final_holdout.execution').run_account_context)


def inputs(days=DAYS):
    return dict(assets=[Asset('X', '2018-01-02')], sessions=list(days),
                quotes=[Quote(d, 'X', '10', '10') for d in days],
                exchanges={'X': 'SSE'}, segments={(d, 'X'): 0 for d in days}, actions=[])


def predictions(strategy='A', days=DAYS):
    family = FAMILIES[strategy]
    return tuple(Prediction(d, 'SSE', 'X', 0, 'test_' + d[:4], *family,
                            Decimal('.6' if strategy == 'A' else '.002')) for d in days)


def test_context_keeps_anchor_phase_across_synthetic_year_boundary():
    DateContext, _ = api()
    development = ('2018-01-02', '2022-01-04', '2025-12-29', '2025-12-30', '2025-12-31')
    future = ('2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08')
    context = DateContext(development, future, synthetic=True)
    assert context.calendar == development + future
    assert context.active_sessions == development[1:] + future
    assert context.rebalance_dates == ('2022-01-04', '2026-01-06')
    assert (context.start, context.end) == ('2018-01-02', '2026-01-08')
    context.validate_dates(('2022-01-04', '2026-01-07'))
    with pytest.raises(ValueError):
        context.validate_dates(('2026-01-09',))


@pytest.mark.parametrize('development,future,synthetic', [
    (DAYS, ('2026-01-05',), False),
    (('2022-01-04', '2026-01-05'), (), True),
    (('2017-12-29', '2022-01-04'), (), True),
    ((DAYS[1], DAYS[0]), (), True),
    ((DAYS[0], DAYS[0]), (), True),
    ((DAYS[1],), (), True),
    (DAYS, ('2025-12-31',), True),
    (DAYS, ('2026-01-06', '2026-01-05'), True),
    (DAYS, ('2026-01-05', '2026-01-05'), True),
    (DAYS, (), 1),
])
def test_context_rejects_unscoped_or_ambiguous_dates(development, future, synthetic):
    DateContext, _ = api()
    with pytest.raises(ValueError):
        DateContext(development, future, synthetic=synthetic)


@pytest.mark.parametrize('strategy', ['A', 'B', 'C'])
@pytest.mark.parametrize('fee', Rules().fee_scenarios)
def test_context_path_matches_all_frozen_development_accounts_byte_for_byte(strategy, fee):
    DateContext, run_context = api()
    kwargs = inputs()
    preds = predictions(strategy) if strategy != 'C' else ()
    policy = replace(Policy(), fee_rate=fee)
    old = run_account(strategy, policy, preds, **kwargs)
    new = run_context(strategy, policy, preds, context=DateContext(DAYS, synthetic=True), **kwargs)
    assert old.fingerprint() == new.fingerprint()
    assert stable_bytes(asdict(old)) == stable_bytes(asdict(new))
    assert {name: csv_bytes(rows) for name, rows in old.tables.items()} == {
        name: csv_bytes(rows) for name, rows in new.tables.items()}
    assert new.rule_hash == rules_hash(Rules())
    assert all(row['date'] > row['decision_date'] for row in new.tables['fills'])
    assert [row['date'] for row in new.decisions] == [DAYS[0], DAYS[5], DAYS[10]]


@pytest.mark.parametrize('scenario', ['gap', 'missing_open', 'dividend', 'split', 'delayed'])
def test_context_preserves_execution_and_corporate_action_accounting(scenario):
    DateContext, run_context = api()
    kwargs = inputs()
    if scenario == 'gap':
        kwargs['quotes'][1] = Quote(DAYS[1], 'X', '100', '100')
    elif scenario == 'missing_open':
        kwargs['quotes'][1] = Quote(DAYS[1], 'X', None, '10')
    elif scenario == 'dividend':
        kwargs['actions'] = [Action('div', 'X', 'cash_dividend', '2022-01-03T23:59:59+08:00',
                                   DAYS[2], record_date=DAYS[1], payment_date=DAYS[4],
                                   cash_per_share='1', evidence_sha256='a' * 64)]
        kwargs['quotes'] = [Quote(d, 'X', '10' if i < 2 else '9', '10' if i < 2 else '9')
                            for i, d in enumerate(DAYS)]
    elif scenario == 'split':
        kwargs['actions'] = [Action('split', 'X', 'split', '2022-01-03T23:59:59+08:00',
                                   DAYS[5], phase='after_close', record_date=DAYS[5],
                                   share_ratio='2', rounding='exact', evidence_sha256='a' * 64)]
        kwargs['quotes'] = [Quote(d, 'X', '10' if i <= 5 else '5', '10' if i <= 5 else '5')
                            for i, d in enumerate(DAYS)]
    else:
        kwargs['actions'] = [DelayedShareAction('late', 'X', 'conversion', DAYS[7] + 'T08:00:00+08:00',
            DAYS[5], phase='after_close', record_date=DAYS[5], share_ratio='2', rounding='exact',
            evidence_sha256='a' * 64, announced_at=DAYS[4] + 'T12:00:00+08:00')]
        kwargs['quotes'] = [Quote(d, 'X', '10' if i <= 5 else '5', '10' if i <= 5 else '5')
                            for i, d in enumerate(DAYS)]
    old = run_account('C', Policy(), (), **kwargs)
    new = run_context('C', Policy(), (), context=DateContext(DAYS, synthetic=True), **kwargs)
    assert stable_bytes(asdict(new)) == stable_bytes(asdict(old))
    assert all(abs(r['unexplained_pnl']) <= Decimal('1e-8') for r in new.tables['daily'])


def test_synthetic_future_executes_on_actual_next_date_without_resetting_or_mutating_old_entry():
    DateContext, run_context = api()
    development = ('2022-01-04', '2025-12-29', '2025-12-30', '2025-12-31')
    future = ('2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08')
    context = DateContext(development, future, synthetic=True)
    kwargs = inputs(context.active_sessions)
    preds = tuple(replace(p, prediction=Decimal('.1')) if p.date == '2026-01-06' else p
                  for p in predictions(days=context.active_sessions))
    rule_before = rules_hash(Rules())
    new = run_context('A', Policy(), preds, context=context, **kwargs)
    assert [r['date'] for r in new.decisions] == ['2022-01-04', '2026-01-06']
    assert [(r['decision_date'], r['date'], r['side']) for r in new.tables['fills']] == [
        ('2022-01-04', '2025-12-29', 'buy'), ('2026-01-06', '2026-01-07', 'sell')]
    assert [r['execution_date'] for r in new.order_provenance] == ['2025-12-29', '2026-01-07']
    assert [r['fold_id'] for r in new.signals] == ['test_2022', 'test_2026']
    assert new.rule_hash == rule_before == rules_hash(Rules())
    with pytest.raises(ValueError):
        run_account('A', Policy(), preds, **kwargs)
    with pytest.raises(ValueError):
        replace(Policy(), end='2026-01-08').validate()


def test_context_rejects_skipped_sessions_and_data_outside_its_calendar():
    DateContext, run_context = api()
    context = DateContext(DAYS, synthetic=True)
    cases = []
    skipped = inputs(); skipped['sessions'] = [DAYS[0], *DAYS[2:]]; cases.append(((), skipped))
    quote = inputs(); quote['quotes'].append(Quote('2022-01-20', 'X', '10', '10')); cases.append(((), quote))
    segment = inputs(); segment['segments']['2022-01-20', 'X'] = 0; cases.append(((), segment))
    cases.append((predictions(days=('2022-01-20',)), inputs()))
    for preds, kwargs in cases:
        with pytest.raises(ValueError):
            run_context('A', Policy(), preds, context=context, **kwargs)


def test_context_execution_is_input_order_invariant_and_keeps_frozen_policy_checks():
    DateContext, run_context = api()
    context = DateContext(DAYS, synthetic=True)
    kwargs = inputs()
    expected = run_context('A', Policy(), predictions(), context=context, **kwargs)
    shuffled = {k: list(reversed(v)) if isinstance(v, list) else v for k, v in kwargs.items()}
    actual = run_context('A', Policy(), tuple(reversed(predictions())), context=context, **shuffled)
    assert expected.fingerprint() == actual.fingerprint()
    for policy in (replace(Policy(), fee_rate='.0007'), replace(Policy(), slippage='0'),
                   replace(Policy(), initial_cash='99999'), replace(Policy(), epsilon='0'),
                   replace(Policy(), sale_proceeds_reuse_policy='same_day_reuse')):
        with pytest.raises(ValueError):
            run_context('A', policy, predictions(), context=context, **kwargs)


def test_synthetic_fresh_future_account_keeps_global_phase():
    DateContext,run_context=api()
    past=('2022-01-04','2025-12-29','2025-12-30','2025-12-31')
    future=('2026-01-05','2026-01-06','2026-01-07','2026-01-08')
    context=DateContext(past,future,synthetic=True)
    result=run_context('C',Policy(),(),context=context,account_start=future[0],**inputs(future))
    assert [r['date'] for r in result.decisions]==['2026-01-06']
    assert result.tables['daily'][0]['nav']==Decimal('100000')
    assert result.tables['fills'][0]['date']=='2026-01-07'
