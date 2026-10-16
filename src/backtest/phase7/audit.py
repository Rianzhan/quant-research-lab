"""Descriptive execution/coverage audits; no strategy rankings or optimization."""
from collections import Counter,defaultdict
from decimal import Decimal,localcontext
from src.backtest.contracts import Result,number,require
from src.backtest.phase6a5.publication import validate_research
from src.backtest.phase7.engine import AccountRun

ZERO=Decimal(0)

def account_tables(run: AccountRun) -> tuple[dict,list | dict]:
    """Check the ledger then add a same-fill cost bridge, not a counterfactual run."""
    with localcontext() as ctx:
        ctx.prec=40
        validate_research(Result(run.tables),initial_cash=number(run.policy.initial_cash))
        tables={k:[dict(r) for r in v] for k,v in run.tables.items()}
        counts=Counter();max_weights=defaultdict(lambda:ZERO)
        by_date={r['date']:r for r in tables['daily']}
        for h in tables['holdings']:
            if h['shares']>0:counts[h['date']]+=1
            max_weights[h['date']]=max(max_weights[h['date']],h['market_value']/by_date[h['date']]['nav'])
        fees=slip=market=accrual=adjustment=ZERO
        for d in tables['daily']:
            fees+=d['fees_paid'];slip+=d['slippage_cost'];market+=d['market_revaluation']
            accrual+=d['dividend_accrual'];adjustment+=d['share_rounding_value']
            gross=d['nav']+fees+slip;residual=gross-number(run.policy.initial_cash)-market-accrual-adjustment
            require(abs(residual)<=number(run.policy.epsilon),'Cumulative unexplained PnL')
            d.update(net_account_value=d['nav'],same_fill_gross_account_value=gross,
                cumulative_explicit_fees=fees,cumulative_slippage_cost=slip,cumulative_market_revaluation=market,
                cumulative_dividend_accrual=accrual,cumulative_share_economic_effect=adjustment,
                gross_net_bridge_residual=residual,held_asset_count=counts[d['date']],
                cash_ratio=d['cash']/d['nav'],cash_available_ratio=d['cash_available']/d['nav'],
                gross_exposure=d['gross_asset_value']/d['nav'],max_asset_weight=max_weights[d['date']])
        tables['costs']=[{k:d[k] for k in ['date','fees_paid','slippage_cost','cumulative_explicit_fees',
            'cumulative_slippage_cost','net_account_value','same_fill_gross_account_value','gross_net_bridge_residual']} for d in tables['daily']]
        tables['turnover']=[{k:d[k] for k in ['date','pre_trade_nav','executed_notional','turnover']} for d in tables['daily']]
        tables['cancelled_unfilled']=[o for o in tables['orders'] if o['status']!='filled']
        tables['corporate_actions']=[j for j in tables['journal'] if j['kind'] in
            {'entitlement','dividend_accrual','dividend_payment','share_adjustment','pending_corporate_action'}]
        payment=sum((r['amount'] for r in tables['receivables'] if r['kind']=='payment'),ZERO)
        days=tables['daily'];n=len(days);orders=tables['orders'];last=days[-1]
        summary={'strategy':run.strategy,'fee_rate':run.policy.fee_rate,'slippage':run.policy.slippage,
            'start_date':days[0]['date'],'end_date':last['date'],'sessions':n,'initial_nav':number(run.policy.initial_cash),
            'terminal_nav':last['nav'],'same_fill_gross_terminal_value':last['same_fill_gross_account_value'],
            'explicit_fee_total':fees,'slippage_cost_total':slip,'dividend_cash_total':payment,
            'dividend_accrual_total':accrual,'share_economic_effect_total':adjustment,'market_revaluation_total':market,
            'executed_notional':sum((d['executed_notional'] for d in days),ZERO),
            'turnover_cumulative':sum((d['turnover'] for d in days),ZERO),
            'turnover_average_per_session':sum((d['turnover'] for d in days),ZERO)/n,
            'average_cash_ratio':sum((d['cash_ratio'] for d in days),ZERO)/n,
            'average_gross_exposure':sum((d['gross_exposure'] for d in days),ZERO)/n,
            'max_gross_exposure':max(d['gross_exposure'] for d in days),'max_actual_asset_weight':max(max_weights.values()),
            'average_held_assets':Decimal(sum(counts.values()))/n,'sessions_invested':sum(v>0 for v in counts.values()),
            'fills':len(tables['fills']),'orders':len(orders),'rebalance_dates':len(run.decisions),
            'rebalance_dates_with_orders':sum(d['order_count']>0 for d in run.decisions),
            'cancelled_orders':sum(o['status'].startswith('cancelled') for o in orders),
            'unfilled_orders':sum(o['status'].startswith('not_filled') for o in orders),
            'partial_remainder_cancelled_orders':sum(o['status']=='partially_filled_remainder_cancelled' for o in orders),
            'terminal_cash':last['cash'],'terminal_receivable':last['dividend_receivable'],
            'terminal_held_assets':last['held_asset_count'],'max_abs_daily_unexplained_pnl':max(abs(d['unexplained_pnl']) for d in days),
            'rule_hash':run.rule_hash,'account_hash':run.fingerprint()}
        tables['holding_count_distribution']=[{'held_assets':k,'sessions':v} for k,v in sorted(Counter(d['held_asset_count'] for d in days).items())]
    return tables,summary

def coverage(run: AccountRun,diagnostics: dict) -> tuple[list[dict],list[dict]]:
    """Post-replay join only. Missing supervised labels never feed order generation."""
    rows=[];groups=defaultdict(list)
    for s in run.signals:
        key=(s['date'],s['asset_code']);meta=diagnostics['availability'].get(key)
        available=s['prediction_available'];reason=s['reason']
        feature=meta['feature_available'] if meta else False
        label=meta['label_valid'] if meta else None
        remaining=sum(d>s['date'] for d in diagnostics['observed_dates'].get(s['asset_code'],[]))
        tail=s['date'].startswith('2025-') and remaining<5 and label is False
        status=diagnostics['asset_fold_status'].get((s['fold_id'],s['asset_code']),'unknown')
        if run.strategy!='C' and available is False and reason=='missing_prediction':
            if meta is None:reason='missing_feature_row'
            elif not feature:reason='feature_incomplete'
            elif status not in {'available','unknown'}:reason='fold_asset_'+status
            elif tail:reason='label_horizon_outside_development'
            elif label is False:reason='invalid_label_event_or_gap'
            else:reason='missing_frozen_prediction_unexplained'
        row={'strategy':run.strategy,'fold_id':s['fold_id'],'year':s['date'][:4],'asset_code':s['asset_code'],
            'exchange':s['exchange'],'rebalance_date':s['date'],'price_segment':s['price_segment'],
            'prediction_available':available,'prediction_required':run.strategy!='C','feature_available':feature,
            'strategy_eligible':s['reason']=='eligible','target_weight':s['weight'],'missing_reason':'' if s['reason']=='eligible' else reason,
            'signal_reason':s['reason'],'label_valid_diagnostic_only':label,'label_horizon_outside_development':tail,
            'fold_asset_status':status,'diagnostics_used_for_orders':False}
        rows.append(row);groups[(row['strategy'],row['year'],row['asset_code'])].append(row)
    summary=[]
    for (strategy,year,code),group in sorted(groups.items()):
        model=strategy!='C';count=len(group)
        missing=sum(r['prediction_available'] is False for r in group)
        summary.append({'strategy':strategy,'year':year,'asset_code':code,'rebalance_dates':count,
            'predictions_available':sum(r['prediction_available'] is True for r in group) if model else None,
            'prediction_coverage_rate':Decimal(count-missing)/count if model else None,
            'feature_available_dates':sum(r['feature_available'] for r in group),
            'zero_targets_below_hurdle':sum(r['target_weight']==0 and r['signal_reason']=='below_or_equal_hurdle' for r in group),
            'zero_targets_missing_prediction':sum(r['target_weight']==0 and r['signal_reason']=='missing_prediction' for r in group),
            'missing_due_to_end_label':sum(r['prediction_available'] is False and r['label_horizon_outside_development'] for r in group),
            'eligible_dates':sum(r['strategy_eligible'] for r in group)})
    return rows,summary
