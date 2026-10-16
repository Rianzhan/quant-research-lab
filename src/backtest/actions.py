"""Translate frozen issuer evidence into auditable accounting events and gates."""
from dataclasses import asdict
from hashlib import sha256
from src.backtest.contracts import Action, Asset, stable_bytes


def applicability_issues(action: Action, assets: list[Asset] | None) -> list[str]:
    """Only an evidenced prelisting share event may bypass accounting completeness."""
    if action.applicable:
        return []
    asset=next((a for a in assets or [] if a.asset_code==action.asset_code),None)
    if asset is None:
        return ['inapplicability_requires_asset_listing_evidence']
    if action.kind not in {'split','conversion'} or action.effective_date>=asset.first_tradable_date:
        return ['inapplicable_event_is_not_a_prelisting_share_event']
    return []


def register_actions(research: dict, policy: dict) -> tuple[list[Action],list[dict]]:
    """Keep missing fields missing; augment only specifically reviewed share notices."""
    evidence={e['evidence_id']:e for e in research['evidence']}
    actions=[];rows=[]
    for e in sorted(research['events'],key=lambda e:(e['asset_code'],e['action_date'],e['event_id'])):
        ids=e['evidence_ids'];sources=[evidence[i] for i in ids]
        hashes={s['evidence_id']:s['sha256'] for s in sources}
        cash=e['kind']=='cash_dividend'
        details={} if cash else policy['share_event_details'].get(e['event_id'],{})
        known=max(e['announced_date'],e.get('ratio_known_at',e['announced_date']))
        a=Action(event_id=e['event_id'],asset_code=e['asset_code'],kind=e['kind'],
            known_at=known+'T23:59:59+08:00',effective_date=e['action_date'],
            phase=details.get('phase','pre_open'),record_date=e.get('registration_date',details.get('record_date')),
            payment_date=e.get('pay_date'),cash_per_share=str(e['cash_cny_per_share']) if 'cash_cny_per_share' in e else None,
            share_ratio=str(e['new_shares_per_old_share']) if 'new_shares_per_old_share' in e else None,
            rounding=details.get('rounding'),evidence_sha256=sha256(stable_bytes(hashes)).hexdigest() if sources else '',
            suspended_dates=tuple(details.get('suspended_dates',[])),applicable=details.get('applicable',True),
            limitation=details.get('limitation','Ordinary designated secondary-market account; payment available after close.'))
        actions.append(a)
        issues=a.issues()
        rows.append({**asdict(a),'announced_at':e['announced_date']+'T23:59:59+08:00',
            'price_effective_date':e.get('price_effective_date'),
            'accounting_complete':not issues if a.applicable else False,
            'gate_status':('ready_with_research_timing_assumptions' if not issues else 'blocked') if a.applicable else 'not_applicable_prelisting',
            'missing_or_invalid_fields':';'.join(issues),
            'record_date_present':a.record_date is not None,'payment_date_present':a.payment_date is not None,
            'cash_per_share_present':a.cash_per_share is not None,'share_ratio_present':a.share_ratio is not None,
            'evidence_ids':';'.join(ids),'evidence_hashes':hashes,'source_urls':';'.join(s['url'] for s in sources)})
    return actions,rows


def readiness(actions: list[Action], engine_verified: bool=False, assets: list[Asset] | None=None) -> dict:
    """Scope readiness separately; real broker/trading readiness is never inferred."""
    blockers=[]
    for a in actions:
        issues=a.issues() if a.applicable else applicability_issues(a,assets)
        if issues:blockers.append({'event_id':a.event_id,'issues':issues})
    complete=bool(actions) and not blockers
    return {'accounting_engine_ready':engine_verified,
            'corporate_action_accounting_ready':complete,
            'historical_execution_ready':engine_verified and complete,
            'backtest_total_return_ready':engine_verified and complete,
            'broker_rules_ready':False,'broker_realism_ready':False,
            'strategy_backtest_performed':False,'model_signals_used':False,
            'holdout_files_opened':False,'live_execution_ready':False,
            'scope':'cash_only_secondary_market_research_with_registered_assumptions',
            'blockers':blockers}
