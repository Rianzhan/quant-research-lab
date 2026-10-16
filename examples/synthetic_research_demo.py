"""Demonstrate causal features and an audited manual-order cash-flow example."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.features.builder import build_features
from src.labels.forward_return import build_labels
from src.backtest.contracts import Action, Asset, OrderIntent, Policy, Quote
from src.backtest.engine import run


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    config=load_config(root/'config/config.yaml')
    calendar=pd.bdate_range('2020-01-02',periods=100).strftime('%Y-%m-%d').tolist()
    asset=config['universe']['assets'][0]
    panel=pd.DataFrame([dict(date=d,exchange=asset['exchange'],asset_code=asset['code'],
        asset_name=asset['name'],category=asset['category'],close=10+i*.01+np.sin(i/4)*.1,
        research_close=10+i*.01+np.sin(i/4)*.1,volume=1000.+i,research_volume=1000.+i,
        volume_valid=True,price_segment=0,price_basis='none',
        research_price_policy='unadjusted_event_segmented_v1') for i,d in enumerate(calendar)])
    original=build_features(panel,config,calendar)
    altered=panel.copy()
    altered.loc[altered.date>calendar[70],['close','research_close']]*=9
    changed=build_features(altered,config,calendar)
    pd.testing.assert_frame_equal(original.iloc[:71],changed.iloc[:71],check_exact=True)
    labels=build_labels(panel,config,calendar)
    assert labels.tail(5).classification_target.isna().all()

    days=['2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-08']
    quotes=[Quote(d,'SYNTHETIC','10' if i<2 else '9','10' if i<2 else '9') for i,d in enumerate(days)]
    order=OrderIntent('manual-1',days[0],'SYNTHETIC','buy',days[0]+'T23:59:59+08:00',
                      'manual_test',requested_shares='10')
    action=Action('dividend-fixture','SYNTHETIC','cash_dividend','2024-01-01T23:59:59+08:00',
                  days[2],record_date=days[1],payment_date=days[4],cash_per_share='1',evidence_sha256='a'*64)
    result=run(Policy(initial_cash='1000',fee_rate='0',slippage='0'),
               [Asset('SYNTHETIC',days[0])],days,quotes,[order],[action])
    assert result.tables['fills'][0]['date']==days[1]
    assert all(row['nav']==1000 for row in result.tables['daily'])
    print(json.dumps(dict(data='synthetic_only',historical_performance_claim=False,
        feature_rows=len(original),future_perturbation_invariant=True,missing_tail_labels_preserved=True,
        dividend_nav_conservation=True,next_session_execution=True,
        accounting_fingerprint=result.fingerprint()),indent=2))


if __name__=='__main__':main()
