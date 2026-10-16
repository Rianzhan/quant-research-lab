"""Synthetic Phase 3 data; no market or holdout observations are loaded here."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_public_suite(monkeypatch):
    """Public packaging guard: tests cannot contact data providers or the network."""
    import socket
    def deny(*args, **kwargs):
        raise AssertionError('Public synthetic tests must remain offline')
    monkeypatch.setattr(socket, 'create_connection', deny)
    monkeypatch.setattr(socket.socket, 'connect', deny)


@pytest.fixture
def phase3_config():
    return load_config(ROOT/'config/config.yaml')


@pytest.fixture
def make_panel(phase3_config):
    def make(days=120, start='2020-01-02', event_at=None):
        calendar = pd.bdate_range(start, periods=days).strftime('%Y-%m-%d').tolist()
        rows = []
        for j, asset in enumerate(phase3_config['universe']['assets'][:2]):
            for i, day in enumerate(calendar):
                price = 10+j*30+i*.1+np.sin(i/3)*.2
                rows.append({'date': day, 'exchange': asset['exchange'], 'asset_code': asset['code'],
                    'asset_name': asset['name'], 'category': asset['category'], 'close': price,
                    'research_close': price, 'volume': float(1000+i*7), 'research_volume': float(1000+i*7),
                    'volume_valid': True, 'price_segment': int(event_at is not None and i>=event_at),
                    'price_basis': 'none', 'research_price_policy': 'unadjusted_event_segmented_v1'})
        return pd.DataFrame(rows).sort_values(['date','exchange','asset_code']).reset_index(drop=True), calendar
    return make
