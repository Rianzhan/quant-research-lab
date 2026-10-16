"""The provider boundary must fail visibly, with no implicit source changes."""
import importlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]

def api():
    assert (ROOT/'src/data/providers.py').exists(), 'provider boundary not implemented'
    return importlib.import_module('src.data.providers')

def req():
    from src.data.contracts import DataRequest
    return DataRequest('159915','SZSE','akshare_sina','2025-12-22','2025-12-31',purpose='probe')

def test_request_parameters_preserve_domestic_symbol():
    api()
    assert req().interface()['params']=={'symbol':'sz159915'}
    assert req().interface()['returns_full_history'] is True

def test_bounded_worker_timeout_is_an_explicit_failure(monkeypatch):
    p=api()
    def timeout(*args,**kwargs):
        assert kwargs['timeout']==7
        raise subprocess.TimeoutExpired(args[0],7)
    monkeypatch.setattr(p.subprocess,'run',timeout)
    with pytest.raises(p.ProviderError,match='timeout'): p.fetch_prices(req(),7)

def test_worker_failure_not_retried_on_another_source(monkeypatch):
    p=api(); calls=[]
    def failed(*args,**kwargs):
        calls.append(json.loads(kwargs['input']))
        return subprocess.CompletedProcess(args[0],1,'','remote disconnected')
    monkeypatch.setattr(p.subprocess,'run',failed)
    with pytest.raises(p.ProviderError,match='disconnected'): p.fetch_prices(req(),7)
    assert len(calls)==1 and calls[0]['source']=='akshare_sina'

def test_worker_response_retains_vendor_notes(monkeypatch):
    p=api()
    def response(*args,**kwargs):
        return subprocess.CompletedProcess(args[0],0,json.dumps({'csv':'date,close\n2025-12-22,3.14\n','notes':['vendor note']}),'')
    monkeypatch.setattr(p.subprocess,'run',response)
    frame=p.fetch_prices(req(),7)
    assert frame.close.tolist()==['3.14'] and frame.attrs['provider_notes']==['vendor note']

def test_pooled_scope_can_be_configured_without_training():
    from src.utils.config import load_config,validate_config
    cfg=load_config(ROOT/'config/config.yaml'); cfg['models']['scope']='pooled'
    validate_config(cfg)

def test_download_config_timeout_and_ranges_reject_bad_values():
    from src.utils.config import load_config,validate_config,ConfigError
    cfg=load_config(ROOT/'config/config.yaml')
    assert 'download' in cfg['data'], 'data download configuration not implemented'
    cfg['data']['download']['timeout_seconds']=0
    with pytest.raises(ConfigError): validate_config(cfg)
