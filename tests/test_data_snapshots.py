"""Exercise real disk snapshots with network-only fakes."""
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]

def api():
    assert (ROOT/'src/data/loader.py').exists(), 'snapshot store not implemented'
    return importlib.import_module('src.data.loader')

def request(**kw):
    c=importlib.import_module('src.data.contracts')
    return c.DataRequest(**{'asset_code':'510300','exchange':'SSE','source':'akshare_sina',
                           'start_date':'2024-01-02','end_date':'2024-01-04',**kw})

def frame(close=10):
    return pd.DataFrame({'date':['2024-01-02'],'open':[10],'high':[12],'low':[9],
                         'close':[close],'volume':[200],'amount':[2000]})

def test_cache_hit_never_downloads_and_hashes_are_verified(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    first=store.get(request(),lambda r:frame())
    def forbidden(r): raise AssertionError('cache hit attempted network')
    second=store.get(request(),forbidden)
    assert first.snapshot_id==second.snapshot_id and second.cache_hit
    assert first.manifest['request']['asset_code']=='510300'
    assert len(first.manifest['raw_sha256'])==64
    (first.path/'raw.csv').write_text('corrupted',encoding='utf-8')
    with pytest.raises(api().SnapshotError,match='hash'): store.get(request(),forbidden)

def test_failed_download_has_no_complete_snapshot_or_cache_pointer(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    def fail(r): raise ConnectionError('remote closed')
    with pytest.raises(api().SnapshotError): store.get(request(),fail)
    assert not list((tmp_path/'raw').glob('**/manifest.json'))
    assert not list((tmp_path/'cache').glob('requests/*.json'))
    assert len(list((tmp_path/'cache').glob('failures/*.json')))==1

def test_empty_response_is_failure_not_complete_snapshot(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    with pytest.raises(api().SnapshotError,match='empty'): store.get(request(),lambda r:pd.DataFrame())
    assert not list((tmp_path/'raw').glob('**/manifest.json'))

def test_refresh_reports_revision_and_preserves_original(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    first=store.get(request(),lambda r:frame()); original=(first.path/'raw.csv').read_bytes()
    second=store.get(request(),lambda r:frame(11),refresh=True)
    assert first.snapshot_id!=second.snapshot_id
    assert (first.path/'raw.csv').read_bytes()==original
    assert second.manifest['revision']['changed'] is True
    assert second.manifest['revision']['changed_dates']==['2024-01-02']
    assert store.read(first.path).frame.close.tolist()==['10']

def test_failed_refresh_keeps_old_cache_usable(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    first=store.get(request(),lambda r:frame())
    def fail(r): raise ConnectionError('failed refresh')
    with pytest.raises(api().SnapshotError): store.get(request(),fail,refresh=True)
    assert store.get(request(),fail).snapshot_id==first.snapshot_id

@pytest.mark.parametrize('problem', ['infinite_close', 'duplicate_date', 'wrong_asset',
                                    'missing_field', 'bad_ohlc', 'negative_volume', 'bad_date'])
def test_invalid_refresh_is_quarantined_and_keeps_good_cache(tmp_path, problem):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    first=store.get(request(),lambda r:frame())
    bad=frame()
    if problem=='infinite_close': bad=frame(float('inf'))
    elif problem=='duplicate_date': bad=pd.concat([bad,bad],ignore_index=True)
    elif problem=='wrong_asset': bad['code']='sh.510500'
    elif problem=='missing_field': bad=bad.drop(columns='open')
    elif problem=='bad_ohlc': bad['high']=8
    elif problem=='negative_volume': bad['volume']=-1
    elif problem=='bad_date': bad['date']='2024-02-30'
    with pytest.raises(api().SnapshotError): store.get(request(),lambda r:bad,refresh=True)
    def forbidden(r): raise AssertionError('valid old cache must not download')
    assert store.get(request(),forbidden).snapshot_id==first.snapshot_id
    assert len(list((tmp_path/'raw').glob('**/manifest.json')))==1
    quarantine=list((tmp_path/'raw/quarantine').glob('*/*/rejected.json'))
    assert len(quarantine)==1
    meta=json.loads(quarantine[0].read_bytes())
    payload=(quarantine[0].parent/'raw.csv').read_bytes()
    assert meta['status']=='quarantined' and meta['request']['asset_code']=='510300'
    assert meta['raw_sha256']==api().digest(payload)
    assert payload==bad.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode('utf-8')
    failure=json.loads(next((tmp_path/'cache/failures').glob('*.json')).read_bytes())
    assert failure['quarantine_path']==quarantine[0].parent.relative_to(tmp_path/'raw').as_posix()

def test_invalid_first_response_has_no_complete_snapshot(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    with pytest.raises(api().SnapshotError): store.get(request(),lambda r:frame(float('inf')))
    assert not list((tmp_path/'raw').glob('**/manifest.json'))
    assert not list((tmp_path/'cache/requests').glob('*.json'))

@pytest.mark.parametrize('dates', [['2024-01-02','2024-01-02'],
                                  ['2024-01-04','2024-01-02'], ['2024-02-30']])
def test_invalid_calendar_cannot_be_cached(tmp_path,dates):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    req=request(asset_code='CALENDAR',exchange='CN',source='akshare_calendar',purpose='calendar')
    with pytest.raises(api().SnapshotError):
        store.get(req,lambda r:pd.DataFrame({'trade_date':dates}))
    assert not list((tmp_path/'cache/requests').glob('*.json'))

def test_bounded_provider_must_respect_requested_dates(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    bad=frame(); bad['date']='2023-12-29'
    with pytest.raises(api().SnapshotError,match='range'):
        store.get(request(source='akshare_em'),lambda r:bad)
    # Sina explicitly returns full history; preservation precedes downstream slicing.
    assert store.get(request(),lambda r:bad).frame.date.tolist()==['2023-12-29']

def test_structurally_invalid_legacy_cache_rejected_even_when_hashes_match(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    first=store.get(request(),lambda r:frame())
    payload=frame(float('inf')).to_csv(index=False,lineterminator='\n').encode('utf-8')
    (first.path/'raw.csv').write_bytes(payload)
    meta=first.manifest; meta['raw_sha256']=api().digest(payload)
    manifest_bytes=api().json_bytes(meta); checksum=api().digest(manifest_bytes)
    (first.path/'manifest.json').write_bytes(manifest_bytes)
    (first.path/'manifest.sha256').write_text(checksum,encoding='ascii')
    pointer=tmp_path/'cache/requests'/f'{request().key}.json'
    index=json.loads(pointer.read_bytes()); index['manifest_sha256']=checksum
    pointer.write_bytes(api().json_bytes(index))
    def forbidden(r): raise AssertionError('do not silently repair corrupt legacy cache')
    with pytest.raises(api().SnapshotError,match='nonfinite'): store.get(request(),forbidden)

def test_partial_write_not_published(tmp_path,monkeypatch):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    original=api().write_bytes
    def fail(path,payload):
        if path.name=='manifest.json': raise OSError('disk full')
        return original(path,payload)
    monkeypatch.setattr(api(),'write_bytes',fail)
    with pytest.raises(api().SnapshotError): store.get(request(),lambda r:frame())
    assert not list((tmp_path/'cache').glob('requests/*.json'))
    assert not [p for p in (tmp_path/'raw').glob('*/*/manifest.json') if '.pending-' not in str(p)]

def test_requests_separate_assets_sources_and_date_ranges(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    ids=[store.get(request(**kw),lambda r:frame()).snapshot_id for kw in
         ({},{'asset_code':'510500'},{'source':'akshare_em'},{'end_date':'2024-01-05'})]
    assert len(set(ids))==4

def test_tampered_manifest_and_cache_path_cannot_escape_store(tmp_path):
    store=api().SnapshotStore(tmp_path/'raw',tmp_path/'cache')
    first=store.get(request(),lambda r:frame())
    pointer=next((tmp_path/'cache/requests').glob('*.json'))
    p=json.loads(pointer.read_text()); p['relative_path']='../outside'; pointer.write_text(json.dumps(p))
    with pytest.raises(api().SnapshotError,match='path'): store.get(request(),lambda r:frame())

@pytest.mark.parametrize('kw',[{'asset_code':'../bad'},{'start_date':'2024-02-30'},
    {'end_date':'2023-01-01'},{'source':'auto'},{'exchange':'NYSE'}])
def test_invalid_requests_rejected(kw):
    api()
    with pytest.raises(ValueError): request(**kw)
