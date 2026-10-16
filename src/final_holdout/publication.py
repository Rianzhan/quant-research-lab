"""First-attempt receipts and immutable hash-verified final model publication."""
from hashlib import sha256
import json
from pathlib import Path
import pickle
from src.backtest.contracts import require,stable_bytes
from src.governance.final_holdout import safe_path


def begin(directory: Path, binding: dict) -> Path:
    """Record an attempt before fitting. A failed attempt is never silently retried."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/'first_attempt.json'
    require(not receipt.exists(),'A final fit attempt already exists; never refit automatically')
    require(not any(directory.iterdir()),'Final attempt directory is not empty')
    with receipt.open('xb') as stream:
        stream.write(stable_bytes({'status':'started','binding':binding,'final_holdout_evaluation_performed':False}))
    return receipt


def finish(directory: Path, files: dict[str,bytes], binding: dict, *, gates: dict) -> Path:
    """Publish the completion manifest last; preserve partial files on failure."""
    directory=Path(directory);manifest=directory/'final_model_manifest.json'
    require((directory/'first_attempt.json').exists() and not manifest.exists(),'Invalid/completed final attempt')
    require(json.loads((directory/'first_attempt.json').read_bytes())['binding']==binding,'Attempt binding changed')
    reserved={'first_attempt.json','final_model_manifest.json','final_model_hashes.json','gates.json'}
    require(not set(files)&reserved,'Reserved final artifact name')
    hashes={n:sha256(b).hexdigest() for n,b in sorted(files.items())}
    outputs={**files,'final_model_hashes.json':stable_bytes(hashes),'gates.json':stable_bytes(gates)}
    for name,payload in sorted(outputs.items()):
        path=safe_path(directory.resolve(),name);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(payload)
        require(sha256(path.read_bytes()).hexdigest()==sha256(payload).hexdigest(),'Write verification failed')
    body={'version':'phase9b-v1','status':'complete','binding':binding,
        'files':{n:sha256(b).hexdigest() for n,b in sorted(outputs.items())},'gates':gates}
    body['build_key']=sha256(stable_bytes(body)).hexdigest()
    payload=stable_bytes(body)
    with manifest.open('xb') as stream:stream.write(payload)
    for name in outputs:
        (directory/name).chmod(0o444)
    manifest.chmod(0o444)
    verify(manifest,expected_sha256=sha256(payload).hexdigest())
    return manifest


def verify(manifest: Path, *, expected_sha256: str | None=None) -> dict:
    """Verify manifest and all artifacts. Production loading requires external SHA."""
    manifest=Path(manifest);payload=manifest.read_bytes()
    if expected_sha256 is not None:require(sha256(payload).hexdigest()==expected_sha256,'Final manifest hash changed')
    meta=json.loads(payload);body={k:v for k,v in meta.items() if k!='build_key'}
    require(meta['status']=='complete' and meta['build_key']==sha256(stable_bytes(body)).hexdigest(),'Final manifest build hash changed')
    for name,expected in meta['files'].items():
        require(sha256(safe_path(manifest.parent.resolve(),name).read_bytes()).hexdigest()==expected,'Final artifact hash changed: '+name)
    return meta


def load_model(manifest: Path, model_id: str, *, expected_manifest_sha256: str):
    """Load only a trusted local pickle after external manifest and binary checks."""
    require(len(expected_manifest_sha256)==64,'External frozen manifest identity required')
    meta=verify(manifest,expected_sha256=expected_manifest_sha256)
    name='models/'+model_id+'.pkl'
    require(name in meta['files'],'Frozen model unavailable; no substitution')
    payload=safe_path(Path(manifest).parent.resolve(),name).read_bytes()
    require(sha256(payload).hexdigest()==meta['files'][name],'Model changed before deserialize')
    return pickle.loads(payload)
