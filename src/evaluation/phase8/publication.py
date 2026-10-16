"""Atomic, immutable evaluation publication; failures never become complete results."""
import json,os,uuid
from pathlib import Path,PurePosixPath
from typing import Callable
from .common import stable_bytes,digest,require


def _manifest(files: dict[str,bytes],bindings: dict) -> dict:
    require(bool(files),'Empty publication')
    for name,payload in files.items():
        p=PurePosixPath(name)
        require(isinstance(payload,bytes) and not p.is_absolute() and '..' not in p.parts
                and p.as_posix()==name and ':' not in name and '\\' not in name
                and not name.lower().startswith('data/') and 'holdout' not in name.lower()
                and name not in {'manifest.json','failure.json'},'Unsafe output name: '+name)
    core={'version':'phase8-v1','bindings':bindings,'files':{k:digest(v) for k,v in sorted(files.items())}}
    return {**core,'build_key':digest(stable_bytes(core)),'status':'complete'}


def verify_saved(path: Path,files: dict[str,bytes],bindings: dict) -> dict:
    """Compare a publication to regenerated evaluation bytes and the same bindings."""
    path=Path(path);expected=_manifest(files,bindings)
    require(not path.is_symlink(),'Aliased publication')
    members=list(path.rglob('*'));require(not any(p.is_symlink() for p in members),'Aliased member')
    require({p.relative_to(path).as_posix() for p in members if p.is_file()}==set(files)|{'manifest.json'},'Publication member set changed')
    require((path/'manifest.json').read_bytes()==stable_bytes(expected)+b'\n','Evaluation manifest/bindings changed')
    for name,sha in expected['files'].items():
        require(digest((path/name).read_bytes())==sha,'Evaluation output changed: '+name)
    return expected


def _write(path: Path,payload: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())


def write_complete(parent: Path,files: dict[str,bytes],bindings: dict,before_commit: Callable[[],None]) -> Path:
    """Recheck frozen sources after staging; rename only a fully verified publication."""
    meta=_manifest(files,bindings);parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    target=parent/('phase8-v1-'+meta['build_key'][:16])
    if target.exists():
        before_commit();verify_saved(target,files,bindings);return target
    staging=parent/('.phase8-pending-'+uuid.uuid4().hex);staging.mkdir()
    try:
        for name,payload in sorted(files.items()):_write(staging/name,payload)
        before_commit()
        _write(staging/'manifest.json',stable_bytes(meta)+b'\n')
        verify_saved(staging,files,bindings)
        # Windows rename fails if a target already exists, preserving immutability.
        require(not target.exists(),'Concurrent publication already exists')
        staging.rename(target)
    except Exception as error:
        marker=staging/'manifest.json'
        if marker.exists():marker.unlink()
        if staging.exists():_write(staging/'failure.json',stable_bytes({'status':'failed','error_type':type(error).__name__})+b'\n')
        raise
    return target
