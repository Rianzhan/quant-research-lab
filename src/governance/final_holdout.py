"""Metadata-only Phase 9A protocol verification and immutable publication.

This module never imports a market loader, estimator, or backtest engine.
The allowlist and audit hook are applied before opening inputs. Constants bind
the reviewed protocol, not a caller-supplied replacement specification.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import uuid
from typing import Any, Iterator

SOURCE_FILE = 'config/phase9a_sources.json'
SOURCE_SHA256 = '2d552fefc197aff84c71dc95a1f15f423351039e4340a0218f711e032dbb3019'
# Set once after document review; all byte changes require a new registration.
DOCUMENT_PINS: dict[str, str] = {'config/final_holdout_spec.json': 'd74fb1b1e6c0233ded6c8c6cc554c7650bdf908aa94ec54169a5e2baa995e04c', 'config/final_holdout_metric_definitions.json': '876a660af3173940d5632e3888faaa4b977e399de91751ba16e8c005d7ca92ef', 'config/final_holdout_interpretation_rules.json': 'f309474d6a6895a5fd61c5191a69222a923ca0203607c92ddee39a4bcda245dd', 'config/allowed_inputs.json': '1cd8de39a225ace07c566b23d3f57cead42529343cc3e898e1f2946299890fb5', 'config/forbidden_inputs.json': 'e4ff80da64dabdcbc829c42f633a6cc8b18a2911138dabcb40df6e8ce105cd21', 'docs/phase9a-final-holdout-protocol.md': 'df3f60e4ede3647761970fb8e2a38a1ee7e1c62ce5510cf9eb468ed8c60dcb18'}  # PHASE9A_DOCUMENT_PINS
DOCUMENT_DIGEST = '338bd62afc0d4e6b33c1859975bbf63ca355fe26d91a41cf05c8fb300b531c55'  # PHASE9A_DOCUMENT_DIGEST
CODE_FILES = ('src/governance/__init__.py', 'src/governance/final_holdout.py')
_ACTIVE: ContextVar[tuple | None] = ContextVar('phase9a_access', default=None)


def canonical(value: Any) -> bytes:
    """Stable mapping order; lists retain their meaningful order (features)."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def digest(data: bytes) -> str:
    """SHA-256 of exact bytes."""
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    """Reject aliases and ambiguous spellings before any content access."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('input path alias rejected')
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
        raise ValueError('input path alias rejected')
    root = root.resolve()
    path = root.joinpath(*relative.parts)
    if path.resolve() != path or not path.is_relative_to(root):
        raise ValueError('input path alias rejected')
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError('input path alias rejected')
    return path


def _audit(event: str, args: tuple) -> None:
    """Process-wide hook, active only within this context; reject before open."""
    state = _ACTIVE.get()
    if state is None or event != 'open':
        return
    root, reads, writes = state
    filename, mode, flags = args
    if not isinstance(filename, (str, bytes, os.PathLike)):
        raise ValueError('Phase9A access denied: file descriptor input')
    absolute = Path(os.fsdecode(filename)).absolute()
    try:
        name = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError('Phase9A access denied: outside exact allowlist') from exc
    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if name not in (writes if writing else reads):
        raise ValueError('Phase9A access denied: ' + name)
    try:
        safe_path(root, name)
    except ValueError as exc:
        raise ValueError('Phase9A access denied: alias') from exc


sys.addaudithook(_audit)


@contextmanager
def access_guard(root: Path, reads: set[str], writes: set[str]) -> Iterator[None]:
    """Restrict Python file opens to exact current-stage paths, including writes."""
    token = _ACTIVE.set((root.resolve(), frozenset(reads), frozenset(writes)))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


class PinnedReader:
    """Read only independently pinned input identities; never follow manifest links."""

    def __init__(self, root: Path, pins: dict[str, str]) -> None:
        self.root, self.pins = root.resolve(), dict(pins)
        self.opened: dict[str, str] = {}

    def read(self, name: str) -> bytes:
        """Check path permission before open, then exact bytes before parsing."""
        if name not in self.pins:
            raise ValueError('Phase9A access denied: nonallowlisted input')
        path = safe_path(self.root, name)
        value = path.read_bytes()
        if digest(value) != self.pins[name]:
            raise ValueError('Frozen input hash mismatch: ' + name)
        self.opened[name] = self.pins[name]
        return value


def load_documents(root: Path) -> dict[str, Any]:
    """Load only the five new JSON contracts; no market metadata dereference."""
    reader = PinnedReader(root, DOCUMENT_PINS)
    with access_guard(root, set(DOCUMENT_PINS), set()):
        return {p: json.loads(reader.read(p)) for p in sorted(DOCUMENT_PINS)
                if p.endswith('.json')}


def document_digest(documents: dict[str, Any]) -> str:
    """Semantic digest independent of mapping insertion order."""
    return digest(canonical(documents))


def validate_documents(documents: dict[str, Any]) -> None:
    """Reject any change to the preregistered research contract."""
    if document_digest(documents) != DOCUMENT_DIGEST:
        raise ValueError('Frozen protocol differs; rule/metric/model edits prohibited')
    spec = documents['config/final_holdout_spec.json']
    if spec['source_lock_sha256'] != SOURCE_SHA256:
        raise ValueError('Source lock changed')


def build(root: Path) -> dict[str, Any]:
    """Validate frozen metadata and return a deterministic in-memory publication."""
    boot = PinnedReader(root, {SOURCE_FILE: SOURCE_SHA256})
    with access_guard(root, {SOURCE_FILE}, set()):
        source = json.loads(boot.read(SOURCE_FILE))
    pins = {**source['files_sha256'], **DOCUMENT_PINS, SOURCE_FILE: SOURCE_SHA256}
    reader = PinnedReader(root, pins)
    code_hashes = {}
    with access_guard(root, set(pins) | set(CODE_FILES), set()):
        # Pinned manifests are parsed only after hash verification. Payload names
        # inside the sealed manifest are never appended to the read allowlist.
        content = {p: reader.read(p) for p in sorted(pins)}
        documents = {p: json.loads(content[p]) for p in DOCUMENT_PINS if p.endswith('.json')}
        validate_documents(documents)
        for p in CODE_FILES:
            code_hashes[p] = digest(safe_path(root, p).read_bytes())
    spec = documents['config/final_holdout_spec.json']
    old = json.loads(content['config/strategy_spec.json'])
    if old['rules'] != spec['strategy_rules']:
        raise ValueError('Phase6B rules changed')
    for label in ['A', 'B']:
        family = spec['families'][label]
        if old['families'][label] != [family['scope'], family['model'], family['task']]:
            raise ValueError('Phase6B model family changed')
    allowed = documents['config/allowed_inputs.json']
    if allowed['metadata_and_development_evaluation_files_sha256'] != source['files_sha256']:
        raise ValueError('Allowlist differs from frozen source lock')
    gates = spec['gates']
    contract = {
        'version': 'phase9a-v1', 'accepted_v1_commit': source['accepted_v1_commit'],
        'protocol_digest': document_digest(documents),
        'document_sha256': dict(DOCUMENT_PINS), 'source_lock_sha256': SOURCE_SHA256,
        'inputs_sha256': reader.opened, 'code_sha256': code_hashes,
        'gates': gates, 'sealed_identity': source['sealed_identity'],
        'phase8_build_key': source['phase8_build_key'],
        'strategy_spec_hash': spec['strategy_spec_hash'], 'rules_hash': spec['rules_hash'],
    }
    return {**contract, 'build_key': digest(canonical(contract))}


def publish(root: Path, bundle: dict[str, Any], *, source_root: Path | None = None) -> Path:
    """Atomic new-directory publish; exact replay only, never overwrite outputs."""
    if bundle['build_key'] != digest(canonical({k: v for k, v in bundle.items() if k != 'build_key'})):
        raise ValueError('Invalid bundle hash')
    # Self-consistency is insufficient: independently re-read every frozen
    # metadata input, verify policy pins and rebuild the complete contract.
    # source_root permits synthetic output-directory tests, never different rules.
    if bundle != build(source_root if source_root is not None else root):
        raise ValueError('Publication differs from verified frozen contract')
    root = root.resolve()
    relative = 'results/reports/phase9a-v1-' + bundle['build_key'][:16]
    target = root / relative
    if target.resolve() != target:
        raise ValueError('Publication path alias')
    outputs = {
        'protocol_bindings.json': canonical(bundle),
        'gates.json': canonical(bundle['gates']),
        'validation.json': canonical({
            'all_pinned_inputs_verified': True, 'inputs_count': len(bundle['inputs_sha256']),
            'market_payloads_opened': False, 'holdout_files_opened': False,
            'model_retrained': False, 'final_holdout_evaluation_performed': False,
            'protocol_digest': bundle['protocol_digest'],
            'note': 'Unit-test results are separate recorded evidence, not asserted by this builder.',
        }),
    }
    outputs['manifest.json'] = canonical({
        'version': 'phase9a-v1', 'status': 'complete', 'build_key': bundle['build_key'],
        'files': {p: digest(v) for p, v in sorted(outputs.items())},
        'document_sha256': bundle['document_sha256'],
        'source_lock_sha256': SOURCE_SHA256,
    })
    if target.exists():
        if set(p.name for p in target.iterdir()) != set(outputs):
            raise ValueError('Existing publication membership changed')
        reader = PinnedReader(root, {relative + '/' + p: digest(v) for p, v in outputs.items()})
        with access_guard(root, set(reader.pins), set()):
            for p in outputs:
                reader.read(relative + '/' + p)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = root / '.tmp' / ('phase9a-stage-' + uuid.uuid4().hex)
    if stage.resolve() != stage:
        raise ValueError('Staging path alias')
    stage.mkdir(parents=True, exist_ok=False)
    writes = {(stage / p).relative_to(root).as_posix() for p in outputs}
    # Manifest is last. An interrupted stage stays in .tmp, never a publication.
    with access_guard(root, set(), writes):
        for name, value in outputs.items():
            safe_path(root, (stage / name).relative_to(root).as_posix()).write_bytes(value)
    stage.rename(target)
    return target


def main() -> None:
    """Verify/replay protocol only; no switch capable of opening the holdout."""
    root = Path(__file__).resolve().parents[2]
    result = build(root)
    output = publish(root, result)
    print(json.dumps({'publication': str(output), 'build_key': result['build_key'],
                      'gates': result['gates']}, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
