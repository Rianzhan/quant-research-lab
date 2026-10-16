"""Fail a public release on forbidden artifacts or common sensitive strings.

Only reports relative filenames and finding categories, never secret values.
No network calls. This is a focused heuristic guard, not a comprehensive DLP tool.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','venv','__pycache__','.pytest_cache','.tmp'}
BLOCKED_DIRS={'data','results','artifacts','models','outputs','private'}
BLOCKED_EXT={'.csv','.parquet','.pkl','.pickle','.joblib','.pdf','.zip','.log','.xml','.ipynb','.pem','.key','.p12','.pfx'}
PATTERNS={
    'personal_home_path':re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s"\']+'),
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'api_key':re.compile(r'\bsk-[A-Za-z0-9_-]{32,}\b'),
    'credential_url':re.compile(r'https?://[^/\s:@]+:[^/\s@]+@'),
    'personal_email':re.compile(r'\b[A-Za-z0-9_.+-]+@(?!users\.noreply\.github\.com\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}


def inspect(root: Path) -> tuple[list[dict],dict[str,str]]:
    """Inspect the working public set without following symlinks."""
    findings=[];hashes={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if any(part in SKIP for part in rel.parts):continue
        if p.is_symlink():
            findings.append(dict(path=rel.as_posix(),kind='symlink'));continue
        if not p.is_file():continue
        if (rel.parts[0] in BLOCKED_DIRS or p.suffix.lower() in BLOCKED_EXT
            or p.name.startswith('.env') or rel.parts[:2]==('docs','evidence')):
            findings.append(dict(path=rel.as_posix(),kind='excluded_artifact'));continue
        if p.stat().st_size>2_000_000:
            findings.append(dict(path=rel.as_posix(),kind='unexpected_large_file'));continue
        b=p.read_bytes();hashes[rel.as_posix()]=hashlib.sha256(b).hexdigest()
        try:text=b.decode('utf-8-sig')
        except UnicodeDecodeError:
            findings.append(dict(path=rel.as_posix(),kind='unexpected_binary'));continue
        for category,pattern in PATTERNS.items():
            if pattern.search(text):findings.append(dict(path=rel.as_posix(),kind=category))
    return findings,hashes


def main() -> int:
    findings,hashes=inspect(ROOT)
    manifest=ROOT/'PUBLICATION_MANIFEST.json'
    if not manifest.exists():
        findings.append(dict(path='PUBLICATION_MANIFEST.json',kind='manifest_missing'))
    else:
        expected=json.loads(manifest.read_text(encoding='utf-8'))['files_sha256']
        actual={p:h for p,h in hashes.items() if p!='PUBLICATION_MANIFEST.json'}
        for path in sorted(set(expected)|set(actual)):
            if expected.get(path)!=actual.get(path):
                findings.append(dict(path=path,kind='manifest_mismatch'))
    print(json.dumps(dict(files_scanned=len(hashes),findings=findings,
        passed=not findings,scope='public_working_files_only; original_history_not_imported'),indent=2))
    return int(bool(findings))


if __name__=='__main__':raise SystemExit(main())
