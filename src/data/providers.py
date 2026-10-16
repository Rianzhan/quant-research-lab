"""Isolate SDK calls in bounded subprocesses; never silently switch providers."""
from __future__ import annotations
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.data.contracts import DataRequest


class ProviderError(RuntimeError):
    """Explicit remote/API failure, not proof of a missing asset or suspension."""


def fetch_prices(request: DataRequest, timeout_seconds: int = 45) -> pd.DataFrame:
    """Bound the entire SDK call, including login, JS decoding and socket retries."""
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError('timeout_seconds must be a positive integer')
    try:
        result = subprocess.run([sys.executable, '-m', 'src.data._worker'], input=json.dumps(asdict(request)),
                                cwd=Path(__file__).resolve().parents[2], capture_output=True,
                                text=True, encoding='utf-8', timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f'{request.source} whole-call timeout after {timeout_seconds}s') from exc
    if result.returncode:
        raise ProviderError(f'{request.source}: {result.stderr.strip()}')
    try:
        payload = json.loads(result.stdout)
        frame = pd.read_csv(io.StringIO(payload['csv']), dtype=str, keep_default_na=False)
        frame.attrs['provider_notes'] = payload['notes'] + ([result.stderr.strip()] if result.stderr.strip() else [])
        return frame
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        raise ProviderError(f'{request.source} malformed/empty worker response: {exc}') from exc
