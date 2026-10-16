"""Private SDK subprocess entry point. No models, features or automatic retries."""
from __future__ import annotations
import contextlib
import io
import json
import sys

from src.data.contracts import DataRequest


def main() -> int:
    """Keep vendor stdout separate from the machine-readable data response."""
    messages = io.StringIO()
    try:
        request = DataRequest(**json.load(sys.stdin))
        interface = request.interface()
        with contextlib.redirect_stdout(messages):
            if request.source.startswith('akshare'):
                import akshare as ak
                function = getattr(ak, interface['function'].split('.')[-1])
                frame = function(**interface['params'])
            else:
                import baostock as bs
                import pandas as pd
                login = bs.login()
                if login.error_code != '0':
                    raise RuntimeError(f'BaoStock login {login.error_code}: {login.error_msg}')
                try:
                    result = bs.query_history_k_data_plus(**interface['params'])
                    if result.error_code != '0':
                        raise RuntimeError(f'BaoStock query {result.error_code}: {result.error_msg}')
                    rows = []
                    while result.next():
                        rows.append(result.get_row_data())
                    if result.error_code != '0':
                        raise RuntimeError(f'BaoStock pagination {result.error_code}: {result.error_msg}')
                    frame = pd.DataFrame(rows, columns=result.fields)
                finally:
                    logout = bs.logout()
                    if logout.error_code != '0':
                        print(f'WARNING BaoStock logout {logout.error_code}: {logout.error_msg}')
        if frame.empty:
            raise RuntimeError('empty provider response; coverage unknown')
        print(json.dumps({'csv': frame.to_csv(index=False, lineterminator='\n', float_format='%.17g'),
                          'notes': messages.getvalue().splitlines()}, ensure_ascii=True))
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}\n{messages.getvalue()}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
