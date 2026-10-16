"""Data-only CLI. Explicit commands never train models or compute features."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from src.data.pipeline import read_audit, replay_dataset, run_download, run_probe
from src.data.providers import fetch_prices
from src.utils.config import load_config
from src.utils.logger import configure_logging


def main(argv: list[str] | None = None) -> int:
    """Return nonzero for failed probes/downloads, including partially available pools."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Phase 2 historical ETF data only')
    parser.add_argument('command',choices=['probe','download','replay'])
    parser.add_argument('--config',type=Path,default=root/'config/config.yaml')
    parser.add_argument('--probe-report',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--refresh',action='store_true',help='Explicit new immutable snapshots; report revisions')
    args=parser.parse_args(argv); logger=configure_logging()
    try:
        if sys.version_info[:3] != (3,12,14):
            raise ValueError('Use the project Python 3.12.14 interpreter')
        if args.command=='replay':
            if not args.manifest or args.refresh:
                raise ValueError('replay requires --manifest and does not permit --refresh')
            panel=replay_dataset(args.manifest,root)
            print(json.dumps({'status':'verified_offline','rows':len(panel),'manifest':str(args.manifest)}))
            return 0
        config=load_config(args.config)
        fetch=lambda request:fetch_prices(request,config['data']['download']['timeout_seconds'])
        if args.command=='probe':
            result=run_probe(config,root,fetch,refresh=args.refresh)
        else:
            if not args.probe_report:
                raise ValueError('download requires --probe-report from a successful matching probe')
            result=run_download(config,root,args.probe_report,fetch,refresh=args.refresh)
        report=read_audit(result)
        print(json.dumps({'status':report['status'],'report':str(result)},ensure_ascii=True))
        return 0 if report['status']=='complete' else 1
    except Exception as exc:
        logger.error('Data operation failed: %s: %s',type(exc).__name__,exc)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
