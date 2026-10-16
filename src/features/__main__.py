"""Build or verify Phase 3 artifacts offline; nothing is fitted or backtested."""
import argparse
import json
from pathlib import Path
from src.data.pipeline import PipelineError
from src.features.pipeline import publish,verify
from src.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','verify'))
    parser.add_argument('--config',type=Path,default=root/'config/config.yaml')
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args(argv)
    try:
        config=load_config(args.config)
        path=publish(config,root) if args.command=='build' else args.manifest
        if path is None:
            parser.error('verify requires --manifest')
        print(json.dumps({'manifest':str(path),**verify(path,config,root)},ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,PipelineError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)},ensure_ascii=False))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
