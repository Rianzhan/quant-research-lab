"""Build or verify registered development folds; no model fitting or holdout mode."""
import argparse
import json
from pathlib import Path
import yaml
from src.data.pipeline import PipelineError
from src.utils.config import load_config
from src.validation.config import load_validation
from src.validation.pipeline import publish, verify


def main(argv: list[str] | None=None) -> int:
    """Return a clear failure status rather than publishing partial fold results."""
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','verify'))
    parser.add_argument('--config',type=Path,default=root/'config/config.yaml')
    parser.add_argument('--validation',type=Path,default=root/'config/validation.yaml')
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args(argv)
    try:
        config=load_config(args.config);policy=load_validation(args.validation,config)
        path=publish(config,policy,root) if args.command=='build' else args.manifest
        if path is None: parser.error('verify requires --manifest')
        print(json.dumps({'manifest':str(path),**verify(path,config,policy,root)},ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,PipelineError,yaml.YAMLError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)},ensure_ascii=False))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
