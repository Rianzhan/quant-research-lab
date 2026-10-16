"""Offline build/verify entry point for shared Phase4.5 model inputs."""
import argparse
import json
from pathlib import Path
import yaml
from src.data.pipeline import PipelineError
from src.features.scale_free import read_registration
from src.features.scale_free_pipeline import publish, verify
from src.utils.config import load_config


def main(argv: list[str] | None=None, *, root: Path | None=None) -> int:
    """Build a new immutable version or validate one; no model/holdout option."""
    root=Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','verify'))
    parser.add_argument('--config',type=Path,default=root/'config/config.yaml')
    parser.add_argument('--registration',type=Path,default=root/'config/scale_free.yaml')
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args(argv)
    try:
        config=load_config(args.config);registration=read_registration(args.registration)
        path=publish(config,registration,root) if args.command=='build' else args.manifest
        if path is None: parser.error('verify requires --manifest')
        print(json.dumps({'manifest':str(path),**verify(path,config,registration,root)},ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,PipelineError,yaml.YAMLError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)},ensure_ascii=False))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
