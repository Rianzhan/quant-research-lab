"""Offline Phase5 registered model build/verify command, development data only."""
import argparse
import json
from pathlib import Path
import yaml
from src.data.pipeline import PipelineError
from src.models.config import read_experiment
from src.models.pipeline import publish, verify
from src.utils.config import load_config
from src.utils.logger import configure_logging


def main(argv: list[str] | None=None, *, root: Path | None=None) -> int:
    """Return a failed status for fit warnings, corrupt files or unapproved inputs."""
    root=Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','verify'))
    parser.add_argument('--config',type=Path,default=root/'config/config.yaml')
    parser.add_argument('--experiment',type=Path,default=root/'config/model_experiment.yaml')
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args(argv);configure_logging()
    try:
        config=load_config(args.config);experiment=read_experiment(args.experiment)
        path=publish(config,experiment,root) if args.command=='build' else args.manifest
        if path is None: parser.error('verify requires --manifest')
        result=verify(path,config,experiment,root)
        result.pop('files_sha256')
        print(json.dumps({'manifest':str(path),**result},ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,Warning,PipelineError,yaml.YAMLError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)},ensure_ascii=False))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
