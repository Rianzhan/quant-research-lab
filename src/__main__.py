"""Inspect configuration only; the separate src.data CLI handles historical data."""

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import ConfigError, config_digest, load_config, resolve_paths
from src.utils.logger import configure_logging


def main(argv: list[str] | None = None) -> int:
    """Validate configuration and print a stable machine-readable summary."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate research configuration only")
    parser.add_argument("--check-config", action="store_true", required=True)
    parser.add_argument("--config", type=Path, default=root / "config" / "config.yaml")
    args = parser.parse_args(argv)
    logger = configure_logging()
    try:
        if sys.version_info[:3] != (3, 12, 14):
            raise ConfigError("Python 3.12.14 is required; use the project .venv interpreter")
        config = load_config(args.config)
        paths = resolve_paths(config, root)
        configure_logging(config["reporting"]["log_level"])
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2
    summary = {
        "status": "configuration_valid",
        "phase": "4",
        "research_ready": None,
        "research_gate_check": "python -m src.data.research_dataset check",
        "experiment_version": config["experiment"]["version"],
        "config_sha256": config_digest(config),
        "asset_count": len(config["universe"]["assets"]),
        "registration_status": config["universe"]["registration_status"],
        "timing": config["timing"],
        "fee_rates": config["costs"]["fee_rates"],
        "slippage_rate": config["costs"]["slippage_rate"],
        "paths": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    logger.info("Configuration check only. Review look-ahead bias, survivorship bias and transaction costs before reporting results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
