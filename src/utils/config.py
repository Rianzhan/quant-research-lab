"""Load explicit research settings without downloading data or fitting models."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """An unreadable or invalid research configuration."""


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silent overwrites by duplicate keys."""


def _unique_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ConfigError(f"YAML key must be a unique string: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)

# Explicit keys catch spelling mistakes instead of silently using a default.
_SCHEMA: dict[str, Any] = {
    "schema_version": None,
    "experiment": dict.fromkeys(["version", "seed", "python_version", "mode", "timezone"]),
    "timing": dict.fromkeys(["signal_time", "execution_time", "prediction_horizon_days", "rebalance_frequency_days", "holding_policy"]),
    "universe": dict.fromkeys(["registration_status", "registered_at", "change_reason", "assets"]),
    "data": {
        **dict.fromkeys(["provider_priority", "start_date", "end_date", "interval", "raw_adjustment", "research_price_policy", "missing_price_policy"]),
        "download": dict.fromkeys(["source", "timeout_seconds", "request_pause_seconds", "probe_start_date", "probe_end_date", "coverage_start_date", "as_of_date", "price_conflict_tolerance", "volume_conflict_relative_tolerance", "amount_price_tolerance", "suspicious_price_change_fraction"]),
        "paths": dict.fromkeys(["raw", "processed", "cache", "results"]),
    },
    "features": dict.fromkeys(["return_windows", "momentum_windows", "moving_average_windows", "moving_average_diff_pairs", "volatility_windows", "volatility_ddof", "volume_change_periods", "volume_average_window", "rsi_window", "macd_fast", "macd_slow", "macd_signal"]),
    "labels": dict.fromkeys(["tasks", "classification_return_threshold", "missing_target_policy"]),
    "validation": dict.fromkeys(["method", "initial_train_years", "test_years", "purge_days", "shuffle"]),
    "models": {
        "scope": None, "n_jobs": None,
        "logistic": dict.fromkeys(["C", "max_iter", "solver"]),
        "random_forest": dict.fromkeys(["n_estimators", "max_depth", "min_samples_leaf"]),
        "xgboost": dict.fromkeys(["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "tree_method", "device"]),
    },
    "signals": dict.fromkeys(["threshold_buy", "regression_method", "regression_min_return"]),
    "portfolio": dict.fromkeys(["allocation_method", "max_asset_weight", "max_total_weight", "initial_capital", "cash_annual_rate", "allow_short", "allow_leverage"]),
    "costs": dict.fromkeys(["fee_rates", "slippage_rate", "charging_basis"]),
    "reporting": dict.fromkeys(["annualization_days", "risk_free_rate", "log_level"]),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _keys(value: Any, schema: dict[str, Any], path: str = "config") -> None:
    _require(isinstance(value, dict), f"{path} must be a mapping")
    _require(all(isinstance(k, str) for k in value), f"{path} keys must be strings")
    missing, unknown = set(schema) - set(value), set(value) - set(schema)
    _require(not missing and not unknown, f"{path}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for key, child in schema.items():
        if child is not None:
            _keys(value[key], child, f"{path}.{key}")


def _text(value: Any, name: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")


def _number(value: Any, name: str, minimum: float | None = None, maximum: float | None = None) -> None:
    _require(type(value) in (int, float), f"{name} must be a number, not boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ConfigError(f"{name} is too large for numerical computation") from exc
    _require(finite, f"{name} must be finite")
    _require(minimum is None or value >= minimum, f"{name} must be >= {minimum}")
    _require(maximum is None or value <= maximum, f"{name} must be <= {maximum}")


def _choice(value: Any, options: tuple[Any, ...], name: str) -> None:
    _require(any(type(value) is type(option) and value == option for option in options), f"{name} must be one of {options}")


def universe_digest(universe: dict[str, Any]) -> str:
    """Bind registered identities, rationale, limitations and registration time."""
    record = {k: v for k, v in universe.items() if k != 'freeze_sha256'}
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def _validate_universe(universe: dict[str, Any], schema_version: int = 1) -> None:
    _choice(universe["registration_status"], ("draft", "frozen"), "registration_status")
    _text(universe["change_reason"], "universe.change_reason")
    registered = universe["registered_at"]
    if universe["registration_status"] == "draft":
        _require(registered is None, "draft registration must have registered_at: null")
    else:
        _text(registered, "universe.registered_at")
        try:
            stamp = datetime.fromisoformat(registered)
        except ValueError as exc:
            raise ConfigError("registered_at must be an ISO timestamp") from exc
        _require(stamp.utcoffset() is not None, "registered_at must contain a timezone")
    assets = universe["assets"]
    _require(isinstance(assets, list) and 6 <= len(assets) <= 10, "universe must contain 6–10 assets")
    categories = {"china_broad", "growth_technology", "bond", "gold", "cross_border_equity"}
    codes: set[str] = set()
    seen_categories: set[str] = set()
    for asset in assets:
        fields = ["code", "exchange", "name", "category", "selection_reason"]
        if schema_version == 2:
            fields += ['data_available_from', 'known_limitations']
        _keys(asset, dict.fromkeys(fields), "asset")
        code = asset["code"]
        _require(isinstance(code, str) and re.fullmatch(r"[0-9]{6}", code) is not None, "asset.code must be a quoted six-digit string")
        _require(code not in codes, f"duplicate asset code: {code}")
        codes.add(code)
        _choice(asset["exchange"], ("SSE", "SZSE"), "asset.exchange")
        _choice(asset["category"], tuple(sorted(categories)), "asset.category")
        seen_categories.add(asset["category"])
        _text(asset["name"], "asset.name")
        _text(asset["selection_reason"], "asset.selection_reason")
        if schema_version == 2:
            try:
                date.fromisoformat(asset['data_available_from'])
            except (ValueError, TypeError) as exc:
                raise ConfigError('Invalid available date') from exc
            limits = asset['known_limitations']
            _require(isinstance(limits, list) and bool(limits), 'Known limitations must be recorded')
            for limitation in limits:
                _text(limitation, 'known_limitations')
    _require(seen_categories == categories, "asset pool must cover all five asset categories")
    if schema_version == 2:
        _require(universe['registration_status'] == 'frozen', 'Phase 2.5 requires a frozen universe')
        _require(universe['freeze_sha256'] == universe_digest(universe), 'Universe freeze digest mismatch; register a new experiment version')


def _validate_data(data: dict[str, Any]) -> None:
    providers = data["provider_priority"]
    _require(isinstance(providers, list) and bool(providers), "provider_priority must be a nonempty list")
    for provider in providers:
        _choice(provider, ("akshare", "baostock"), "provider_priority")
    _require(len(set(providers)) == len(providers), "duplicate data provider")
    dates = []
    for name in ("start_date", "end_date"):
        _text(data[name], name)
        _require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", data[name]) is not None, f"{name} must be YYYY-MM-DD")
        try:
            dates.append(date.fromisoformat(data[name]))
        except ValueError as exc:
            raise ConfigError(f"invalid {name}") from exc
    _require(dates[0] <= dates[1], "start_date must not exceed end_date")
    _choice(data["interval"], ("daily",), "data.interval")
    _choice(data["raw_adjustment"], ("none",), "raw_adjustment")
    _choice(data["research_price_policy"], ("pending_phase2_review", "unadjusted_event_segmented_v1"), "research_price_policy")
    _choice(data["missing_price_policy"], ("reject_no_backfill",), "missing_price_policy")
    for name, value in data["paths"].items():
        _text(value, f"data.paths.{name}")
    download = data["download"]
    _choice(download["source"], ("akshare_sina", "akshare_em", "baostock"), "download.source")
    _integer(download["timeout_seconds"], "download.timeout_seconds")
    for name in ("request_pause_seconds", "price_conflict_tolerance", "volume_conflict_relative_tolerance", "amount_price_tolerance", "suspicious_price_change_fraction"):
        _number(download[name], f"download.{name}", 0)
    for name in ("probe_start_date", "probe_end_date", "coverage_start_date", "as_of_date"):
        _text(download[name], name)
        try:
            _require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", download[name]) is not None, f"invalid {name}")
            date.fromisoformat(download[name])
        except ValueError as exc:
            raise ConfigError(f"invalid {name}") from exc
    _require(download["probe_start_date"] <= download["probe_end_date"] <= download["as_of_date"], "probe date range must end by as_of_date")
    _require(download["coverage_start_date"] <= data["start_date"] <= data["end_date"] <= download["as_of_date"], "research range must fit coverage range")


def _validate_features(features: dict[str, Any]) -> None:
    for key in ("return_windows", "momentum_windows", "moving_average_windows", "volatility_windows"):
        windows = features[key]
        _require(isinstance(windows, list) and bool(windows), f"{key} must be a nonempty list")
        for value in windows:
            _integer(value, key)
        _require(len(set(windows)) == len(windows), f"{key} has duplicates")
    pairs = features["moving_average_diff_pairs"]
    _require(isinstance(pairs, list) and bool(pairs), "moving_average_diff_pairs must be a nonempty list")
    for pair in pairs:
        _require(isinstance(pair, list) and len(pair) == 2, "moving_average_diff_pairs must contain pairs")
        for value in pair:
            _integer(value, "moving average difference window")
        _require(pair[0] < pair[1] and all(v in features["moving_average_windows"] for v in pair), "difference windows must exist and be increasing")
    for key in ("volume_change_periods", "volume_average_window", "rsi_window", "macd_fast", "macd_slow", "macd_signal"):
        _integer(features[key], key)
    _integer(features["volatility_ddof"], "volatility_ddof", 0)
    _require(features["volatility_ddof"] < min(features["volatility_windows"]), "volatility_ddof must be less than every volatility window")
    _require(features["macd_fast"] < features["macd_slow"], "macd_fast must be less than macd_slow")


def _validate_models(models: dict[str, Any]) -> None:
    _choice(models["scope"], ("per_asset", "pooled"), "models.scope")
    _integer(models["n_jobs"], "models.n_jobs")
    logistic = models["logistic"]
    _number(logistic["C"], "logistic.C", 0)
    _require(logistic["C"] > 0, "logistic.C must be positive")
    _integer(logistic["max_iter"], "logistic.max_iter")
    _choice(logistic["solver"], ("lbfgs",), "logistic.solver")
    for key, value in models["random_forest"].items():
        _integer(value, f"random_forest.{key}")
    xgb = models["xgboost"]
    for key in ("n_estimators", "max_depth"):
        _integer(xgb[key], f"xgboost.{key}")
    for key in ("learning_rate", "subsample", "colsample_bytree"):
        _number(xgb[key], f"xgboost.{key}", 0, 1)
        _require(xgb[key] > 0, f"xgboost.{key} must be positive")
    _choice(xgb["tree_method"], ("hist",), "xgboost.tree_method")
    _choice(xgb["device"], ("cpu",), "xgboost.device")


def _validate_portfolio(config: dict[str, Any]) -> None:
    signals, portfolio, costs = config["signals"], config["portfolio"], config["costs"]
    _number(signals["threshold_buy"], "threshold_buy", 0, 1)
    _choice(signals["regression_method"], ("positive_prediction_weight",), "regression_method")
    _number(signals["regression_min_return"], "regression_min_return", 0)
    _choice(portfolio["allocation_method"], ("equal_weight", "signal_weighted"), "allocation_method")
    for key in ("max_asset_weight", "max_total_weight"):
        _number(portfolio[key], key, 0, 1)
        _require(portfolio[key] > 0, f"{key} must be positive")
    _require(portfolio["max_asset_weight"] <= portfolio["max_total_weight"], "max_asset_weight exceeds max_total_weight")
    _number(portfolio["initial_capital"], "initial_capital", 0)
    _require(portfolio["initial_capital"] > 0, "initial_capital must be positive")
    _number(portfolio["cash_annual_rate"], "cash_annual_rate", 0, 1)
    for key in ("allow_short", "allow_leverage"):
        _choice(portfolio[key], (False,), key)
    _choice(costs["charging_basis"], ("per_side_traded_notional",), "charging_basis")
    rates = costs["fee_rates"]
    _require(isinstance(rates, list) and bool(rates), "fee_rates must be a nonempty list")
    for value in [*rates, costs["slippage_rate"]]:
        _number(value, "fee/slippage rate", 0, 1)
        _require(value < 1, "fee/slippage rate must be less than 1")
    _require(len(set(rates)) == len(rates), "duplicate fee_rates")


def _validate_governance(config: dict[str, Any]) -> None:
    g = config['governance']
    _require((config['data']['start_date'], config['data']['end_date']) == ('2018-01-01', '2025-12-31'),
             'Frozen development range must remain 2018-01-01 through 2025-12-31')
    _require((g['holdout_start_date'], g['holdout_end_date']) == ('2026-01-01', '2026-09-30'),
             'Frozen holdout candidate range changed')
    _choice(g['holdout_status'], ('sealed_candidate',), 'holdout_status')
    _choice(g['method_freeze_status'], ('unfrozen',), 'method_freeze_status; release is a separately approved future phase')
    _choice(g['research_scope'], ('development_price_features',), 'research_scope')
    _choice(config['data']['research_price_policy'], ('unadjusted_event_segmented_v1',), 'research_price_policy')
    _require(isinstance(g['policy_sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', g['policy_sha256']) is not None,
             'Invalid policy_sha256')
    for key in ('policy_file', 'dataset_manifest', 'phase2_manifest'):
        _text(g[key], key)
        path = Path(g[key])
        _require(not path.is_absolute() and not path.drive and '..' not in path.parts, f'{key} must stay inside project')


def validate_config(config: dict[str, Any]) -> None:
    """Reject invalid values and inconsistent research assumptions; do no I/O."""
    _require(isinstance(config, dict), 'Configuration must be a mapping')
    schema_version = config.get('schema_version')
    _choice(schema_version, (1, 2), "schema_version")
    schema = _SCHEMA
    if schema_version == 2:
        schema = {**_SCHEMA, 'universe': {**_SCHEMA['universe'], 'freeze_sha256': None},
                  'governance': dict.fromkeys(['policy_file', 'policy_sha256', 'dataset_manifest',
                    'phase2_manifest', 'holdout_start_date', 'holdout_end_date', 'holdout_status',
                    'method_freeze_status', 'research_scope'])}
    _keys(config, schema)
    experiment, timing, validation = config["experiment"], config["timing"], config["validation"]
    _text(experiment["version"], "experiment.version")
    _integer(experiment["seed"], "experiment.seed", 0)
    _require(experiment["seed"] < 2**32, "seed must be below 2**32")
    _choice(experiment["python_version"], ("3.12.14",), "python_version")
    _choice(experiment["mode"], ("historical_research",), "experiment.mode")
    _choice(experiment["timezone"], ("Asia/Shanghai",), "timezone")
    _choice(timing["signal_time"], ("t_close_after",), "signal_time")
    _choice(timing["execution_time"], ("t_plus_1_open",), "execution_time")
    _choice(timing["holding_policy"], ("until_next_rebalance",), "holding_policy")
    for key in ("prediction_horizon_days", "rebalance_frequency_days"):
        _integer(timing[key], key)
    _choice(validation["method"], ("expanding_walk_forward",), "validation.method")
    _choice(validation["shuffle"], (False,), "validation.shuffle")
    for key in ("initial_train_years", "test_years", "purge_days"):
        _integer(validation[key], key)
    _require(validation["purge_days"] >= timing["prediction_horizon_days"], "purge_days must cover prediction_horizon_days")
    _validate_universe(config["universe"], schema_version)
    _validate_data(config["data"])
    if schema_version == 2:
        _validate_governance(config)
    _validate_features(config["features"])
    _validate_models(config["models"])
    _validate_portfolio(config)
    labels, reporting = config["labels"], config["reporting"]
    _require(labels["tasks"] == ["classification", "regression"], "labels.tasks must include classification and regression in that order")
    _number(labels["classification_return_threshold"], "classification_return_threshold")
    _choice(labels["missing_target_policy"], ("drop_from_supervised_samples",), "missing_target_policy")
    _integer(reporting["annualization_days"], "annualization_days")
    _number(reporting["risk_free_rate"], "risk_free_rate", 0, 1)
    _choice(reporting["log_level"], ("DEBUG", "INFO", "WARNING", "ERROR"), "log_level")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read UTF-8 safe YAML and return a freshly validated settings dictionary."""
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            config = yaml.load(stream, Loader=_UniqueSafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration: {exc}") from exc
    validate_config(config)
    return config


def resolve_paths(config: dict[str, Any], project_root: str | Path) -> dict[str, Path]:
    """Anchor configured relative directories to the project, independent of CWD."""
    try:
        root = Path(project_root).resolve()
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot resolve project root path: {exc}") from exc
    result = {}
    for name, value in config["data"]["paths"].items():
        relative = Path(value)
        _require(not relative.is_absolute() and not relative.drive and ".." not in relative.parts, f"data.paths.{name} must stay inside project")
        try:
            target = (root / relative).resolve()
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Cannot resolve data.paths.{name}: {exc}") from exc
        _require(target.is_relative_to(root) and target != root, f"data.paths.{name} must be a project subdirectory")
        result[name] = target
    return result


def config_digest(config: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 of the validated semantic settings."""
    validate_config(config)
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
