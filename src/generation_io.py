"""Run-directory and logging helpers for generation workflows."""

from __future__ import annotations

from dataclasses import is_dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd


def safe_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
    return safe.strip("_") or "generation"


def timestamped_run_dir(base_dir: str | Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(base_dir)
    stem = f"{stamp}_{safe_run_name(run_name)}"
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"_{attempt:03d}"
        run_dir = base_dir / f"{stem}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique run directory under {base_dir} for {stem}")


def setup_run_logger(
    name: str,
    log_path: str | Path,
    *,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logger.level)
        logger.addHandler(console_handler)

    return logger


def config_to_jsonable(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return _jsonable(asdict(config))
    if isinstance(config, dict):
        return _jsonable(config)
    raise TypeError(f"Unsupported config type: {type(config).__name__}")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(data), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_table(path: str | Path, table: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def copy_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return value
