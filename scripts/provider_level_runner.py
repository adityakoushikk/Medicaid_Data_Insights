"""Run scripts/create_provider_level_from_month.py as a subprocess.

If provider_level_features is provided, a temporary config.yaml is written
from those values and passed via --config so all tuning lives in the Hydra
config rather than scripts/config.yaml.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def run_provider_level(
    input_csv: str,
    output_csv: str,
    provider_level_script: str,
    min_months: int = 6,
    no_filter: bool = False,
    provider_level_features: dict | None = None,
    python_exe: str | None = None,
) -> str:
    """Invoke create_provider_level_from_month.py as a subprocess.

    If provider_level_features is provided, writes a temp config.yaml
    containing those values and passes --config to the script so that
    rolling_flags, changepoints, etc. are fully controlled from Hydra.
    Date range filtering is applied upstream in create_provider_month_dataset.py.
    """
    python = python_exe or sys.executable
    script = str(Path(provider_level_script).resolve())

    cmd = [
        python, script,
        str(input_csv),
        "--output",     str(output_csv),
        "--min-months", str(min_months),
    ]
    if no_filter:
        cmd.append("--no-filter")

    # Write a temp config.yaml from Hydra params so the subprocess uses them
    # instead of the static scripts/config.yaml.
    tmp_config = None
    if provider_level_features is not None:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="provider_level_cfg_"
        )
        features_plain = OmegaConf.to_container(provider_level_features, resolve=True)
        yaml.dump(
            {"provider_level_features": features_plain},
            tmp,
        )
        tmp.flush()
        tmp_config = tmp.name
        cmd += ["--config", tmp_config]

    print(f"[provider_level_runner] {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)

    if tmp_config:
        Path(tmp_config).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"create_provider_level_from_month.py exited with code {result.returncode}"
        )
    return output_csv
