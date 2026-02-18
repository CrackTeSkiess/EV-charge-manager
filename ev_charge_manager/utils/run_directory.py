"""
Per-run output directory manager.

Creates timestamped, named directories with standard subdirectories
so each simulation or training run has its own isolated folder.
"""

import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional


class RunDirectory:
    """Manages a per-run output directory with standard subdirectories.

    Example layout::

        simulation_output/
          2026-02-17_143052_rush-hour-test/
            metadata.json
            data/
            plots/
            reports/
    """

    def __init__(
        self,
        base_dir: str,
        run_name: Optional[str] = None,
        subdirs: Optional[List[str]] = None,
    ):
        """
        Create a timestamped run directory.

        Args:
            base_dir: Parent directory (e.g. ``./simulation_output``).
            run_name: Human-readable name for this run.  Converted to a
                filesystem-safe slug.  If *None*, a short UUID is used.
            subdirs: Subdirectories to create inside the run folder.
                Defaults to ``["data", "plots"]``.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        slug = self._slugify(run_name) if run_name else str(uuid.uuid4())[:8]
        self.name = f"{timestamp}_{slug}"
        self.root = os.path.join(base_dir, self.name)

        subdirs = subdirs or ["data", "plots"]
        self.dirs: Dict[str, str] = {}
        os.makedirs(self.root, exist_ok=True)
        for sd in subdirs:
            path = os.path.join(self.root, sd)
            os.makedirs(path, exist_ok=True)
            self.dirs[sd] = path

    # -- convenience properties ---------------------------------------------

    @property
    def data_dir(self) -> str:
        return self.dirs.get("data", self.root)

    @property
    def plots_dir(self) -> str:
        return self.dirs.get("plots", self.root)

    @property
    def models_dir(self) -> str:
        return self.dirs.get("models", self.root)

    @property
    def reports_dir(self) -> str:
        return self.dirs.get("reports", self.root)

    # -- path helpers --------------------------------------------------------

    def data_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def plot_path(self, filename: str) -> str:
        return os.path.join(self.plots_dir, filename)

    def model_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)

    # -- metadata ------------------------------------------------------------

    def save_metadata(self, cli_args: dict, extra: Optional[dict] = None):
        """Write ``metadata.json`` with CLI args, timestamp, and extras."""
        metadata = {
            "run_name": self.name,
            "created_at": datetime.now().isoformat(),
            "cli_args": {k: str(v) for k, v in cli_args.items()},
        }
        if extra:
            metadata.update(extra)
        with open(os.path.join(self.root, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a human name to a filesystem-safe slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        return slug[:50] or "unnamed"
