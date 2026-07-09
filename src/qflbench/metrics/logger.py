"""Run logger: collects per-round metrics + communication accounting to JSON.

Kept dependency-free (json) so it runs anywhere. W&B/MLflow can wrap this later;
the JSON record is the source of truth for the heterogeneity-tax plots.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

from ..core.types import RoundMetrics


class RunLogger:
    def __init__(self, run_dir: str, config: Dict[str, Any]) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.config = config
        self.rounds: List[RoundMetrics] = []
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_round(self, rm: RoundMetrics) -> None:
        self.rounds.append(rm)
        g = rm.global_metrics or {}
        gstr = " ".join(f"{k}={v:.3f}" for k, v in g.items()) if g else "(no global)"
        print(
            f"[round {rm.round_idx:03d}] {gstr} | "
            f"up={rm.uplink_bytes/1e3:.1f}KB down={rm.downlink_bytes/1e3:.1f}KB"
        )

    def finalize(self) -> Dict[str, Any]:
        history = [asdict(r) for r in self.rounds]
        summary = {
            "config": self.config,
            "history": history,
            "totals": {
                "rounds": len(self.rounds),
                "total_uplink_bytes": sum(r.uplink_bytes for r in self.rounds),
                "total_downlink_bytes": sum(r.downlink_bytes for r in self.rounds),
            },
        }
        with open(os.path.join(self.run_dir, "history.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary
