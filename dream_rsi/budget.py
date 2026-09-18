"""GPU budget accounting for the discovery loop.

Kaggle grants a fixed weekly GPU allowance, and the submission fit has to come
out of the same pot as the search. A loop that spends the whole allowance
discovering a good candidate and then cannot train it has achieved nothing, so
the reserve is withheld from the search rather than hoped for.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass
class Budget:
    remaining_hours: float
    reserve_hours: float          # withheld for the final fit + test inference

    @property
    def searchable_hours(self) -> float:
        return max(0.0, self.remaining_hours - self.reserve_hours)

    def affords(self, estimate_hours: float) -> bool:
        return estimate_hours <= self.searchable_hours

    def spend(self, hours: float) -> "Budget":
        return Budget(max(0.0, self.remaining_hours - hours), self.reserve_hours)


def read_quota(resource: str = "GPU") -> float | None:
    """Remaining accelerator hours, or None if the quota cannot be read."""
    try:
        out = subprocess.run(["kaggle", "quota"], capture_output=True, text=True,
                             timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if line.strip().startswith(resource):
            nums = re.findall(r"([\d.]+)h", line)
            if len(nums) >= 2:
                return float(nums[1])        # used, remaining, total
    return None


def estimate_round_hours(history_seconds: list[float], workers: int,
                         default_minutes: float = 20.0) -> float:
    """Project a round's cost from what attempts have actually cost so far."""
    if history_seconds:
        per_attempt = sum(history_seconds) / len(history_seconds)
    else:
        per_attempt = default_minutes * 60
    return per_attempt * workers / 3600
