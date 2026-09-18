"""GPU budget accounting for the discovery loop.

Kaggle grants a fixed weekly GPU allowance, and the submission fit has to come
out of the same pot as the search. A loop that spends the whole allowance
discovering a good candidate and then cannot train it has achieved nothing, so
the reserve is withheld from the search rather than hoped for.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from dataclasses import dataclass


@dataclass
class Budget:
    remaining_hours: float
    reserve_hours: float          # withheld for the final fit + test inference
    expires_soon: bool = False    # allowance refreshes before the reserve is needed

    @property
    def searchable_hours(self) -> float:
        """Hours the search may spend.

        An allowance that refreshes before the submission has to be trained is
        use-it-or-lose-it: withholding a reserve from it protects nothing and
        simply throws the hours away, because the final fit will be paid for out
        of the next allowance.
        """
        if self.expires_soon:
            return max(0.0, self.remaining_hours)
        return max(0.0, self.remaining_hours - self.reserve_hours)

    def affords(self, estimate_hours: float) -> bool:
        return estimate_hours <= self.searchable_hours

    def spend(self, hours: float) -> "Budget":
        return Budget(max(0.0, self.remaining_hours - hours), self.reserve_hours,
                      self.expires_soon)


def read_quota(resource: str = "GPU") -> tuple[float, dt.datetime | None] | None:
    """(remaining hours, refresh time) for an accelerator, or None if unreadable."""
    try:
        out = subprocess.run(["kaggle", "quota"], capture_output=True, text=True,
                             timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if not line.strip().startswith(resource):
            continue
        nums = re.findall(r"([\d.]+)h", line)
        if len(nums) < 2:
            return None
        stamp = re.search(r"(\d{4}-\d{2}-\d{2}T[\d:]+)", line)
        when = None
        if stamp:
            try:
                when = dt.datetime.fromisoformat(stamp.group(1)).replace(tzinfo=dt.timezone.utc)
            except ValueError:
                when = None
        return float(nums[1]), when          # used, remaining, total
    return None


def make_budget(reserve_hours: float, needed_by: dt.datetime | None = None) -> Budget | None:
    """Read the quota and decide whether the reserve applies to this allowance."""
    read = read_quota()
    if read is None:
        return None
    remaining, refresh_at = read
    now = dt.datetime.now(dt.timezone.utc)
    expires_soon = bool(
        refresh_at and refresh_at > now
        and (needed_by is None or refresh_at < needed_by)
    )
    return Budget(remaining, reserve_hours, expires_soon)


def estimate_round_hours(history_seconds: list[float], workers: int,
                         default_minutes: float = 20.0) -> float:
    """Project a round's cost from what attempts have actually cost so far."""
    if history_seconds:
        per_attempt = sum(history_seconds) / len(history_seconds)
    else:
        per_attempt = default_minutes * 60
    return per_attempt * workers / 3600
