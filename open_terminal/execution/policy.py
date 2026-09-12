"""Explicit deployment budgets for the cgroup execution backend."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Limits:
    cpu_millis: int
    memory_bytes: int
    pids: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive integer")

    def fits_within(self, other: "Limits") -> bool:
        return all(
            getattr(self, f.name) <= getattr(other, f.name) for f in fields(self)
        )

    def __add__(self, other: "Limits") -> "Limits":
        return Limits(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    def __mul__(self, count: int) -> "Limits":
        return Limits(**{f.name: getattr(self, f.name) * count for f in fields(self)})


@dataclass(frozen=True)
class ExecutionPolicy:
    cgroup_root: Path
    service: Limits
    helpers: Limits
    compute: Limits
    user: Limits
    max_tasks: int
    max_user_tasks: int
    max_runtime: float = 3600
    # Internal operating defaults, not deployment settings.
    start_timeout: float = 10
    cleanup_timeout: float = 5
    terminate_grace: float = 5
    max_queue: int = 32
    max_user_queue: int = 4
    queue_timeout: float = 30

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cgroup_root, Path)
            or not self.cgroup_root.is_absolute()
            or ".." in self.cgroup_root.parts
        ):
            raise ValueError("cgroup_root must be an absolute path")
        for name in ("service", "helpers", "compute", "user"):
            if not isinstance(getattr(self, name), Limits):
                raise ValueError(f"{name} must contain resource limits")  # noqa: TRY004 - invalid policy
        if not self.user.fits_within(self.compute):
            raise ValueError("user resource limits must fit within compute budgets")
        for name in ("max_tasks", "max_user_tasks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_user_tasks > self.max_tasks:
            raise ValueError("max_user_tasks cannot exceed max_tasks")
        for name in ("max_queue", "max_user_queue"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "queue_timeout",
            "start_timeout",
            "cleanup_timeout",
            "max_runtime",
            "terminate_grace",
        ):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")

    @property
    def total(self) -> Limits:
        return self.service + self.helpers + self.compute


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate policy key: {key}")
        result[key] = value
    return result


def load_policy(path: str | Path) -> ExecutionPolicy:
    data = json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)
    if not isinstance(data, dict):
        raise ValueError("execution policy must be a JSON object")  # noqa: TRY004 - invalid policy
    required = {
        "cgroup_root", "service", "helpers", "compute", "user",
        "max_tasks", "max_user_tasks",
    }
    allowed = required | {"max_runtime"}
    if data.keys() - allowed or required - data.keys():
        raise ValueError("execution policy has unknown or missing keys")
    for name in ("service", "helpers", "compute", "user"):
        limits = data[name]
        if not isinstance(limits, dict) or limits.keys() != {
            f.name for f in fields(Limits)
        }:
            raise ValueError(
                f"{name} must specify exactly cpu_millis, memory_bytes, pids"
            )
        data[name] = Limits(**limits)
    if not isinstance(data["cgroup_root"], str):
        raise ValueError("cgroup_root must be an absolute path string")  # noqa: TRY004 - invalid policy
    data["cgroup_root"] = Path(data["cgroup_root"])
    return ExecutionPolicy(**data)
