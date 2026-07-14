"""Load config.yaml and cases.yaml into typed objects."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from .providers import Case


@dataclass
class RunConfig:
    run_id: str
    providers: dict[str, dict[str, Any]]
    out_dir: str
    cases_file: str
    timeout_s: float = 900.0
    poll_interval_s: float = 6.0
    mode: str = "text"                       # default input mode: text | image
    reference_images: dict[str, Any] = field(default_factory=dict)
    refs_dir: str = "cases/refs"
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str = "config.yaml") -> RunConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    ref = raw.get("reference_images", {}) or {}
    return RunConfig(
        run_id=str(raw.get("run_id", "run")),
        providers=raw.get("providers", {}) or {},
        out_dir=raw.get("out_dir", "results"),
        cases_file=raw.get("cases_file", "cases/cases.yaml"),
        timeout_s=float(raw.get("timeout_s", 900)),
        poll_interval_s=float(raw.get("poll_interval_s", 6)),
        mode=str(raw.get("mode", "text")),
        reference_images=ref,
        refs_dir=str(ref.get("dir", "cases/refs")),
        extra={"semantic": raw.get("semantic", {}) or {}},
    )


def load_cases(path: str) -> list[Case]:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cases = []
    for c in raw.get("cases", []):
        cases.append(Case(
            id=c["id"],
            category=c.get("category", "prop"),
            prompt=c.get("prompt", ""),
            mode=c.get("mode", "text"),
            image_path=c.get("image_path"),
            expect_watertight=bool(c.get("expect_watertight", True)),
            expect_symmetry=bool(c.get("expect_symmetry", False)),
            notes=c.get("notes", ""),
        ))
    return cases


def stable_seed(run_id: str) -> int:
    """Deterministic 32-bit seed from the run id (no wall-clock)."""
    h = 2166136261
    for ch in run_id:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h
