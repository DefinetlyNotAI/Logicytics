"""Local semantic flag matching, opt-in history, and usage reporting."""

from __future__ import annotations

import gzip
import json
import platform
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from html import escape
from pathlib import Path

FLAG_DESCRIPTIONS: Mapping[str, str] = {
    "default": "standard collection sequentially",
    "threaded": "standard collection with bounded parallel workers",
    "minimal": "quick basic essential collection",
    "depth": "deep exhaustive slow collection",
    "performance-check": "sequential collector duration performance analysis",
    "usage": "interaction statistics and flag usage graph",
    "debug": "diagnostic environment configuration and integrity checks",
    "update": "check or explicitly update the Git checkout",
    "dev": "developer contribution manifest and version checks",
    "mode": "select a user facing typed execution mode",
    "modes": "show execution mode profiles scheduling and compatibility aliases",
}
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class FlagMatch:
    """One deterministic flag suggestion and its evidence."""

    input: str
    matched_flag: str | None
    accuracy: float
    source: str
    model_name: str


def _normalized(value: str) -> str:
    """Normalize free-form input into lowercase, whitespace-separated tokens."""
    return " ".join(_TOKEN.findall(value.casefold()))


def _score(query: str, flag: str, description: str) -> float:
    """Score a query using sequence similarity and token overlap with a flag description."""
    normalized_query = _normalized(query)
    candidate = _normalized(f"{flag} {description}")
    sequence = SequenceMatcher(None, normalized_query, candidate).ratio()
    query_tokens = set(normalized_query.split())
    candidate_tokens = set(candidate.split())
    overlap = len(query_tokens.intersection(candidate_tokens)) / max(1, len(query_tokens))
    flag_score = SequenceMatcher(None, normalized_query, _normalized(flag)).ratio()
    return round(max(sequence, overlap, flag_score), 6)


def match_flag(
        user_input: str,
        *,
        threshold: float,
        model_name: str,
        history: Iterable[Mapping[str, object]] = (),
) -> FlagMatch:
    """Match names and descriptions, then consult prior accepted inputs when weak."""
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("semantic flag input must be non-empty")
    ranked = sorted(
        ((_score(user_input, flag, description), flag) for flag, description in FLAG_DESCRIPTIONS.items()),
        key=lambda item: (-item[0], item[1]),
    )
    direct_score, direct_flag = ranked[0]
    if direct_score >= threshold:
        return FlagMatch(user_input, direct_flag, direct_score, "direct", model_name)
    historical: list[tuple[float, str]] = []
    for item in history:
        old_input = item.get("input")
        old_flag = item.get("matched_flag")
        if isinstance(old_input, str) and isinstance(old_flag, str) and old_flag in FLAG_DESCRIPTIONS:
            historical.append(
                (
                    SequenceMatcher(None, _normalized(user_input), _normalized(old_input)).ratio(),
                    old_flag,
                )
            )
    if historical:
        history_score, history_flag = sorted(historical, key=lambda item: (-item[0], item[1]))[0]
        if history_score > direct_score:
            return FlagMatch(user_input, history_flag, round(history_score, 6), "history", model_name)
    return FlagMatch(user_input, None, direct_score, "below_threshold", model_name)


def load_history(path: Path) -> list[dict[str, object]]:
    """Read bounded compressed local history; malformed history fails closed."""
    if not path.exists():
        return []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload[-10_000:] if isinstance(item, dict)]


def record_match(path: Path, match: FlagMatch) -> None:
    """Atomically append one compressed, local-only interaction record."""
    history = load_history(path)
    history.append(
        {
            **asdict(match),
            "timestamp": datetime.now(UTC).isoformat(),
            "device_name": platform.node() or "unknown",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(history[-10_000:], stream, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def record_command(path: Path, command: str, *, mode: str | None = None) -> None:
    """Record one local command interaction when history tracking is enabled."""
    if not command:
        raise ValueError("command history requires a command name")
    matched_flag = mode or command
    input_value = command if mode is None else f"{command} --profile {mode}"
    record_match(
        path,
        FlagMatch(
            input=input_value,
            matched_flag=matched_flag,
            accuracy=1.0,
            source="command",
            model_name="command-history",
        ),
    )


def usage_statistics(history: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate total, accuracy, common values, and per-flag frequencies."""
    records = list(history)

    accuracies: list[float] = []
    flags: Counter[str] = Counter()
    devices: Counter[str] = Counter()
    inputs: Counter[str] = Counter()

    for item in records:
        accuracy = item.get("accuracy")
        matched_flag = item.get("matched_flag")
        device_name = item.get("device_name")
        input_value = item.get("input")

        if isinstance(accuracy, (int, float)):
            accuracies.append(float(accuracy))

        if isinstance(matched_flag, str):
            flags[matched_flag] += 1

        if isinstance(device_name, str):
            devices[device_name] += 1

        if isinstance(input_value, str):
            inputs[input_value] += 1

    return {
        "total_interactions": len(records),
        "average_accuracy": (round(sum(accuracies) / len(accuracies), 6) if accuracies else 0.0),
        "common_device": devices.most_common(1)[0][0] if devices else None,
        "common_input": inputs.most_common(1)[0][0] if inputs else None,
        "per_flag_frequency": dict(sorted(flags.items())),
    }


def write_usage_graph(path: Path, statistics: Mapping[str, object]) -> Path:
    """Write a portable SVG graph for the currently recorded commands and modes."""
    raw_counts = statistics.get("per_flag_frequency", {})
    source_counts = raw_counts if isinstance(raw_counts, Mapping) else {}
    counts = {
        label: count
        for label, count in source_counts.items()
        if isinstance(label, str)
           and isinstance(count, int)
           and not isinstance(count, bool)
           and count > 0
    }
    labels = sorted(counts)
    width, row_height = 760, 30
    height = 70 + row_height * max(len(labels), 1)
    maximum = max(counts.values(), default=1)
    rows: list[str] = []
    for index, label in enumerate(labels):
        count = counts[label]
        y = 48 + index * row_height
        bar_width = int(500 * count / maximum)
        rows.extend(
            (
                f'<text x="12" y="{y + 16}" font-family="monospace" font-size="13">{escape(label)}</text>',
                f'<rect x="190" y="{y}" width="{bar_width}" height="20" fill="#3b82f6"/>',
                f'<text x="{200 + bar_width}" y="{y + 16}" font-family="monospace" font-size="13">{count}</text>',
            )
        )
    if not rows:
        rows.append('<text x="12" y="64" font-family="sans-serif" font-size="13">No tracked interactions yet.</text>')
    svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
            '<text x="12" y="28" font-family="sans-serif" font-size="20">Logicytics command and mode usage</text>'
            + "".join(rows)
            + "</svg>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
