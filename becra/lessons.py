from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .tools import evidence_toolchain_preprocessing_depth


@dataclass
class Lesson:
    lesson_id: str
    lesson_type: str
    toolchain: str
    task_category: str
    activation_conditions: dict[str, Any]
    phenomenon: str
    analysis: str
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    tool: str = ""
    evidence_pred_len: int | None = None
    verification_effect: float | None = None
    verification_support: int | None = None
    verification_episodes: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in (
            "evidence_pred_len",
            "verification_effect",
            "verification_support",
            "verification_episodes",
        ):
            if d.get(key) is None:
                d.pop(key, None)
        return d


def read_lessons(path: str | Path) -> list[Lesson]:
    payload = json.loads(Path(path).read_text())
    return [_lesson_from_dict(item) for item in payload]


def write_lessons(
    path: str | Path,
    lessons: Iterable[Lesson | dict[str, Any]],
    append: bool = False,
    dedupe_by: str = "lesson_id",
) -> list[dict[str, Any]]:
    """Persist lessons to JSON.

    When append=True and the file already exists, existing entries are merged with new ones;
    on collision of dedupe_by the newer entry overwrites the older one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_payload = [item.to_dict() if isinstance(item, Lesson) else dict(item) for item in lessons]
    if append and path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = []
        merged = _merge_by_key(existing, new_payload, key=dedupe_by)
    else:
        merged = _merge_by_key([], new_payload, key=dedupe_by)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return merged


def merge_lesson_collections(*collections: Iterable[Lesson | dict[str, Any]], dedupe_by: str = "lesson_id") -> list[Lesson]:
    """Combine multiple lesson sources into a single Lesson list; later sources override earlier ones."""
    by_key: dict[str, Lesson] = {}
    for collection in collections:
        for item in collection:
            lesson = item if isinstance(item, Lesson) else _lesson_from_dict(item)
            key = getattr(lesson, dedupe_by)
            by_key[key] = lesson
    return list(by_key.values())


def load_lessons(paths: Iterable[str | Path] | None = None) -> list[Lesson]:
    """Load lessons from one or more JSON files; later files override earlier ones on lesson_id."""
    collections: list[list[Lesson]] = []
    for path in paths or []:
        p = Path(path)
        if not p.exists():
            continue
        collections.append(read_lessons(p))
    return merge_lesson_collections(*collections)


_ACTIVATION_KEY_PRIORITY = (
    "missing_rate",
    "num_variates",
    "seasonality_level",
    "trend_level",
    "volatility_level",
    "anomaly_ratio",
    "series_length",
    "mean_abs_correlation",
    "missing_pattern",
)

# Ordinal categoricals: strategy lessons may widen to lists covering positive explore runs.
_CATEGORICAL_ORDINAL: dict[str, list[str]] = {
    "volatility_level": ["low", "medium", "high"],
    "trend_level": ["weak", "moderate", "strong"],
    "seasonality_level": ["weak", "moderate", "strong"],
}

_NUMERIC_WIDEN_REL_SLACK = 0.05


def _widen_numeric_rule(rule: str, pos_vals: list[Any]) -> str:
    stripped = rule.strip()
    for op in (">=", ">", "<=", "<"):
        if not stripped.startswith(op):
            continue
        try:
            pos_nums = [float(v) for v in pos_vals]
            rhs = float(stripped[len(op) :].strip())
        except (TypeError, ValueError):
            return rule
        if op in (">", ">="):
            floor = min(pos_nums)
            margin = max(abs(floor) * _NUMERIC_WIDEN_REL_SLACK, 0.01)
            new_bound = min(rhs, floor - margin)
            return f">{new_bound:.6g}" if op == ">" else f">={new_bound:.6g}"
        ceil = max(pos_nums)
        margin = max(abs(ceil) * _NUMERIC_WIDEN_REL_SLACK, 0.01)
        new_bound = max(rhs, ceil + margin)
        return f"<{new_bound:.6g}" if op == "<" else f"<={new_bound:.6g}"
    return rule


def _heuristic_widen_numeric(key: str, rule: str) -> str:
    """Planning-time slack when positive explore metas are not stored on the lesson."""
    stripped = rule.strip()
    for op in (">=", ">"):
        if not stripped.startswith(op):
            continue
        try:
            rhs = float(stripped[len(op) :].strip())
        except ValueError:
            return rule
        if key in ("mean_abs_correlation", "anomaly_ratio"):
            new_rhs = max(0.0, rhs * 0.7)
        elif key == "num_variates":
            new_rhs = max(1.0, rhs * 0.5)
        elif key == "series_length":
            new_rhs = max(1000.0, rhs * 0.6)
        elif key == "dominant_period":
            new_rhs = max(1.0, rhs * 0.5)
        else:
            new_rhs = rhs * 0.8
        return (f">{new_rhs:.6g}" if op == ">" else f">={new_rhs:.6g}").rstrip("0").rstrip(".")
    for op in ("<=", "<"):
        if not stripped.startswith(op):
            continue
        try:
            rhs = float(stripped[len(op) :].strip())
        except ValueError:
            return rule
        scale = 1.5 if key == "num_variates" else 1.2
        new_rhs = rhs * scale
        return (f"<{new_rhs:.6g}" if op == "<" else f"<={new_rhs:.6g}").rstrip("0").rstrip(".")
    return rule


def widen_strategy_activation(
    conditions: dict[str, Any],
    positive_metas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Broaden strategy-lesson activation for hold-out transfer (paper Alg. 2 / 4).

    When ``positive_metas`` is provided (induce), envelope all positive explore datasets.
    Otherwise apply conservative heuristic slack at planning time for legacy lessons.
    """
    base = cap_activation_conditions(dict(conditions), max_keys=2)
    if not base:
        return base
    metas = positive_metas or []
    out: dict[str, Any] = {}

    for key, rule in base.items():
        if key == "pred_len":
            out[key] = rule
            continue
        pos_vals = [m.get(key) for m in metas if key in m and m.get(key) is not None]

        if key in _CATEGORICAL_ORDINAL:
            levels = _CATEGORICAL_ORDINAL[key]
            stated = [str(x) for x in rule] if isinstance(rule, list) else [str(rule)]
            from_pos = [str(v) for v in pos_vals if str(v) in levels]
            if from_pos:
                idx_lo = min(levels.index(v) for v in from_pos)
                idx_hi = max(levels.index(v) for v in from_pos)
                for s in stated:
                    if s in levels:
                        idx_lo = min(idx_lo, levels.index(s))
                        idx_hi = max(idx_hi, levels.index(s))
                idx_lo = max(0, idx_lo - 1)
                idx_hi = min(len(levels) - 1, idx_hi + 1)
                combined = levels[idx_lo : idx_hi + 1]
            elif any(s in levels for s in stated):
                idx_lo = min(levels.index(s) for s in stated if s in levels)
                idx_hi = max(levels.index(s) for s in stated if s in levels)
                idx_lo = max(0, idx_lo - 1)
                idx_hi = min(len(levels) - 1, idx_hi + 1)
                combined = levels[idx_lo : idx_hi + 1]
            else:
                combined = stated
            out[key] = combined if len(combined) > 1 else combined[0]
            continue

        if pos_vals:
            out[key] = _widen_numeric_rule(str(rule), pos_vals)
        elif isinstance(rule, str):
            out[key] = _heuristic_widen_numeric(key, rule)
        else:
            out[key] = rule

    return cap_activation_conditions(out, max_keys=2)


def _pos_neg_numeric_overlap(key: str, pos_metas: list[dict[str, Any]], neg_metas: list[dict[str, Any]]) -> bool:
    """True when pos and neg groups share overlapping numeric support (rule would be non-discriminative)."""
    pos_vals = [float(m[key]) for m in pos_metas if m.get(key) is not None]
    neg_vals = [float(m[key]) for m in neg_metas if m.get(key) is not None]
    if not pos_vals or not neg_vals:
        return False
    return max(min(pos_vals), min(neg_vals)) <= min(max(pos_vals), max(neg_vals))


def derive_strategy_activation_from_contrast(
    pos_metas: list[dict[str, Any]],
    neg_metas: list[dict[str, Any]],
    *,
    llm_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build broad strategy activation from contrastive explore groups (Alg. 2)."""
    from .induction import _contrast_summary

    if not pos_metas:
        return cap_activation_conditions(dict(llm_fallback or {}), max_keys=2)
    summary = _contrast_summary(pos_metas, neg_metas)
    cat_diff = summary.get("categorical_differences") or {}
    num_diff = summary.get("numeric_differences") or {}
    out: dict[str, Any] = {}

    for key in _ACTIVATION_KEY_PRIORITY:
        if len(out) >= 2:
            break
        if key not in cat_diff:
            continue
        vals = [m.get(key) for m in pos_metas if m.get(key) is not None]
        if key in _CATEGORICAL_ORDINAL:
            levels = _CATEGORICAL_ORDINAL[key]
            uniq = sorted({str(v) for v in vals if str(v) in levels}, key=levels.index)
            if uniq:
                idx_lo = max(0, min(levels.index(v) for v in uniq) - 1)
                idx_hi = min(len(levels) - 1, max(levels.index(v) for v in uniq) + 1)
                rule: Any = levels[idx_lo : idx_hi + 1]
                out[key] = rule if len(rule) > 1 else rule[0]
        else:
            out[key] = sorted({str(v) for v in vals})

    for key in _ACTIVATION_KEY_PRIORITY:
        if len(out) >= 2:
            break
        if key not in num_diff or key in out:
            continue
        if _pos_neg_numeric_overlap(key, pos_metas, neg_metas):
            continue
        vals = [float(m[key]) for m in pos_metas if m.get(key) is not None]
        if not vals:
            continue
        floor = min(vals)
        margin = max(abs(floor) * _NUMERIC_WIDEN_REL_SLACK, 0.01)
        out[key] = f">{floor - margin:.6g}".rstrip("0").rstrip(".")

    if out:
        return cap_activation_conditions(out, max_keys=2)
    if llm_fallback:
        return widen_strategy_activation(llm_fallback, pos_metas)
    return {}


def toolchain_explore_mean_reward(
    toolchain: str,
    explore_csv: str | Path,
    source_datasets: list[str],
) -> float:
    """Mean explore reward for a toolchain on source datasets (planning tie-break)."""
    from .induction import load_result_rows

    path = Path(explore_csv)
    if not path.is_file():
        return float("-inf")
    rows = load_result_rows(path)
    rewards = [
        float(r["reward"])
        for r in rows
        if r["toolchain"] == toolchain and r["dataset"] in source_datasets
    ]
    if not rewards:
        return float("-inf")
    return float(sum(rewards) / len(rewards))


def _toolchain_strict_match_raw(
    pool: list[Lesson],
    meta: dict[str, Any],
    pred_len: int,
    toolchain: str,
) -> bool:
    for lesson in pool:
        if lesson.toolchain != toolchain:
            continue
        cond = normalize_activation_conditions(dict(lesson.activation_conditions))
        if lesson.evidence_pred_len is not None:
            cond = bind_pred_len_condition(cond, int(lesson.evidence_pred_len))
        if condition_matches(meta_for_horizon(meta, pred_len), cond):
            return True
    return False


def apply_strategy_activation_for_planning(
    pool: list[Lesson],
    meta: dict[str, Any],
    pred_len: int,
    explore_csv: str | Path,
    source_datasets: list[str],
    *,
    threshold_quantile: float = 0.5,
) -> list[Lesson]:
    """Selectively replace strategy activation when a toolchain has no raw strict match."""
    from .induction import dataset_meta_map, load_result_rows

    path = Path(explore_csv)
    if not path.is_file():
        return pool
    rows = load_result_rows(path)
    meta_by_ds = dataset_meta_map(source_datasets)
    contrast_by_tc: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

    for toolchain in {lesson.toolchain for lesson in pool if _is_strategy_lesson(lesson)}:
        subset = [r for r in rows if r["toolchain"] == toolchain and r["dataset"] in meta_by_ds]
        if len(subset) < 2:
            continue
        import numpy as np

        rewards = [float(r["reward"]) for r in subset]
        threshold = float(np.quantile(rewards, threshold_quantile))
        pos_rows = [r for r in subset if float(r["reward"]) >= threshold]
        neg_rows = [r for r in subset if float(r["reward"]) < threshold]
        if not pos_rows:
            continue
        contrast_by_tc[toolchain] = (
            [meta_by_ds[r["dataset"]] for r in pos_rows],
            [meta_by_ds[r["dataset"]] for r in neg_rows],
        )

    out: list[Lesson] = []
    for lesson in pool:
        if not _is_strategy_lesson(lesson) or lesson.toolchain not in contrast_by_tc:
            out.append(lesson)
            continue
        if evidence_toolchain_preprocessing_depth(lesson.toolchain) == 0:
            out.append(lesson)
            continue
        if _toolchain_strict_match_raw(pool, meta, pred_len, lesson.toolchain):
            out.append(lesson)
            continue
        pos_metas, neg_metas = contrast_by_tc[lesson.toolchain]
        new_cond = derive_strategy_activation_from_contrast(
            pos_metas,
            neg_metas,
            llm_fallback=lesson.activation_conditions,
        )
        out.append(
            Lesson(
                lesson_id=lesson.lesson_id,
                lesson_type=lesson.lesson_type,
                toolchain=lesson.toolchain,
                task_category=lesson.task_category,
                activation_conditions=new_cond,
                phenomenon=lesson.phenomenon,
                analysis=lesson.analysis,
                confidence=lesson.confidence,
                tags=list(lesson.tags or []) + ["strategy_activation_widened"],
                tool=lesson.tool,
                evidence_pred_len=lesson.evidence_pred_len,
            )
        )
    return out


def cap_activation_conditions(conditions: dict[str, Any], max_keys: int = 2) -> dict[str, Any]:
    """Keep at most ``max_keys`` meta-feature keys (excluding pred_len, added separately)."""
    cond = normalize_activation_conditions(dict(conditions))
    cond = {k: v for k, v in cond.items() if k != "pred_len"}
    if len(cond) <= max_keys:
        return cond
    kept: list[str] = []
    for key in _ACTIVATION_KEY_PRIORITY:
        if key in cond and len(kept) < max_keys:
            kept.append(key)
    for key in cond:
        if key not in kept and len(kept) < max_keys:
            kept.append(key)
    return {k: cond[k] for k in kept}


def bind_pred_len_condition(conditions: dict[str, Any], pred_len: int) -> dict[str, Any]:
    """Scope a lesson to one forecast horizon (used after induce and for matching)."""
    out = cap_activation_conditions(dict(conditions), max_keys=2)
    out = normalize_activation_conditions(out)
    pl = int(pred_len)
    existing = out.get("pred_len")
    if existing is None:
        out["pred_len"] = [pl]
    elif isinstance(existing, list):
        if pl not in [int(x) for x in existing]:
            out["pred_len"] = [int(x) for x in existing] + [pl]
    else:
        out["pred_len"] = [int(existing), pl] if int(existing) != pl else [pl]
    return out


def effective_activation_conditions(lesson: Lesson) -> dict[str, Any]:
    """Dataset meta rules plus optional horizon lock from ``evidence_pred_len``."""
    cond = normalize_activation_conditions(dict(lesson.activation_conditions))
    if lesson.evidence_pred_len is not None:
        return bind_pred_len_condition(cond, int(lesson.evidence_pred_len))
    return cond


def meta_for_horizon(meta: dict[str, Any], pred_len: int) -> dict[str, Any]:
    """Attach the planning/verify horizon so ``pred_len`` rules can match."""
    out = dict(meta)
    out["pred_len"] = int(pred_len)
    return out


def lesson_matches(meta: dict[str, Any], lesson: Lesson, pred_len: int) -> bool:
    return condition_matches(
        meta_for_horizon(meta, pred_len),
        effective_activation_conditions(lesson),
    )


def activation_overlap_count(meta: dict[str, Any], lesson: Lesson, pred_len: int) -> int:
    """How many non-pred_len activation keys match this meta horizon profile."""
    cond = effective_activation_conditions(lesson)
    meta_h = meta_for_horizon(meta, pred_len)
    meta_keys = [k for k in cond if k != "pred_len"]
    return sum(
        1
        for key in meta_keys
        if key in meta_h and _match_rule(meta_h[key], cond[key])
    )


def _is_strategy_lesson(lesson: Lesson) -> bool:
    if lesson.task_category == "Strategy":
        return True
    tags = lesson.tags or []
    return "strategy_level" in tags


def _vacuous_strategy_activation(lesson: Lesson) -> bool:
    """True when a Strategy lesson's categorical rules cover every ordinal level (no info)."""
    if not _is_strategy_lesson(lesson):
        return False
    cond = lesson.activation_conditions or {}
    for key, levels in _CATEGORICAL_ORDINAL.items():
        rule = cond.get(key)
        if isinstance(rule, (list, tuple, set)) and set(map(str, rule)) >= set(levels):
            return True
    return False


def _planning_rank_key(
    lesson: Lesson,
    overlap: int,
    strict: bool,
) -> tuple[int, int, float, int, float, int, str]:
    """Alg. 4 retrieval order: strict > overlap > verified effect > parsimony > confidence.

    Verified causal effect still dominates. When effects are equal/absent (e.g. plan
    before verify, or tied verify), prefer shallower evidence toolchains so induce
    noise + deep-preprocess confidence cannot crowd out plain baselines. Chain
    parsimony is a documented Alg. 4 fuzzy-area tie-break (no dataset hardcoding).
    Vacuous all-level Strategy activations do not receive the strategy bonus.
    """
    vacuous = _vacuous_strategy_activation(lesson)
    strategy_bonus = int(_is_strategy_lesson(lesson) and not vacuous)
    return (
        int(strict),
        overlap,
        float(lesson.verification_effect or 0.0),
        -evidence_toolchain_preprocessing_depth(lesson.toolchain),
        # Vacuous all-level Strategy lessons sort below concrete stage lessons.
        -int(vacuous),
        float(lesson.confidence),
        strategy_bonus,
        lesson.lesson_id,
    )


def rank_lessons_by_meta_overlap(
    lesson_pool: Iterable[Lesson],
    meta: dict[str, Any],
    pred_len: int,
) -> list[tuple[Lesson, int, bool]]:
    """Sort lessons for Alg. 4 retrieval (strict match, overlap, strategy, evidence parsimony)."""
    pl = int(pred_len)
    scored: list[tuple[Lesson, int, bool]] = []
    for lesson in lesson_pool:
        cond = effective_activation_conditions(lesson)
        meta_h = meta_for_horizon(meta, pl)
        if "pred_len" in cond and not _match_rule(meta_h.get("pred_len"), cond["pred_len"]):
            continue
        overlap = activation_overlap_count(meta, lesson, pl)
        strict = lesson_matches(meta, lesson, pl)
        scored.append((lesson, overlap, strict))
    scored.sort(key=lambda item: _planning_rank_key(item[0], item[1], item[2]), reverse=True)
    return scored


def _toolchain_has_mechanism_match(matched: list[Lesson], toolchain: str) -> bool:
    """True if toolchain has a strict-matched Forecasting or non-vacuous Strategy lesson."""
    for lesson in matched:
        if lesson.toolchain != toolchain:
            continue
        if lesson.task_category == "Forecasting":
            return True
        if _is_strategy_lesson(lesson) and not _vacuous_strategy_activation(lesson):
            return True
    return False


def _top_k_toolchain_diverse(matched: list[Lesson], top_k: int) -> list[Lesson]:
    """Cap prompt size while keeping at least one lesson per evidence toolchain slug.

    Prefer toolchains with Forecasting / non-vacuous Strategy support so a
    Normalization-only match cannot nominate a forecast model by itself.
    """
    if top_k <= 0 or len(matched) <= top_k:
        return list(matched)
    buckets: dict[str, list[Lesson]] = {}
    toolchain_order: list[str] = []
    for lesson in matched:
        tc = lesson.toolchain
        if tc not in buckets:
            buckets[tc] = []
            toolchain_order.append(tc)
        buckets[tc].append(lesson)

    primary_order = [tc for tc in toolchain_order if _toolchain_has_mechanism_match(matched, tc)]
    fallback_order = [tc for tc in toolchain_order if tc not in primary_order]

    selected: list[Lesson] = []

    def _fill(order: list[str]) -> None:
        while len(selected) < top_k:
            added = False
            for tc in order:
                if buckets.get(tc):
                    selected.append(buckets[tc].pop(0))
                    added = True
                    if len(selected) >= top_k:
                        return
            if not added:
                return

    _fill(primary_order)
    if len(selected) < top_k:
        _fill(fallback_order)
    return selected


def _matched_toolchain_categories(matched: list[Lesson], toolchain: str) -> set[str]:
    return {lesson.task_category for lesson in matched if lesson.toolchain == toolchain}


def _prefer_narrow_over_widened_strategy(
    narrow_matched: list[Lesson],
    narrow_primary: str,
    wide_primary: str,
    wide_matched: list[Lesson],
) -> bool:
    """Keep a strong narrow stage winner when widening only adds generic strategy slugs."""
    if narrow_primary == wide_primary:
        return False
    if _matched_toolchain_categories(narrow_matched, narrow_primary) == {"Decomposition"}:
        return False
    narrow_has_non_strategy = any(
        lesson.toolchain == narrow_primary and not _is_strategy_lesson(lesson)
        for lesson in narrow_matched
    )
    wide_is_widened_strategy = any(
        lesson.toolchain == wide_primary
        and _is_strategy_lesson(lesson)
        and "strategy_activation_widened" in (lesson.tags or [])
        for lesson in wide_matched
    )
    return narrow_has_non_strategy and wide_is_widened_strategy


def _primary_toolchain_rank(
    toolchain: str,
    matched: list[Lesson],
    meta: dict[str, Any],  # noqa: ARG001 — kept for API compatibility
    explore_key,
) -> tuple[float, float, int, str]:
    """Evidence-first hint ranking: verified causal support of matched lessons first,
    then mean explore reward on source datasets; chain parsimony only as final tie-break."""
    support = 0.0
    for lesson in matched:
        if lesson.toolchain != toolchain:
            continue
        direction = 1.0 if lesson.lesson_type == "positive" else -1.0
        effect = float(lesson.verification_effect or 0.0)
        support += direction * float(lesson.confidence) * (1.0 + max(0.0, effect))
    return (
        support,
        explore_key(toolchain),
        -evidence_toolchain_preprocessing_depth(toolchain),
        toolchain,
    )


def primary_evidence_toolchain(
    matched: list[Lesson],
    *,
    raw_pool: list[Lesson] | None = None,
    meta: dict[str, Any] | None = None,
    pred_len: int | None = None,
    explore_csv: str | Path | None = None,
    source_datasets: list[str] | None = None,
) -> str | None:
    """Primary evidence slug for stage composition (Strategy lesson or top-ranked match)."""
    if not matched:
        return None

    meta_h = meta or {}
    pl = int(pred_len if pred_len is not None else meta_h.get("pred_len") or 0)
    explore_key = (
        lambda tc: toolchain_explore_mean_reward(tc, explore_csv, source_datasets or [])
        if explore_csv and source_datasets
        else float("-inf")
    )
    toolchains = sorted({lesson.toolchain for lesson in matched})
    ranked_primary = max(
        toolchains,
        key=lambda tc: _primary_toolchain_rank(tc, matched, meta_h, explore_key),
    )

    widened_strategies = [
        lesson
        for lesson in matched
        if _is_strategy_lesson(lesson) and "strategy_activation_widened" in (lesson.tags or [])
    ]
    if widened_strategies and raw_pool and pl and meta_h:
        narrow_ranked = rank_lessons_by_meta_overlap(raw_pool, meta_h, pl)
        narrow_matched = [lesson for lesson, _, strict in narrow_ranked if strict]
        narrow_primary = primary_evidence_toolchain(
            narrow_matched,
            meta=meta_h,
            pred_len=pl,
            explore_csv=explore_csv,
            source_datasets=source_datasets,
        )
        if narrow_primary and _prefer_narrow_over_widened_strategy(
            narrow_matched,
            narrow_primary,
            ranked_primary,
            matched,
        ):
            return narrow_primary

    return ranked_primary


def lessons_for_paper_planning(
    lesson_pool: Iterable[Lesson],
    meta: dict[str, Any],
    pred_len: int,
    *,
    top_k: int = 5,
    raw_pool: list[Lesson] | None = None,
    explore_csv: str | Path | None = None,
    source_datasets: list[str] | None = None,
    inject_scope: str = "primary",
) -> tuple[list[Lesson], list[Lesson]]:
    """Appendix Alg. 4: L_x = {phi_k | f_k(x_new) true}; optional diverse evidence budget.

    inject_scope:
      primary — inject only lessons on primary_evidence_toolchain (legacy default).
      matched — inject top-k diverse lessons from all strict matches. No toolchain is
                reserved or forced; ranking uses meta match and verified evidence.
    """
    ranked = rank_lessons_by_meta_overlap(lesson_pool, meta, pred_len)
    matched = [lesson for lesson, _, strict in ranked if strict]
    primary = primary_evidence_toolchain(
        matched,
        raw_pool=raw_pool,
        meta=meta,
        pred_len=pred_len,
        explore_csv=explore_csv,
        source_datasets=source_datasets,
    )
    if inject_scope == "matched":
        if top_k > 0:
            llm_pool = _top_k_toolchain_diverse(matched, top_k)
        else:
            llm_pool = list(matched)
    else:
        if primary:
            focused = [lesson for lesson in matched if lesson.toolchain == primary]
        else:
            focused = list(matched)
        if top_k > 0:
            llm_pool = _top_k_toolchain_diverse(focused or matched, top_k)
        else:
            llm_pool = list(focused or matched)
    return llm_pool, matched


def lesson_matches_for_planning(
    meta: dict[str, Any],
    lesson: Lesson,
    pred_len: int,
    *,
    min_meta_overlap: int = 1,
) -> bool:
    """Loose matcher (≥ min_meta_overlap keys). Prefer ``lesson_matches`` for verify/planning."""
    cond = effective_activation_conditions(lesson)
    meta_h = meta_for_horizon(meta, pred_len)
    if "pred_len" in cond and not _match_rule(meta_h.get("pred_len"), cond["pred_len"]):
        return False
    meta_keys = [k for k in cond if k != "pred_len"]
    if not meta_keys:
        return True
    overlap = sum(
        1
        for key in meta_keys
        if key in meta_h and _match_rule(meta_h[key], cond[key])
    )
    return overlap >= max(1, int(min_meta_overlap))


def normalize_activation_conditions(conditions: dict[str, Any]) -> dict[str, Any]:
    """Convert common LLM JSON shapes into rules understood by condition_matches."""
    if not conditions:
        return {}
    if "feature_name" in conditions:
        key = str(conditions["feature_name"])
        op = str(conditions.get("operator", "==")).strip().lower()
        threshold = conditions.get("threshold")
        if op in {"==", "=", "eq"}:
            return {key: [threshold] if not isinstance(threshold, list) else threshold}
        if op in {"<=", "lt", "<"}:
            return {key: f"<={threshold}" if op in {"<=", "lt"} else f"<{threshold}"}
        if op in {">=", "gt", ">"}:
            return {key: f">={threshold}" if op in {">=", "gt"} else f">{threshold}"}
    normalized: dict[str, Any] = {}
    for key, rule in conditions.items():
        if not isinstance(rule, str):
            normalized[key] = rule
            continue
        text = rule.strip().lower()
        num = re.search(r"[-+]?\d*\.?\d+", rule)
        if "greater than" in text or "more than" in text:
            normalized[key] = f">={num.group()}" if num else rule
        elif "less than" in text or "fewer than" in text:
            normalized[key] = f"<={num.group()}" if num else rule
        elif text.startswith((">=", "<=", ">", "<", "==")):
            normalized[key] = rule.strip()
        else:
            normalized[key] = rule.strip()
    return normalized


def _lesson_from_dict(item: dict[str, Any]) -> Lesson:
    pl = item.get("evidence_pred_len")
    effect = item.get("verification_effect")
    support = item.get("verification_support")
    return Lesson(
        lesson_id=item["lesson_id"],
        lesson_type=item.get("lesson_type", "positive"),
        toolchain=item["toolchain"],
        task_category=item.get("task_category", "Forecasting"),
        activation_conditions=cap_activation_conditions(
            dict(item.get("activation_conditions") or {}),
            max_keys=2,
        ),
        phenomenon=item.get("phenomenon", ""),
        analysis=item.get("analysis", ""),
        confidence=float(item.get("confidence", 1.0)),
        tags=list(item.get("tags") or []),
        tool=str(item.get("tool") or ""),
        evidence_pred_len=int(pl) if pl is not None else None,
        verification_effect=float(effect) if effect is not None else None,
        verification_support=int(support) if support is not None else None,
        verification_episodes=list(item.get("verification_episodes") or []) or None,
    )


def _merge_by_key(old: list[dict[str, Any]], new: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {item[key]: item for item in old if key in item}
    order: list[str] = [item[key] for item in old if key in item]
    for item in new:
        if key not in item:
            continue
        if item[key] not in by_key:
            order.append(item[key])
        by_key[item[key]] = item
    return [by_key[k] for k in order]


def condition_matches(meta: dict[str, Any], conditions: dict[str, Any]) -> bool:
    conditions = normalize_activation_conditions(conditions)
    for key, rule in conditions.items():
        if key not in meta:
            return False
        if not _match_rule(meta[key], rule):
            return False
    return True


def _match_rule(value: Any, rule: Any) -> bool:
    if isinstance(rule, list):
        try:
            left = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
            return any(left == int(x) for x in rule)
        except (TypeError, ValueError):
            return value in rule
    if isinstance(rule, str):
        stripped = rule.strip()
        for op in (">=", "<=", ">", "<", "=="):
            if stripped.startswith(op):
                rhs = stripped[len(op) :].strip()
                try:
                    left = float(value)
                    right = float(rhs)
                except (TypeError, ValueError):
                    return op == "==" and str(value) == rhs
                if op == ">=":
                    return left >= right
                if op == "<=":
                    return left <= right
                if op == ">":
                    return left > right
                if op == "<":
                    return left < right
                if op == "==":
                    return left == right
        return str(value) == stripped
    return value == rule
