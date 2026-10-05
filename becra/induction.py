from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import get_dataset, get_toolchain
from .lessons import (
    Lesson,
    bind_pred_len_condition,
    cap_activation_conditions,
    lesson_matches,
    merge_lesson_collections,
    meta_for_horizon,
    normalize_activation_conditions,
)
from .meta_features import compute_meta_features
from .planner import plan_toolchain
from .prompt_templates import (
    lesson_induction_group_prompt,
    lesson_induction_paper_prompt,
)
from .tools import (
    is_preprocessing_lesson,
    is_valid_lesson_tool,
    normalize_tool_name,
    stages_dict,
    tool_at_stage,
    toolchains_matching_tool,
)
from .runner import run_forecast
from .ucb import forecasting_model_from_arm, preprocessing_depth_from_arm, reward_from_metrics


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    cleaned = []
    for row in rows:
        if row.get("mse") in {"", "None", None} or row.get("mae") in {"", "None", None}:
            continue
        row = dict(row)
        row["mse"] = float(row["mse"])
        row["mae"] = float(row["mae"])
        row["pred_len"] = int(row["pred_len"])
        row["reward"] = reward_from_metrics(row["mse"], row["mae"])
        cleaned.append(row)
    return cleaned


def dataset_meta_map(dataset_names: list[str]) -> dict[str, dict[str, Any]]:
    return {name: compute_meta_features(get_dataset(name)).to_dict() for name in dataset_names}


def _dedupe_rows_by_dataset_pred_len(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], int(row["pred_len"]))
        if key not in best or row["reward"] > best[key]["reward"]:
            best[key] = row
    return list(best.values())


def _all_contrast_groups(
    rows: list[dict[str, Any]],
    dataset_names: list[str],
    threshold_quantile: float = 0.5,
) -> list[tuple[str, int, list[dict[str, Any]], list[dict[str, Any]], float]]:
    """(toolchain, pred_len, positive_rows, negative_rows, mean_reward_gap)."""
    meta_names = set(dataset_names)
    groups: list[tuple[str, int, list[dict[str, Any]], list[dict[str, Any]], float]] = []
    for toolchain in sorted({row["toolchain"] for row in rows}):
        subset = [
            row for row in _dedupe_rows_by_dataset_pred_len(
                [r for r in rows if r["toolchain"] == toolchain and r["dataset"] in meta_names]
            )
        ]
        if len(subset) < 2:
            continue
        for pred_len in sorted({int(r["pred_len"]) for r in subset}):
            pl_rows = [r for r in subset if int(r["pred_len"]) == pred_len]
            if len(pl_rows) < 2:
                continue
            rewards = np.array([r["reward"] for r in pl_rows], dtype=float)
            threshold = float(np.quantile(rewards, threshold_quantile))
            pos = [r for r in pl_rows if r["reward"] >= threshold]
            neg = [r for r in pl_rows if r["reward"] < threshold]
            if not pos or not neg:
                continue
            gap = float(np.mean([r["reward"] for r in pos]) - np.mean([r["reward"] for r in neg]))
            groups.append((toolchain, pred_len, pos, neg, gap))
    return groups


def select_contrast_groups(
    rows: list[dict[str, Any]],
    dataset_names: list[str],
    threshold_quantile: float = 0.5,
    max_total_groups: int = 10,
    max_groups_per_toolchain: int = 4,
) -> list[tuple[str, int, list[dict[str, Any]], list[dict[str, Any]]]]:
    """One contrast per (toolchain, pred_len): all positive vs all negative datasets at that horizon."""
    raw = _all_contrast_groups(rows, dataset_names, threshold_quantile)
    if not raw:
        return []
    by_toolchain: dict[str, list[tuple[str, int, list[dict[str, Any]], list[dict[str, Any]], float]]] = {}
    for item in raw:
        by_toolchain.setdefault(item[0], []).append(item)

    selected: list[tuple[str, int, list[dict[str, Any]], list[dict[str, Any]], float]] = []
    for toolchain in sorted(by_toolchain):
        tc_items = sorted(by_toolchain[toolchain], key=lambda x: -x[4])[: max(1, max_groups_per_toolchain)]
        selected.extend(tc_items)

    selected = sorted(selected, key=lambda x: -x[4])[: max(1, max_total_groups)]
    return [(tc, pl, pos, neg) for tc, pl, pos, neg, _ in selected]


def iter_contrast_groups(
    rows: list[dict[str, Any]],
    dataset_names: list[str],
    threshold_quantile: float = 0.5,
    max_total_groups: int = 10,
    max_groups_per_toolchain: int = 4,
    **_: Any,
) -> list[tuple[str, int, list[dict[str, Any]], list[dict[str, Any]]]]:
    return select_contrast_groups(
        rows,
        dataset_names,
        threshold_quantile,
        max_total_groups=max_total_groups,
        max_groups_per_toolchain=max_groups_per_toolchain,
    )


def _all_contrast_groups_by_strategy(
    rows: list[dict[str, Any]],
    dataset_names: list[str],
    threshold_quantile: float = 0.5,
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], float]]:
    """(toolchain, positive_rows, negative_rows, mean_reward_gap) — contrast across (dataset, pred_len)."""
    meta_names = set(dataset_names)
    groups: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], float]] = []
    for toolchain in sorted({row["toolchain"] for row in rows}):
        subset = _dedupe_rows_by_dataset_pred_len(
            [r for r in rows if r["toolchain"] == toolchain and r["dataset"] in meta_names]
        )
        if len(subset) < 2:
            continue
        rewards = np.array([r["reward"] for r in subset], dtype=float)
        threshold = float(np.quantile(rewards, threshold_quantile))
        pos = [r for r in subset if r["reward"] >= threshold]
        neg = [r for r in subset if r["reward"] < threshold]
        if not pos or not neg:
            continue
        gap = float(np.mean([r["reward"] for r in pos]) - np.mean([r["reward"] for r in neg]))
        groups.append((toolchain, pos, neg, gap))
    return groups


def select_contrast_strategy_groups(
    rows: list[dict[str, Any]],
    dataset_names: list[str],
    threshold_quantile: float = 0.5,
    max_total_groups: int = 10,
    **_: Any,
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """One contrast per toolchain (paper: fixed strategy a with nonempty C+/C-).

    Budgeting is a fuzzy-area engineering choice. Pure gap-ranking tends to keep
    only high-variance / deep-preprocess variants and starve shallow baselines in
    the same forecast-model family. Select by round-robin across forecast families,
    preferring shallow preprocessing depth then larger mean reward gap within each
    family (no dataset- or toolchain-name hardcoding).
    """
    raw = _all_contrast_groups_by_strategy(rows, dataset_names, threshold_quantile)
    if not raw:
        return []
    budget = max(1, int(max_total_groups))
    return _select_strategy_groups_family_diverse(raw, budget)


def _select_strategy_groups_family_diverse(
    raw: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], float]],
    budget: int,
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """Round-robin across forecasting families; shallow-then-gap within family."""
    by_family: dict[str, list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], float]]] = {}
    for item in raw:
        family = forecasting_model_from_arm(item[0])
        by_family.setdefault(family, []).append(item)
    for family, items in by_family.items():
        items.sort(
            key=lambda x: (
                preprocessing_depth_from_arm(x[0]),
                -x[3],
                x[0],
            )
        )

    # Stable family order: families that contain a shallow (depth-0) arm first,
    # then by their best gap (so strong contrasts still get early slots).
    def family_key(fam: str) -> tuple[int, float, str]:
        items = by_family[fam]
        min_depth = min(preprocessing_depth_from_arm(tc) for tc, *_ in items)
        best_gap = max(gap for *_, gap in items)
        return (min_depth, -best_gap, fam)

    families = sorted(by_family.keys(), key=family_key)
    selected: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    seen: set[str] = set()
    idx = 0
    while len(selected) < budget:
        progressed = False
        for fam in families:
            items = by_family[fam]
            if idx >= len(items):
                continue
            tc, pos, neg, _gap = items[idx]
            if tc in seen:
                continue
            selected.append((tc, pos, neg))
            seen.add(tc)
            progressed = True
            if len(selected) >= budget:
                break
        if not progressed:
            break
        idx += 1

    if len(selected) < budget:
        remainder = sorted(raw, key=lambda x: -x[3])
        for tc, pos, neg, _gap in remainder:
            if tc in seen:
                continue
            selected.append((tc, pos, neg))
            seen.add(tc)
            if len(selected) >= budget:
                break
    return selected


def iter_contrast_strategy_groups(
    rows: list[dict[str, Any]],
    dataset_names: list[str],
    threshold_quantile: float = 0.5,
    max_total_groups: int = 10,
    **_: Any,
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    return select_contrast_strategy_groups(
        rows,
        dataset_names,
        threshold_quantile,
        max_total_groups=max_total_groups,
    )


def build_induction_prompts(
    results_csv: Path,
    output_dir: Path,
    dataset_names: list[str],
    threshold_quantile: float = 0.5,
    max_total_groups: int = 10,
    max_groups_per_toolchain: int = 4,
    max_lessons_per_group: int = 1,
    induction_mode: str = "paper",
    **_: Any,
) -> list[Path]:
    rows = load_result_rows(results_csv)
    meta = dataset_meta_map(dataset_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if induction_mode == "paper":
        for toolchain, pos_rows, neg_rows in iter_contrast_strategy_groups(
            rows,
            dataset_names,
            threshold_quantile,
            max_total_groups=max_total_groups,
        ):
            pos_cases = [_compact_meta_with_result(meta[r["dataset"]], r) for r in pos_rows]
            neg_cases = [_compact_meta_with_result(meta[r["dataset"]], r) for r in neg_rows]
            contrast = _contrast_summary(
                [meta[r["dataset"]] for r in pos_rows],
                [meta[r["dataset"]] for r in neg_rows],
            )
            stages = stages_dict(toolchain)
            prompt = lesson_induction_paper_prompt(
                toolchain,
                stages,
                pos_cases,
                neg_cases,
                contrast_summary=contrast,
                max_strategy_lessons=1,
                max_stage_lessons=max(6, max_lessons_per_group),
            )
            out = output_dir / f"{toolchain}_strategy_induction_prompt.txt"
            out.write_text(prompt)
            paths.append(out)
        return paths

    for toolchain, pred_len, pos_rows, neg_rows in iter_contrast_groups(
        rows,
        dataset_names,
        threshold_quantile,
        max_total_groups=max_total_groups,
        max_groups_per_toolchain=max_groups_per_toolchain,
    ):
        pos_cases = [_compact_meta_with_result(meta[r["dataset"]], r) for r in pos_rows]
        neg_cases = [_compact_meta_with_result(meta[r["dataset"]], r) for r in neg_rows]
        contrast = _contrast_summary(
            [meta[r["dataset"]] for r in pos_rows],
            [meta[r["dataset"]] for r in neg_rows],
        )
        prompt = lesson_induction_group_prompt(
            toolchain,
            stages_dict(toolchain),
            pred_len,
            pos_cases,
            neg_cases,
            contrast_summary=contrast,
            max_lessons=max_lessons_per_group,
        )
        out = output_dir / f"{toolchain}_pl{pred_len}_group_induction_prompt.txt"
        out.write_text(prompt)
        paths.append(out)
    return paths


_COMPACT_META_KEYS = (
    "dataset",
    "series_length",
    "num_variates",
    "sampling_frequency",
    "missing_rate",
    "missing_pattern",
    "volatility_level",
    "trend_level",
    "seasonality_level",
    "dominant_period",
    "stationarity_label",
    "anomaly_ratio",
    "mean_abs_correlation",
)
_CATEGORICAL_KEYS = ("missing_pattern", "volatility_level", "trend_level", "seasonality_level", "stationarity_label")
_NUMERIC_KEYS = ("num_variates", "missing_rate", "anomaly_ratio", "mean_abs_correlation", "dominant_period", "series_length")


def induce_lessons_with_llm(
    results_csv: Path,
    dataset_names: list[str],
    client: Any,
    threshold_quantile: float = 0.5,
    llm_raw_dir: Path | None = None,
    max_total_groups: int = 10,
    max_groups_per_toolchain: int = 4,
    max_lessons_per_group: int = 1,
    induction_mode: str = "paper",
    **_: Any,
) -> tuple[list[Lesson], dict[str, Any]]:
    """LLM lesson induction.

    ``paper`` (default): one contrast per fixed strategy (toolchain) over (dataset, pred_len) runs;
    strategy-level + per-stage lessons including Forecasting.

    ``legacy``: one contrast per (toolchain, pred_len); preprocessing-only lessons.
    """
    empty_summary: dict[str, Any] = {
        "requests": 0,
        "raw_lessons": 0,
        "accepted": 0,
        "dropped": [],
        "induction_mode": induction_mode,
    }
    if not getattr(client, "available", False):
        return [], empty_summary
    rows = load_result_rows(results_csv)
    meta = dataset_meta_map(dataset_names)
    lessons: list[Lesson] = []
    raw_records: list[dict[str, Any]] = []
    if llm_raw_dir is not None:
        llm_raw_dir.mkdir(parents=True, exist_ok=True)

    if induction_mode == "paper":
        groups = iter_contrast_strategy_groups(
            rows,
            dataset_names,
            threshold_quantile,
            max_total_groups=max_total_groups,
        )
        print(
            f"[induce-llm] mode=paper selected {len(groups)} strategy contrast(s) "
            f"(max_total={max_total_groups})"
        )
        max_strategy = 1
        max_stage = max(6, int(max_lessons_per_group))
        for toolchain, pos_rows, neg_rows in groups:
            stages = stages_dict(toolchain)
            pos_cases = [_compact_meta_with_result(meta[r["dataset"]], r) for r in pos_rows]
            neg_cases = [_compact_meta_with_result(meta[r["dataset"]], r) for r in neg_rows]
            contrast = _contrast_summary(
                [meta[r["dataset"]] for r in pos_rows],
                [meta[r["dataset"]] for r in neg_rows],
            )
            prompt = lesson_induction_paper_prompt(
                toolchain,
                stages,
                pos_cases,
                neg_cases,
                contrast_summary=contrast,
                max_strategy_lessons=max_strategy,
                max_stage_lessons=max_stage,
            )
            stem = f"{toolchain}_strategy"
            record: dict[str, Any] = {
                "stem": stem,
                "toolchain": toolchain,
                "positive_count": len(pos_cases),
                "negative_count": len(neg_cases),
                "parsed_lessons": [],
                "error": None,
            }
            if llm_raw_dir is not None:
                (llm_raw_dir / f"{stem}_prompt.txt").write_text(prompt)
            try:
                response = client.complete_json(prompt)
            except Exception as exc:  # pragma: no cover
                record["error"] = str(exc)
                print(f"[induce-llm] {stem} request failed: {exc}")
                raw_records.append(record)
                if llm_raw_dir is not None:
                    (llm_raw_dir / f"{stem}_response.json").write_text(json.dumps(record, indent=2))
                continue
            record["response"] = response
            if not response:
                raw_records.append(record)
                if llm_raw_dir is not None:
                    (llm_raw_dir / f"{stem}_response.json").write_text(json.dumps(record, indent=2))
                continue
            pos_metas = [meta[r["dataset"]] for r in pos_rows]
            neg_metas = [meta[r["dataset"]] for r in neg_rows]
            batch = _lessons_from_paper_response(
                response,
                toolchain,
                pos_rows,
                max_strategy_lessons=max_strategy,
                max_stage_lessons=max_stage,
                contrast_summary=contrast,
                positive_metas=pos_metas,
                negative_metas=neg_metas,
            )
            record["parsed_lessons"] = [l.to_dict() for l in batch]
            lessons.extend(batch)
            raw_records.append(record)
            if llm_raw_dir is not None:
                (llm_raw_dir / f"{stem}_response.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False)
                )
    else:
        groups = iter_contrast_groups(
            rows,
            dataset_names,
            threshold_quantile,
            max_total_groups=max_total_groups,
            max_groups_per_toolchain=max_groups_per_toolchain,
        )
        print(
            f"[induce-llm] mode=legacy selected {len(groups)} contrast group(s) by pred_len "
            f"(max_total={max_total_groups}, per_toolchain={max_groups_per_toolchain})"
        )
        for toolchain, pred_len, pos_rows, neg_rows in groups:
            pos_cases = [_compact_meta_with_result(meta[r["dataset"]], r) for r in pos_rows]
            neg_cases = [_compact_meta_with_result(meta[r["dataset"]], r) for r in neg_rows]
            contrast = _contrast_summary(
                [meta[r["dataset"]] for r in pos_rows],
                [meta[r["dataset"]] for r in neg_rows],
            )
            prompt = lesson_induction_group_prompt(
                toolchain,
                stages_dict(toolchain),
                pred_len,
                pos_cases,
                neg_cases,
                contrast_summary=contrast,
                max_lessons=max_lessons_per_group,
            )
            stem = f"{toolchain}_pl{pred_len}_group"
            record = {
                "stem": stem,
                "toolchain": toolchain,
                "pred_len": pred_len,
                "positive_count": len(pos_cases),
                "negative_count": len(neg_cases),
                "parsed_lessons": [],
                "error": None,
            }
            if llm_raw_dir is not None:
                (llm_raw_dir / f"{stem}_prompt.txt").write_text(prompt)
            try:
                response = client.complete_json(prompt)
            except Exception as exc:  # pragma: no cover
                record["error"] = str(exc)
                print(f"[induce-llm] {stem} request failed: {exc}")
                raw_records.append(record)
                if llm_raw_dir is not None:
                    (llm_raw_dir / f"{stem}_response.json").write_text(json.dumps(record, indent=2))
                continue
            record["response"] = response
            if not response:
                raw_records.append(record)
                if llm_raw_dir is not None:
                    (llm_raw_dir / f"{stem}_response.json").write_text(json.dumps(record, indent=2))
                continue
            parsed = response.get("lessons") or []
            record["parsed_lessons"] = parsed
            for idx, raw in enumerate(parsed[:max_lessons_per_group]):
                lesson = _lesson_from_llm(raw, toolchain, pred_len, idx, len(lessons))
                if lesson is not None:
                    lessons.append(lesson)
            raw_records.append(record)
            if llm_raw_dir is not None:
                (llm_raw_dir / f"{stem}_response.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False)
                )

    deduped, drop_report = dedupe_lessons_with_report(lessons)
    summary = {
        "requests": len(raw_records),
        "raw_lessons": len(lessons),
        "accepted": len(deduped),
        "dropped": drop_report,
        "records": raw_records,
        "induction_mode": induction_mode,
        "group_budget": {
            "selected_groups": len(raw_records),
            "max_total_groups": max_total_groups,
            "max_groups_per_toolchain": max_groups_per_toolchain,
            "max_lessons_per_group": max_lessons_per_group,
        },
    }
    if llm_raw_dir is not None:
        (llm_raw_dir / "all_raw_lessons.json").write_text(
            json.dumps([l.to_dict() for l in lessons], indent=2, ensure_ascii=False)
        )
        (llm_raw_dir / "dedupe_report.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
    return deduped, summary


def dedupe_lessons_with_report(lessons: list[Lesson]) -> tuple[list[Lesson], list[dict[str, Any]]]:
    """Keep one lesson per (evidence toolchain, stage tool); prefer higher confidence."""
    kept: dict[tuple[str, str, str], Lesson] = {}
    order: list[tuple[str, str, str]] = []
    dropped: list[dict[str, Any]] = []

    def _dedupe_key(lesson: Lesson) -> tuple[str, str, str]:
        tool = lesson.tool or ("__strategy__" if lesson.task_category == "Strategy" else "")
        return (lesson.toolchain, lesson.task_category, tool)

    for lesson in lessons:
        key = _dedupe_key(lesson)
        if key in kept:
            prev = kept[key]
            if lesson.confidence > prev.confidence:
                dropped.append({
                    "lesson_id": prev.lesson_id,
                    "reason": "duplicate_toolchain_stage_tool",
                    "replaced_by": lesson.lesson_id,
                    "toolchain": lesson.toolchain,
                    "tool": prev.tool,
                    "phenomenon_preview": (prev.phenomenon or "")[:120],
                })
                kept[key] = lesson
            else:
                dropped.append({
                    "lesson_id": lesson.lesson_id,
                    "reason": "duplicate_toolchain_stage_tool",
                    "replaced_by": kept[key].lesson_id,
                    "toolchain": lesson.toolchain,
                    "tool": lesson.tool,
                    "phenomenon_preview": (lesson.phenomenon or "")[:120],
                })
        else:
            order.append(key)
            kept[key] = lesson
    return [kept[k] for k in order], dropped


PlanFn = Callable[[dict[str, Any], list[Lesson]], "str | None"]


def _default_plan_fn(meta: dict[str, Any], lessons: list[Lesson]) -> str:
    return plan_toolchain(meta, lessons=lessons, skip_meta_filter=True).toolchain.name


def _scoped_verify_candidates(target: Lesson, candidates: list[Lesson]) -> list[Lesson]:
    """Lessons sharing the candidate toolchain (local Alg. 3 library)."""
    toolchain = target.toolchain
    return [lesson for lesson in candidates if lesson.toolchain == toolchain]


def _verify_lesson_pools(
    lesson: Lesson,
    background: list[Lesson],
    verify_mode: str,
    *,
    all_candidates: list[Lesson] | None = None,
    pool_scope: str = "all",
) -> tuple[list[Lesson], list[Lesson]]:
    """Build treat/control lesson pools for paired verification.

    paper / original (Alg. 3): plan with library L vs L \\ {phi_k}.
    pool_scope=all uses all candidates; toolchain restricts L to the candidate's toolchain.
    focused (legacy): only the candidate lesson vs empty pool.
    """
    if verify_mode in ("paper", "original"):
        candidates = list(all_candidates or [])
        if pool_scope == "toolchain":
            candidates = _scoped_verify_candidates(lesson, candidates)
        full_l = merge_lesson_collections(background, candidates)
        pool_with = list(full_l)
        pool_without = [l for l in full_l if l.lesson_id != lesson.lesson_id]
        return pool_with, pool_without
    return [lesson], []


def _lessons_for_verify_episode(
    pool: list[Lesson],
    meta: dict[str, Any],
    pred_len: int,
    *,
    force_include: Lesson | None = None,
    top_k: int = 8,
) -> list[Lesson]:
    """Strict meta-matched top-k for verify planning; optionally force phi_k into treat pool."""
    if top_k <= 0:
        return list(pool)
    from .lessons import lessons_for_paper_planning

    meta_pl = dict(meta)
    meta_pl["pred_len"] = int(pred_len)
    if force_include is not None:
        scoped = [lesson for lesson in pool if lesson.toolchain == force_include.toolchain]
        sibling_pool, _ = lessons_for_paper_planning(
            scoped,
            meta_pl,
            pred_len,
            top_k=max(0, top_k - 1),
        )
        inject = [force_include] + [
            lesson for lesson in sibling_pool if lesson.lesson_id != force_include.lesson_id
        ]
        return inject[:top_k] if top_k > 0 else inject
    llm_pool, _ = lessons_for_paper_planning(pool, meta_pl, pred_len, top_k=top_k)
    return llm_pool


def _plan_toolchain_for_verify(
    planner: PlanFn,
    meta: dict[str, Any],
    lesson_pool: list[Lesson],
) -> str | None:
    """Use rule planner with skip_meta_filter (LLM verify planners already set skip in closure)."""
    if planner is _default_plan_fn:
        return plan_toolchain(meta, lessons=lesson_pool, skip_meta_filter=True).toolchain.name
    return planner(meta, lesson_pool)


def _plan_verify_episode(
    planner: PlanFn,
    meta: dict[str, Any],
    pred_len: int,
    pool: list[Lesson],
    *,
    force_include: Lesson | None = None,
    top_k: int = 8,
) -> str | None:
    meta_pl = meta_for_horizon(meta, pred_len)
    inject = _lessons_for_verify_episode(
        pool,
        meta_pl,
        pred_len,
        force_include=force_include,
        top_k=top_k,
    )
    return _plan_toolchain_for_verify(planner, meta_pl, inject)


def _resolve_episode_success(
    success_metric: str,
    a_treat: str,
    a_ctrl: str,
    treat_reward: float | None,
    ctrl_reward: float | None,
    pool_rewards: list[float],
    top_alpha: float,
    *,
    skip_identical_plans: bool = True,
) -> tuple[float, float] | None:
    """Episode success (y_treat, y_ctrl). Returns None when the episode is not informative."""
    if skip_identical_plans and a_treat == a_ctrl:
        return None
    if treat_reward is None or ctrl_reward is None:
        return None
    if success_metric == "paired":
        return (
            1.0 if treat_reward > ctrl_reward else 0.0,
            1.0 if ctrl_reward > treat_reward else 0.0,
        )
    y_treat = _success_top_alpha_from_rewards(pool_rewards, treat_reward, top_alpha)
    y_ctrl = _success_top_alpha_from_rewards(pool_rewards, ctrl_reward, top_alpha)
    if success_metric == "top_alpha_paired" and y_treat == y_ctrl:
        return (
            1.0 if treat_reward > ctrl_reward else 0.0,
            1.0 if ctrl_reward > treat_reward else 0.0,
        )
    if success_metric in ("top_alpha", "top_alpha_paired"):
        return y_treat, y_ctrl
    return None


def _success_top_alpha_from_rewards(rewards: list[float], reward: float, top_alpha: float) -> float:
    if not rewards:
        return 0.0
    observed_sorted = sorted(rewards, reverse=True)
    cutoff_idx = max(1, int(math.ceil(len(observed_sorted) * top_alpha)))
    threshold = observed_sorted[cutoff_idx - 1]
    return 1.0 if reward >= threshold else 0.0


def _select_verify_episodes(
    lesson: Lesson,
    meta: dict[str, dict[str, Any]],
    eligible: list[str],
    pred_lens: list[int],
    *,
    target_datasets: int = 4,
) -> list[tuple[str, int]]:
    """Up to ``target_datasets`` pairs where all activation conditions (≤2 keys + pred_len) match."""
    active_horizons = [
        int(pl)
        for pl in pred_lens
        if any(lesson_matches(meta[d], lesson, int(pl)) for d in eligible if d in meta)
    ]
    if not active_horizons:
        return []
    pred_len = int(lesson.evidence_pred_len) if lesson.evidence_pred_len is not None else active_horizons[0]
    if pred_len not in active_horizons:
        pred_len = active_horizons[0]
    scored: list[tuple[str, ...]] = []
    for dataset in eligible:
        if dataset not in meta:
            continue
        if not lesson_matches(meta[dataset], lesson, pred_len):
            continue
        scored.append((dataset,))
    scored.sort(key=lambda item: item[0])
    return [(dataset, pred_len) for (dataset,) in scored[:target_datasets]]


def verify_lessons_real_paired_rollout(
    candidate_lessons: list[Lesson],
    dataset_names: list[str],
    pred_lens: list[int],
    reference_results_csv: Path,
    candidate_pool: list[Lesson] | None = None,
    top_alpha: float = 0.35,
    min_effect: float = 0.0,
    min_verify_support: int = 2,
    target_verify_datasets: int = 4,
    seq_len: int = 96,
    use_gpu: bool = True,
    gpu: int = 0,
    quick: bool = False,
    python_bin: str | None = None,
    timeout: int | None = None,
    cache_path: Path | None = None,
    plan_fn: PlanFn | None = None,
    *,
    verify_mode: str = "paper",
    success_metric: str = "top_alpha",
    all_candidates: list[Lesson] | None = None,
    verify_planning_top_k: int = 8,
    verify_pool_scope: str = "all",
    skip_identical_plans: bool = True,
) -> list[dict[str, Any]]:
    """Paper Stage 3 (Alg. 3) with **real** paired episodes.

    For each candidate lesson and each eligible (dataset, pred_len), the planner picks a_treat
    (with phi_k) and a_ctrl (without phi_k for positive lessons; Contradict(phi_k) for negative
    lessons). Each strategy is then *actually* executed via run_forecast; rewards are persisted
    to ``cache_path`` so repeated invocations are cheap. Success y is computed against the
    top-alpha cutoff among ALL observed rewards on the same (dataset, pred_len) key (reference
    CSV ∪ freshly executed runs).
    """
    rows = load_result_rows(reference_results_csv) if Path(reference_results_csv).exists() else []
    rows_by_key = _group_rows(rows) if rows else {}
    meta = dataset_meta_map(dataset_names)

    background: list[Lesson] = list(candidate_pool) if candidate_pool is not None else []

    cache: dict[tuple[str, int, str], float] = {}
    if cache_path and Path(cache_path).exists():
        try:
            raw = json.loads(Path(cache_path).read_text())
            for k, v in raw.items():
                ds, pl, tc = k.split("|", 2)
                cache[(ds, int(pl), tc)] = float(v)
        except (json.JSONDecodeError, ValueError):
            pass

    def _persist_cache() -> None:
        if cache_path is None:
            return
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(
            {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in cache.items()},
            indent=2,
        ))

    def reward_for(dataset: str, pred_len: int, toolchain: str) -> float | None:
        key = (dataset, int(pred_len))
        for row in rows_by_key.get(key, []):
            if row["toolchain"] == toolchain:
                return float(row["reward"])
        ck = (dataset, int(pred_len), toolchain)
        if ck in cache:
            return cache[ck]
        try:
            run = run_forecast(
                get_dataset(dataset),
                get_toolchain(toolchain),
                pred_len=pred_len,
                seq_len=seq_len,
                python_bin=python_bin,
                use_gpu=use_gpu,
                gpu=gpu,
                quick=quick,
                timeout=timeout,
            )
        except Exception as exc:
            print(f"[verify-execute] forecast failed dataset={dataset} pl={pred_len} tc={toolchain}: {exc}")
            return None
        if run.mse is None or run.mae is None:
            return None
        reward = reward_from_metrics(run.mse, run.mae)
        cache[ck] = reward
        rows_by_key.setdefault(key, []).append({
            "dataset": dataset,
            "pred_len": int(pred_len),
            "toolchain": toolchain,
            "model": toolchain,
            "mse": float(run.mse),
            "mae": float(run.mae),
            "reward": reward,
        })
        _persist_cache()
        return reward

    def success_y(dataset: str, pred_len: int, toolchain: str) -> float | None:
        reward = reward_for(dataset, pred_len, toolchain)
        if reward is None:
            return None
        observed = [row["reward"] for row in rows_by_key.get((dataset, int(pred_len)), [])]
        if not observed:
            return None
        observed_sorted = sorted(observed, reverse=True)
        cutoff_idx = max(1, int(math.ceil(len(observed_sorted) * top_alpha)))
        threshold = observed_sorted[cutoff_idx - 1]
        return 1.0 if reward >= threshold else 0.0

    planner = plan_fn or _default_plan_fn
    verified: list[dict[str, Any]] = []
    skipped_no_match = 0
    skipped_no_episodes = 0
    for lesson in candidate_lessons:
        eligible = [d for d in dataset_names if d in meta]
        episode_keys = _select_verify_episodes(
            lesson,
            meta,
            eligible,
            pred_lens,
            target_datasets=target_verify_datasets,
        )
        if not episode_keys:
            skipped_no_match += 1
            continue
        pool_with, pool_without = _verify_lesson_pools(
            lesson,
            background,
            verify_mode,
            all_candidates=all_candidates or candidate_lessons,
            pool_scope=verify_pool_scope,
        )
        y_treat: list[float] = []
        y_ctrl: list[float] = []
        episodes: list[dict[str, Any]] = []
        skipped_identical = 0
        print(
            f"[verify-execute] lesson={lesson.lesson_id} "
            f"evidence_pl={lesson.evidence_pred_len} episodes={episode_keys} "
            f"verify_mode={verify_mode} metric={success_metric} "
            f"plan_top_k={verify_planning_top_k} pool_scope={verify_pool_scope}"
        )
        for dataset, pred_len in episode_keys:
            meta_pl = meta[dataset]
            if lesson.lesson_type == "negative":
                a_treat = _plan_verify_episode(
                    planner,
                    meta_pl,
                    pred_len,
                    pool_with,
                    force_include=lesson,
                    top_k=verify_planning_top_k,
                )
                a_ctrl = _contradict_toolchain(lesson)
            else:
                a_treat = _plan_verify_episode(
                    planner,
                    meta_pl,
                    pred_len,
                    pool_with,
                    force_include=lesson,
                    top_k=verify_planning_top_k,
                )
                a_ctrl = _plan_verify_episode(
                    planner,
                    meta_pl,
                    pred_len,
                    pool_without,
                    top_k=verify_planning_top_k,
                )
            if a_treat is None or a_ctrl is None:
                continue
            print(
                f"[verify-execute] rollout dataset={dataset} pl={pred_len} "
                f"treat={a_treat} ctrl={a_ctrl}"
            )
            rt = reward_for(dataset, pred_len, a_treat)
            rc = reward_for(dataset, pred_len, a_ctrl)
            pool_rewards = [
                float(row["reward"])
                for row in rows_by_key.get((dataset, int(pred_len)), [])
            ]
            resolved = _resolve_episode_success(
                success_metric,
                a_treat,
                a_ctrl,
                rt,
                rc,
                pool_rewards,
                top_alpha,
                skip_identical_plans=skip_identical_plans,
            )
            if resolved is None:
                if skip_identical_plans and a_treat == a_ctrl:
                    skipped_identical += 1
                continue
            y_t, y_c = resolved
            y_treat.append(y_t)
            y_ctrl.append(y_c)
            episodes.append({
                "dataset": dataset,
                "pred_len": int(pred_len),
                "a_treat": a_treat,
                "a_ctrl": a_ctrl,
                "y_treat": y_t,
                "y_ctrl": y_c,
            })
        if not y_treat:
            skipped_no_episodes += 1
            if skipped_identical:
                print(
                    f"[verify] skip lesson={lesson.lesson_id}: no informative episodes "
                    f"(identical_plans={skipped_identical})"
                )
            continue
        effect = float(np.mean(y_treat) - np.mean(y_ctrl))
        pass_support = min(min_verify_support, len(y_treat))
        passed = effect > min_effect and len(y_treat) >= pass_support
        item = lesson.to_dict()
        item.update({
            "verification_mode": "paired_real_rollout",
            "verify_mode": verify_mode,
            "success_metric": success_metric,
            "verification_effect": effect,
            "p_treat": float(np.mean(y_treat)),
            "p_ctrl": float(np.mean(y_ctrl)),
            "verification_support": len(y_treat),
            "verification_episodes": episodes,
            "verified": passed,
            "control_kind": (
                "contradict"
                if lesson.lesson_type == "negative"
                else ("empty_pool" if verify_mode == "focused" else "without_lesson")
            ),
        })
        status = "PASS" if item["verified"] else "fail"
        print(
            f"[verify] {status} lesson={lesson.lesson_id} effect={effect:.3f} "
            f"support={len(y_treat)} mode={verify_mode} metric={success_metric} (real rollout)"
        )
        if item["verified"]:
            verified.append(item)
    print(
        f"[verify] summary mode=paired_real_rollout verify={verify_mode} metric={success_metric} "
        f"candidates={len(candidate_lessons)} verified={len(verified)} "
        f"skipped_no_activation_match={skipped_no_match} skipped_no_episodes={skipped_no_episodes}"
    )
    return verified


def _score_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {"dataset": row["dataset"], "pred_len": row["pred_len"], "mse": row["mse"], "mae": row["mae"], "reward": row["reward"]}


def _mode(values: list[Any]) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], int(row["pred_len"])), []).append(row)
    return grouped


def _compact_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: meta.get(k) for k in _COMPACT_META_KEYS if k in meta}


def _compact_meta_with_result(meta: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    item = _compact_meta(meta)
    item.update(_score_dict(row))
    return item


def _contrast_summary(pos_metas: list[dict[str, Any]], neg_metas: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-digest the contrast between positive and negative meta groups.

    Categorical keys report mode for each side (omitted if identical). Numeric keys report median
    on each side plus the relative gap so the LLM does not have to scan raw 25-field dicts.
    """
    categorical_diffs: dict[str, dict[str, Any]] = {}
    for key in _CATEGORICAL_KEYS:
        pos_vals = [m.get(key) for m in pos_metas if m.get(key) is not None]
        neg_vals = [m.get(key) for m in neg_metas if m.get(key) is not None]
        pos_mode = _mode(pos_vals)
        neg_mode = _mode(neg_vals)
        if pos_mode is None and neg_mode is None:
            continue
        if pos_mode == neg_mode:
            continue
        categorical_diffs[key] = {"positive_mode": pos_mode, "negative_mode": neg_mode}

    numeric_diffs: dict[str, dict[str, Any]] = {}
    for key in _NUMERIC_KEYS:
        pos_vals = [float(m[key]) for m in pos_metas if m.get(key) is not None]
        neg_vals = [float(m[key]) for m in neg_metas if m.get(key) is not None]
        if not pos_vals or not neg_vals:
            continue
        pos_med = float(np.median(pos_vals))
        neg_med = float(np.median(neg_vals))
        denom = max(abs(pos_med), abs(neg_med), 1e-8)
        rel_gap = (pos_med - neg_med) / denom
        if abs(rel_gap) < 0.1:
            continue
        numeric_diffs[key] = {
            "positive_median": round(pos_med, 6),
            "negative_median": round(neg_med, 6),
            "relative_gap": round(rel_gap, 3),
        }

    return {
        "positive_count": len(pos_metas),
        "negative_count": len(neg_metas),
        "categorical_differences": categorical_diffs,
        "numeric_differences": numeric_diffs,
    }


def _lesson_toolchain_names(lesson: Lesson) -> set[str]:
    if lesson.task_category == "Strategy":
        return {lesson.toolchain}
    if lesson.tool:
        names = toolchains_matching_tool(lesson.tool, lesson.task_category)
        if names:
            return set(names)
    return {lesson.toolchain}


def _contradict_toolchain(lesson: Lesson) -> str:
    names = sorted(_lesson_toolchain_names(lesson))
    if lesson.toolchain in names:
        return lesson.toolchain
    return names[0] if names else lesson.toolchain


def _scope_paper_lesson_id(
    toolchain: str,
    raw_id: str,
    tool: str,
    task_category: str,
    lesson_idx: int,
) -> str:
    """Ensure lesson ids are toolchain-scoped so TimeXer lessons are not overwritten globally."""
    if raw_id and raw_id.startswith(f"{toolchain}_"):
        return raw_id
    if task_category == "Strategy":
        return f"{toolchain}_strategy_L{lesson_idx + 1}"
    safe_tool = (tool or "stage").replace(" ", "_")
    return f"{toolchain}_{safe_tool}_L{lesson_idx + 1}"


def _contrast_supports_activation(
    cond: dict[str, Any],
    contrast: dict[str, Any] | None,
) -> bool:
    """Require at least one activation key to differ between pos/neg when contrast is available."""
    if not contrast:
        return True
    cat = contrast.get("categorical_differences") or {}
    num = contrast.get("numeric_differences") or {}
    if not cat and not num:
        return True
    keys = [k for k in cond if k != "pred_len"]
    if not keys:
        return True
    return any(k in cat or k in num for k in keys)


def _ensure_toolchain_phenomenon(
    phenomenon: str,
    toolchain: str,
    tool: str,
    task_category: str,
) -> str:
    """Forecasting lessons must cite evidence toolchain so model choice is not lost."""
    text = (phenomenon or "").strip()
    if task_category != "Forecasting" or not tool:
        return text
    lower = text.lower()
    if toolchain.lower() in lower:
        return text
    prefix = f"[Phenomenon]: On evidence strategy {toolchain}, {tool} forecasting "
    if text.startswith("[Phenomenon]:"):
        body = text.split(":", 1)[1].strip()
        if body.lower().startswith(tool.lower()):
            return f"{prefix}{body[len(tool):].lstrip(' forecasting').lstrip()}"
        return f"{prefix}{body}"
    return f"{prefix}{text}" if text else f"{prefix}tends to succeed under the matched meta-profile."


def _activation_from_paper_llm(
    raw: dict[str, Any],
    pos_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], int | None]:
    cond = cap_activation_conditions(dict(raw.get("activation_conditions") or {}), max_keys=2)
    cond = normalize_activation_conditions(cond)
    pls = sorted({int(r["pred_len"]) for r in pos_rows})
    raw_pl = raw.get("pred_len") or cond.get("pred_len")
    if raw_pl is not None and "pred_len" not in cond:
        cond["pred_len"] = raw_pl if isinstance(raw_pl, list) else [int(raw_pl)]
    elif "pred_len" not in cond and len(pls) == 1:
        cond["pred_len"] = pls
    elif "pred_len" not in cond and 1 < len(pls) <= 4:
        cond["pred_len"] = pls
    evidence_pl = pls[0] if len(pls) == 1 else None
    return cond, evidence_pl


def _lesson_from_llm_paper(
    raw: dict[str, Any],
    toolchain: str,
    pos_rows: list[dict[str, Any]],
    lesson_idx: int,
    *,
    contrast_summary: dict[str, Any] | None = None,
    positive_metas: list[dict[str, Any]] | None = None,
    negative_metas: list[dict[str, Any]] | None = None,
) -> Lesson | None:
    stages = stages_dict(toolchain)
    task_category = str(raw.get("task_category", ""))
    lesson_level = str(raw.get("lesson_level", "stage"))
    tool = normalize_tool_name(
        str(raw.get("tool") or raw.get("tool_name") or raw.get("tool_or_toolchain") or "")
    )
    if task_category == "Strategy" or lesson_level == "strategy":
        task_category = "Strategy"
        if not tool:
            tool = normalize_tool_name(stages.get("forecasting", ""))
    elif not tool:
        task_category = task_category or "Imputation"
        tool = tool_at_stage(get_toolchain(toolchain), task_category)
    if not is_valid_lesson_tool(tool, task_category, stages):
        print(
            f"[induce-llm] skip invalid paper lesson tool={tool!r} "
            f"task_category={task_category!r} toolchain={toolchain!r}"
        )
        return None
    cond, evidence_pl = _activation_from_paper_llm(raw, pos_rows)
    if task_category == "Strategy":
        from .lessons import derive_strategy_activation_from_contrast

        cond = derive_strategy_activation_from_contrast(
            positive_metas or [],
            negative_metas or [],
            llm_fallback=cond,
        )
    if not _contrast_supports_activation(cond, contrast_summary):
        print(
            f"[induce-llm] skip paper lesson: activation keys {list(cond)} "
            f"not supported by contrast on toolchain={toolchain!r}"
        )
        return None
    raw_id = str(raw.get("lesson_id") or "")
    lesson_id = _scope_paper_lesson_id(toolchain, raw_id, tool, task_category, lesson_idx)
    evidence = str(raw.get("evidence_toolchain") or raw.get("toolchain") or toolchain)
    phenomenon = _ensure_toolchain_phenomenon(
        str(raw.get("phenomenon", "")),
        evidence,
        tool,
        task_category,
    )
    tags = list(raw.get("tags") or [])
    if "llm_induced" not in tags:
        tags.append("llm_induced")
    tags.append("paper_strategy_contrast")
    if task_category == "Strategy":
        tags.append("strategy_level")
    else:
        tags.append("stage_level")
    return Lesson(
        lesson_id=lesson_id,
        lesson_type=str(raw.get("lesson_type", "positive")),
        toolchain=evidence,
        task_category=task_category,
        activation_conditions=cond,
        phenomenon=phenomenon,
        analysis=str(raw.get("analysis", "")),
        confidence=float(raw.get("confidence", 0.5)),
        tags=tags,
        tool=tool if task_category != "Strategy" else (tool or ""),
        evidence_pred_len=evidence_pl,
    )


def _lessons_from_paper_response(
    response: dict[str, Any],
    toolchain: str,
    pos_rows: list[dict[str, Any]],
    *,
    max_strategy_lessons: int,
    max_stage_lessons: int,
    contrast_summary: dict[str, Any] | None = None,
    positive_metas: list[dict[str, Any]] | None = None,
    negative_metas: list[dict[str, Any]] | None = None,
) -> list[Lesson]:
    out: list[Lesson] = []
    idx = 0
    for raw in (response.get("strategy_lessons") or [])[:max_strategy_lessons]:
        if not isinstance(raw, dict):
            continue
        raw = dict(raw)
        raw.setdefault("lesson_level", "strategy")
        raw.setdefault("task_category", "Strategy")
        lesson = _lesson_from_llm_paper(
            raw, toolchain, pos_rows, idx, contrast_summary=contrast_summary,
            positive_metas=positive_metas, negative_metas=negative_metas,
        )
        if lesson is not None:
            out.append(lesson)
            idx += 1
    for raw in (response.get("stage_lessons") or response.get("lessons") or [])[:max_stage_lessons]:
        if not isinstance(raw, dict):
            continue
        lesson = _lesson_from_llm_paper(
            raw, toolchain, pos_rows, idx, contrast_summary=contrast_summary,
            positive_metas=positive_metas, negative_metas=negative_metas,
        )
        if lesson is not None:
            out.append(lesson)
            idx += 1
    return out


def _lesson_from_llm(
    raw: dict[str, Any],
    toolchain: str,
    pred_len: int,
    lesson_idx: int,
    offset: int,
) -> Lesson | None:
    tool = normalize_tool_name(
        str(raw.get("tool") or raw.get("tool_name") or raw.get("tool_or_toolchain") or "")
    )
    task_category = str(raw.get("task_category", ""))
    if not tool:
        spec = get_toolchain(toolchain)
        task_category = task_category or "Imputation"
        tool = tool_at_stage(spec, task_category)
    if not is_preprocessing_lesson(tool, task_category):
        print(
            f"[induce-llm] skip non-preprocessing lesson tool={tool!r} "
            f"task_category={task_category!r}"
        )
        return None
    lesson_id = str(
        raw.get("lesson_id")
        or f"llm_{toolchain}_pl{pred_len}_L{lesson_idx + 1}"
    )
    evidence = str(raw.get("evidence_toolchain") or raw.get("toolchain") or toolchain)
    return Lesson(
        lesson_id=lesson_id,
        lesson_type=str(raw.get("lesson_type", "positive")),
        toolchain=evidence,
        task_category=task_category,
        activation_conditions=bind_pred_len_condition(
            cap_activation_conditions(dict(raw.get("activation_conditions") or {}), max_keys=2),
            pred_len,
        ),
        phenomenon=str(raw.get("phenomenon", "")),
        analysis=str(raw.get("analysis", "")),
        confidence=float(raw.get("confidence", 0.5)),
        tags=list(raw.get("tags") or ["llm_induced", "group_by_pred_len", f"pl{pred_len}"]),
        tool=tool,
        evidence_pred_len=int(pred_len),
    )
