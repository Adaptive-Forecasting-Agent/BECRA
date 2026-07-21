from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TOOLCHAINS, ToolchainSpec, tool_library_as_dict
from .lessons import (
    Lesson,
    SEED_VERIFIED_LESSONS,
    condition_matches,
    effective_activation_conditions,
    lesson_matches,
    lesson_matches_for_planning,
)
from .prompt_templates import lesson_guided_planning_prompt
from .toolchain_registry import planning_stage_tool_library, resolve_toolchain_from_stages
from .tools import toolchains_matching_tool


@dataclass
class PlanningResult:
    toolchain: ToolchainSpec
    score: float
    matched_lessons: list[Lesson]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "toolchain": self.toolchain.name,
            "score": self.score,
            "matched_lessons": [lesson.lesson_id for lesson in self.matched_lessons],
            "reason": self.reason,
        }


def _filter_lessons_for_planning(
    lesson_pool: list[Lesson],
    meta: dict[str, Any],
    *,
    for_verify: bool = False,  # noqa: ARG001 — kept for API compatibility
    relaxed: bool = False,
) -> list[Lesson]:
    pred_len = meta.get("pred_len")
    if pred_len is None:
        return [
            lesson
            for lesson in lesson_pool
            if condition_matches(meta, effective_activation_conditions(lesson))
        ]
    if relaxed:
        matcher = lambda m, les, pl: lesson_matches_for_planning(m, les, pl, min_meta_overlap=1)
    else:
        matcher = lesson_matches
    return [
        lesson
        for lesson in lesson_pool
        if matcher(meta, lesson, int(pred_len))
    ]


def _resolve_planned_toolchain(response: dict[str, Any]) -> str | None:
    stages = response.get("stages")
    if isinstance(stages, dict) and stages:
        return resolve_toolchain_from_stages(stages)
    strategy = response.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        return None
    if strategy in TOOLCHAINS:
        return strategy
    from .toolchain_registry import register_toolchain_by_name

    spec = register_toolchain_by_name(strategy)
    return spec.name if spec is not None else None


def paper_planning_context() -> dict[str, Any]:
    """Appendix Alg. 4: encode available toolchains A and executable stage tools."""
    return {
        "stage_tools": planning_stage_tool_library(),
        "paper_tool_library": tool_library_as_dict(),
    }


def _lesson_vote_targets(lesson: Lesson, *, toolchain_scope: bool = False) -> list[str]:
    """Resolve vote targets for a lesson (verify uses toolchain_scope=True)."""
    if toolchain_scope and lesson.toolchain:
        return [lesson.toolchain]
    if lesson.tool:
        return toolchains_matching_tool(lesson.tool, lesson.task_category)
    return [lesson.toolchain]


def _score_toolchain_by_lessons(
    active_lessons: list[Lesson],
    candidate_names: set[str] | list[str],
    *,
    toolchain_scope: bool = False,
) -> dict[str, tuple[float, list[Lesson]]]:
    """Lesson-vote scoring only (paper deployment fallback; no meta heuristics)."""
    names = [n for n in candidate_names if n in TOOLCHAINS]
    if not names:
        return {}
    scores = {name: 0.0 for name in names}
    matched: dict[str, list[Lesson]] = {name: [] for name in names}
    for lesson in active_lessons:
        direction = 1.0 if lesson.lesson_type == "positive" else -0.7
        delta = direction * lesson.confidence
        for name in _lesson_vote_targets(lesson, toolchain_scope=toolchain_scope):
            if name not in scores:
                continue
            scores[name] += delta
            matched[name].append(lesson)
    return {name: (scores[name], matched[name]) for name in names}


def _score_toolchain_candidates(
    meta: dict[str, Any],
    active_lessons: list[Lesson],
    candidate_names: set[str] | list[str],
) -> dict[str, tuple[float, list[Lesson]]]:
    """Lesson votes + optional meta heuristics (legacy champion mode only)."""
    names = [n for n in candidate_names if n in TOOLCHAINS]
    if not names:
        return {}
    scores = {name: 0.0 for name in names}
    matched: dict[str, list[Lesson]] = {name: [] for name in names}
    for lesson in active_lessons:
        direction = 1.0 if lesson.lesson_type == "positive" else -0.7
        delta = direction * lesson.confidence
        targets = (
            toolchains_matching_tool(lesson.tool, lesson.task_category)
            if lesson.tool
            else [lesson.toolchain]
        )
        for name in targets:
            if name not in scores:
                continue
            scores[name] += delta
            matched[name].append(lesson)
    full = {n: 0.0 for n in TOOLCHAINS}
    _add_heuristic_scores(meta, full)
    for name in names:
        scores[name] += full.get(name, 0.0)
    return {name: (scores[name], matched[name]) for name in names}


def _pick_by_lesson_votes(
    meta: dict[str, Any],
    active_lessons: list[Lesson],
    candidate_names: set[str] | list[str] | None = None,
    *,
    note: str,
    toolchain_scope: bool = False,
) -> PlanningResult:
    names = set(candidate_names) if candidate_names is not None else set(TOOLCHAINS.keys())
    scored = _score_toolchain_by_lessons(
        active_lessons,
        names,
        toolchain_scope=toolchain_scope,
    )
    if not scored:
        raise ValueError("No valid toolchain candidates for lesson-vote fallback.")
    from .tools import evidence_toolchain_preprocessing_depth

    selected_name = max(
        scored,
        key=lambda n: (
            scored[n][0],
            -evidence_toolchain_preprocessing_depth(n),
            n,
        ),
    )
    best_score, best_matched = scored[selected_name]
    selected = TOOLCHAINS[selected_name]
    reason = (
        f"{note}; paper rule fallback score={best_score:.2f}; "
        f"matched_lessons={len(best_matched)}; picked={selected_name}"
    )
    return PlanningResult(selected, best_score, best_matched, reason)


def _candidate_verification_summary(
    matched: list[Lesson],
    candidates: list[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate verified paired-rollout outcomes per candidate over ALL matched lessons.

    Two complementary signals are reported:
    - ``verified_rollout_record``: episodes where the candidate was the evaluated
      (treatment) strategy.
    - ``won_as_control``: how often the candidate beat another evaluated strategy when
      used as the control arm. Plain baselines often appear as controls; this recovers
      that evidence even when the retrieval budget keeps only one lesson per toolchain.

    Episodes are deduplicated per (dataset, pred_len, treat, ctrl)."""
    cand_set = set(candidates)
    episodes: dict[tuple[str, int, str, str], tuple[float, float]] = {}
    for lesson in matched:
        for ep in lesson.verification_episodes or []:
            key = (
                str(ep.get("dataset")),
                int(ep.get("pred_len") or 0),
                str(ep.get("a_treat")),
                str(ep.get("a_ctrl")),
            )
            episodes[key] = (float(ep.get("y_treat") or 0), float(ep.get("y_ctrl") or 0))
    treat_wins: dict[str, int] = {name: 0 for name in candidates}
    treat_losses: dict[str, int] = {name: 0 for name in candidates}
    ctrl_wins: dict[str, int] = {name: 0 for name in candidates}
    for (_, _, a_treat, a_ctrl), (y_treat, y_ctrl) in episodes.items():
        if a_treat in cand_set:
            if y_treat > y_ctrl:
                treat_wins[a_treat] += 1
            elif y_treat < y_ctrl:
                treat_losses[a_treat] += 1
        if a_ctrl in cand_set and y_ctrl > y_treat:
            ctrl_wins[a_ctrl] += 1
    summary: dict[str, dict[str, Any]] = {}
    for name in candidates:
        entry: dict[str, Any] = {
            "matched_lessons": sum(1 for lesson in matched if lesson.toolchain == name),
        }
        tw, tl = treat_wins[name], treat_losses[name]
        if tw + tl:
            entry["verified_rollout_record"] = f"{tw}W-{tl}L"
        if ctrl_wins[name]:
            entry["won_as_control"] = int(ctrl_wins[name])
        summary[name] = entry
    return summary


def paper_plan_toolchain(
    meta: dict[str, Any],
    lesson_pool: list[Lesson],
    client: Any | None = None,
    *,
    top_k: int = 5,
    llm_retries: int = 3,
    use_llm: bool = True,
    explore_csv: str | Path | None = None,
    rewiden_source_datasets: list[str] | None = None,
    inject_scope: str = "primary",
    plan_samples: int = 3,
) -> PlanningResult:
    """Appendix Alg. 4: retrieve L_x, LLM plans over full tool library A, lesson-only rule fallback."""
    from .lessons import (
        apply_strategy_activation_for_planning,
        lessons_for_paper_planning,
        primary_evidence_toolchain,
    )

    pred_len = int(meta["pred_len"])
    raw_pool = list(lesson_pool)
    pool = raw_pool
    if explore_csv and rewiden_source_datasets:
        pool = apply_strategy_activation_for_planning(
            pool,
            meta,
            pred_len,
            explore_csv,
            rewiden_source_datasets,
        )
    llm_lessons, matched = lessons_for_paper_planning(
        pool,
        meta,
        pred_len,
        top_k=top_k,
        raw_pool=raw_pool,
        explore_csv=explore_csv,
        source_datasets=rewiden_source_datasets,
        inject_scope=inject_scope,
    )
    primary = primary_evidence_toolchain(
        matched,
        raw_pool=raw_pool,
        meta=meta,
        pred_len=pred_len,
        explore_csv=explore_csv,
        source_datasets=rewiden_source_datasets,
    )
    if inject_scope == "matched":
        rule_lessons = list(matched or llm_lessons)
    else:
        rule_lessons = (
            [lesson for lesson in matched if lesson.toolchain == primary]
            if primary
            else list(matched or llm_lessons)
        )
    ctx = paper_planning_context()
    # Paper Alg. 4 prompt has no "primary" concept: in matched scope the LLM weighs all
    # injected lessons itself. The hint is only kept for the legacy primary scope.
    if primary and inject_scope != "matched":
        ctx = {**ctx, "primary_evidence_toolchain": primary}
    ctx = {**ctx, "planning_inject_scope": inject_scope}
    # Expose retrieved evidence patterns without turning them into a closed candidate
    # set. Paper Algorithm 5 plans over the full executable library one stage at a time.
    evidence_toolchains = sorted(
        {lesson.toolchain for lesson in llm_lessons if lesson.toolchain in TOOLCHAINS}
    )
    if inject_scope == "matched" and evidence_toolchains:
        ctx = {
            **ctx,
            "evidence_toolchains": evidence_toolchains,
            "evidence_toolchain_stages": {
                name: dict(TOOLCHAINS[name].stages) for name in evidence_toolchains
            },
            "evidence_toolchain_verification": _candidate_verification_summary(
                matched, evidence_toolchains
            ),
        }
    print(
        f"[paper-plan] pl={pred_len} strict_matched={len(matched)} "
        f"llm_inject={len(llm_lessons)} inject_scope={inject_scope} "
        f"primary_evidence={primary!r}"
    )
    for lesson in llm_lessons:
        print(f"[paper-plan]   inject lesson={lesson.lesson_id} toolchain={lesson.toolchain}")

    if use_llm and client is not None and getattr(client, "available", False):
        # Self-consistency: sample the agent's plan several times and keep the modal
        # resolved toolchain (generic variance reduction; ties broken by first occurrence).
        n_samples = max(1, int(plan_samples)) if inject_scope == "matched" else 1
        sample_picks: list[PlanningResult] = []
        # Keep sampling until the vote quota is met (invalid LLM outputs return None
        # and would otherwise shrink the vote to a single noisy sample).
        max_attempts = 2 * n_samples
        for _ in range(max_attempts):
            if len(sample_picks) >= n_samples:
                break
            pick = llm_plan_toolchain(
                meta,
                llm_lessons,
                client,
                retries=llm_retries,
                skip_meta_filter=True,
                allowed_toolchains=None,
                planning_context=ctx,
            )
            if pick is not None:
                sample_picks.append(pick)
        llm_pick: PlanningResult | None = None
        if sample_picks:
            counts: dict[str, int] = {}
            for pick in sample_picks:
                counts[pick.toolchain.name] = counts.get(pick.toolchain.name, 0) + 1

            def _tiebreak(name: str) -> tuple[int, int]:
                """When votes tie: prefer candidates that often won as control, then
                prefer not adding imputation when missing_rate is exactly zero."""
                verification = (
                    ctx.get("evidence_toolchain_verification", {})
                    if isinstance(ctx, dict)
                    else {}
                )
                ctrl_wins = int((verification.get(name) or {}).get("won_as_control") or 0)
                stages = TOOLCHAINS.get(name).stages if name in TOOLCHAINS else {}
                unjustified_impute = 0
                if (
                    float(meta.get("missing_rate") or 0.0) < 1e-6
                    and stages.get("imputation") not in {None, "none"}
                ):
                    unjustified_impute = 1
                return (ctrl_wins, -unjustified_impute)

            modal_name = max(
                counts,
                key=lambda name: (
                    counts[name],
                    _tiebreak(name),
                    -next(i for i, p in enumerate(sample_picks) if p.toolchain.name == name),
                ),
            )
            llm_pick = next(p for p in sample_picks if p.toolchain.name == modal_name)
            if n_samples > 1:
                print(f"[paper-plan] self-consistency votes: {counts} -> {modal_name}")
        if llm_pick is not None:
            enforce_primary = inject_scope != "matched"
            if enforce_primary and primary and llm_pick.toolchain.name != primary:
                print(
                    f"[paper-plan] LLM resolved {llm_pick.toolchain.name!r} "
                    f"!= primary evidence {primary!r}; using toolchain-scoped rule fallback"
                )
            else:
                if (
                    not enforce_primary
                    and primary
                    and llm_pick.toolchain.name != primary
                ):
                    print(
                        f"[paper-plan] LLM chose {llm_pick.toolchain.name!r} "
                        f"(primary hint was {primary!r}; matched inject scope)"
                    )
                return PlanningResult(
                    llm_pick.toolchain,
                    0.0,
                    llm_pick.matched_lessons or llm_lessons,
                    f"paper_llm_plan strict_matched={len(matched)}; {llm_pick.reason}",
                )
        fallback_note = (
            "matched lesson pool"
            if inject_scope == "matched"
            else "primary evidence"
        )
        print(f"[paper-plan] LLM failed or deviated; falling back to lesson-vote rule on {fallback_note}")

    return _pick_by_lesson_votes(
        meta,
        rule_lessons,
        set(TOOLCHAINS.keys()),
        note="llm_unavailable" if not use_llm else "llm_failed",
        toolchain_scope=True,
    )


def planning_context_for_champions(champion_names: list[str]) -> dict[str, Any]:
    """Stage tool union and per-champion stage patterns for champion-pool-only LLM planning."""
    imputation: set[str] = set()
    anomaly: set[str] = set()
    decomposition: set[str] = set()
    forecasting: set[str] = set()
    stages_by_champion: dict[str, dict[str, str]] = {}
    for name in champion_names:
        if name not in TOOLCHAINS:
            continue
        stages = TOOLCHAINS[name].stages
        stages_by_champion[name] = stages
        imputation.add(stages["imputation"])
        anomaly.add(stages["anomaly_handling"])
        decomposition.add(stages["decomposition"])
        forecasting.add(stages["forecasting"])
    return {
        "champion_toolchains_only": list(champion_names),
        "stages_by_champion": stages_by_champion,
        "stage_tools": {
            "imputation": sorted(imputation),
            "anomaly_handling": sorted(anomaly),
            "transformation": ["none"],
            "decomposition": sorted(decomposition),
            "normalization": ["standard_scaler"],
            "forecasting": sorted(forecasting),
        },
    }


def _pick_from_champion_pool(
    meta: dict[str, Any],
    score_lessons: list[Lesson],
    champion_names: list[str],
    *,
    note: str,
) -> PlanningResult:
    scored = _score_toolchain_candidates(meta, score_lessons, set(champion_names))
    selected_name = max(scored, key=lambda n: scored[n][0])
    best_score, best_matched = scored[selected_name]
    return PlanningResult(
        TOOLCHAINS[selected_name],
        best_score,
        best_matched,
        f"{note}; champion-pool fallback score={best_score:.2f}; picked={selected_name}",
    )


def meta_champion_toolchains(meta: dict[str, Any]) -> list[str]:
    """Compact registry candidates for hold-out / champion selection (not lesson injection)."""
    names = [
        "none_none_none_none_standard_timexer",
        "none_none_none_none_standard_multipatchformer",
    ]
    num_variates = float(meta.get("num_variates") or 1)
    missing = float(meta.get("missing_rate") or 0.0)
    seasonality = meta.get("seasonality_level")
    if num_variates >= 64:
        names.append("linear_iqr_none_fft_standard_itransformer")
    if seasonality in {"moderate", "strong"}:
        names.append("linear_zscore_none_fft_standard_timesnet")
    if missing < 0.05 and seasonality in {"weak", "moderate"}:
        names.append("linear_none_none_none_standard_patchtst")
    return [n for n in names if n in TOOLCHAINS]


def plan_toolchain(
    meta: dict[str, Any],
    lessons: list[Lesson] | None = None,
    *,
    relaxed_lesson_match: bool = False,
    skip_meta_filter: bool = False,
) -> PlanningResult:
    lesson_pool = list(SEED_VERIFIED_LESSONS) if lessons is None else list(lessons)
    if skip_meta_filter:
        active_lessons = list(lesson_pool)
    else:
        active_lessons = _filter_lessons_for_planning(
            lesson_pool, meta, for_verify=False, relaxed=relaxed_lesson_match
        )
    scored = _score_toolchain_candidates(meta, active_lessons, set(TOOLCHAINS.keys()))
    selected_name = max(scored, key=lambda n: scored[n][0])
    best_score, best_matched = scored[selected_name]
    selected = TOOLCHAINS[selected_name]
    reason = _reason(meta, selected, best_matched, best_score)
    return PlanningResult(selected, best_score, best_matched, reason)


def champion_plan_toolchain(
    meta: dict[str, Any],
    lesson_pool: list[Lesson],
    client: Any | None = None,
    *,
    top_k: int = 5,
    min_overlap: int = 1,
    llm_retries: int = 3,
    use_llm: bool = True,
) -> PlanningResult:
    """Stage-wise LLM planning with matched lessons; champion-pool-only when none match."""
    from .lessons import lessons_for_target_planning

    pred_len = int(meta["pred_len"])
    champions = meta_champion_toolchains(meta)
    llm_lessons, score_lessons = lessons_for_target_planning(
        lesson_pool,
        meta,
        pred_len,
        top_k=top_k,
        min_overlap=min_overlap,
    )
    has_match = bool(score_lessons)
    print(
        f"[champion-plan] pl={pred_len} has_match={has_match} "
        f"llm_lessons={len(llm_lessons)} score_lessons={len(score_lessons)}"
    )

    if use_llm and client is not None and getattr(client, "available", False):
        if has_match and llm_lessons:
            llm_pick = llm_plan_toolchain(
                meta,
                llm_lessons,
                client,
                retries=llm_retries,
                skip_meta_filter=True,
            )
            if llm_pick is not None:
                return PlanningResult(
                    llm_pick.toolchain,
                    0.0,
                    llm_pick.matched_lessons,
                    f"llm_stage_plan matched_lessons={len(llm_lessons)}; {llm_pick.reason}",
                )
            print("[champion-plan] LLM failed with matched lessons; fallback to champion pool")
            return _pick_from_champion_pool(
                meta,
                score_lessons,
                champions,
                note="llm_failed_with_matched_lessons",
            )

        if not has_match:
            champion_ctx = planning_context_for_champions(champions)
            llm_pick = llm_plan_toolchain(
                meta,
                [],
                client,
                retries=llm_retries,
                skip_meta_filter=True,
                allowed_toolchains=frozenset(champions),
                planning_context=champion_ctx,
                champion_pool_only=True,
            )
            if llm_pick is not None:
                return PlanningResult(
                    llm_pick.toolchain,
                    0.0,
                    [],
                    f"llm_stage_plan champion_pool_only; {llm_pick.reason}",
                )
            print("[champion-plan] LLM failed with no lesson match; fallback to champion pool scoring")
            return _pick_from_champion_pool(
                meta,
                [],
                champions,
                note="llm_failed_no_lesson_match",
            )

    if has_match:
        return _pick_from_champion_pool(
            meta,
            score_lessons,
            champions,
            note="llm_unavailable_with_matched_lessons",
        )
    return _pick_from_champion_pool(
        meta,
        [],
        champions,
        note="llm_unavailable_no_lesson_match",
    )


def rank_toolchains(meta: dict[str, Any], lessons: list[Lesson] | None = None) -> list[PlanningResult]:
    primary = plan_toolchain(meta, lessons=lessons)
    results = []
    lesson_pool = list(SEED_VERIFIED_LESSONS) if lessons is None else list(lessons)
    active = _filter_lessons_for_planning(lesson_pool, meta, for_verify=False)
    for name in TOOLCHAINS:
        cloned_meta = dict(meta)
        chosen = TOOLCHAINS[name]
        matched = [
            lesson
            for lesson in active
            if (
                (lesson.tool and name in toolchains_matching_tool(lesson.tool, lesson.task_category))
                or lesson.toolchain == name
            )
        ]
        score = sum((1.0 if lesson.lesson_type == "positive" else -0.7) * lesson.confidence for lesson in matched)
        scores = {n: 0.0 for n in TOOLCHAINS}
        _add_heuristic_scores(cloned_meta, scores)
        score += scores[name]
        results.append(PlanningResult(chosen, score, matched, _reason(cloned_meta, chosen, matched, score)))
    results.sort(key=lambda item: item.score, reverse=True)
    if results and results[0].toolchain.name != primary.toolchain.name:
        results.insert(0, primary)
    return results


def llm_plan_toolchain(
    meta: dict[str, Any],
    lessons: list[Lesson] | None,
    client: Any,
    retries: int = 3,
    *,
    relaxed_lesson_match: bool = False,
    skip_meta_filter: bool = False,
    allowed_toolchains: frozenset[str] | None = None,
    planning_context: dict[str, Any] | None = None,
    champion_pool_only: bool = False,
) -> PlanningResult | None:
    """Stage 4: LLM picks one tool per stage, resolved to a registered toolchain."""
    if not getattr(client, "available", False):
        return None
    lesson_pool = list(SEED_VERIFIED_LESSONS) if lessons is None else list(lessons)
    if skip_meta_filter:
        matched = list(lesson_pool)
    else:
        matched = _filter_lessons_for_planning(
            lesson_pool, meta, for_verify=False, relaxed=relaxed_lesson_match
        )
    lesson_dicts = [lesson.to_dict() for lesson in matched]
    if planning_context is None:
        planning_context = paper_planning_context()
    prompt = lesson_guided_planning_prompt(
        meta,
        lesson_dicts,
        planning_context,
        champion_pool_only=champion_pool_only,
    )
    last_err: Exception | None = None
    for _ in range(max(1, retries + 1)):
        try:
            response = client.complete_json(prompt)
        except Exception as exc:  # pragma: no cover
            last_err = exc
            continue
        if not response:
            continue
        toolchain_name = _resolve_planned_toolchain(response)
        if toolchain_name is None:
            print(
                f"[llm-planner] could not materialize stages (invalid combo or missing fields): "
                f"{response.get('stages')!r}"
            )
            continue
        if allowed_toolchains is not None and toolchain_name not in allowed_toolchains:
            print(
                f"[llm-planner] resolved {toolchain_name!r} not in allowed champion pool; retry"
            )
            continue
        matched_ids = set(response.get("matched_lessons") or [])
        used = [l for l in matched if l.lesson_id in matched_ids] or matched
        reason = str(response.get("reason", "")) or f"LLM stage-wise plan -> {toolchain_name}."
        return PlanningResult(TOOLCHAINS[toolchain_name], 0.0, used, reason)
    if last_err is not None:
        print(f"[llm-planner] giving up after {retries + 1} attempts: {last_err}")
    return None


def _add_heuristic_scores(meta: dict[str, Any], scores: dict[str, float]) -> None:
    num_variates = float(meta.get("num_variates") or 1)
    seasonality = meta.get("seasonality_level")
    trend = meta.get("trend_level")
    volatility = meta.get("volatility_level")
    missing = float(meta.get("missing_rate") or 0.0)
    corr = meta.get("mean_abs_correlation")
    corr_val = float(corr) if corr is not None else 0.0

    def bump(name: str, delta: float) -> None:
        if name in scores:
            scores[name] += delta

    if num_variates >= 64:
        bump("linear_iqr_fft_standard_itransformer", 0.45)
        if corr_val >= 0.05:
            bump("linear_iqr_fft_standard_itransformer", 0.25)
    if seasonality == "strong":
        bump("linear_knn_fft_standard_patchtst", 0.25)
        bump("linear_zscore_fft_standard_timesnet", 0.2)
        bump("linear_iqr_classical_standard_timemixer", 0.2)
    if seasonality in {"moderate", "strong"} and trend in {"moderate", "strong"}:
        bump("linear_iqr_classical_standard_timemixer", 0.35)
    if volatility in {"high", "medium"}:
        bump("linear_iqr_classical_standard_timemixer", 0.15)
        bump("linear_iqr_fft_standard_itransformer", 0.1)
    if missing >= 0.05:
        bump("knn_iqr_fft_standard_patchtst", 0.2)
        bump("mean_zscore_fft_standard_dlinear", 0.15)
    if seasonality in {"weak", "moderate"} and trend in {"weak", "moderate"}:
        bump("linear_knn_fft_standard_dlinear", 0.25)


def _reason(meta: dict[str, Any], toolchain: ToolchainSpec, lessons: list[Lesson], score: float) -> str:
    if lessons:
        tools = ", ".join(sorted({lesson.tool or lesson.toolchain for lesson in lessons}))
        return (
            f"[Phenomenon]: Matched lessons reference tools ({tools}) under meta-features "
            f"seasonality={meta.get('seasonality_level')}, trend={meta.get('trend_level')}, "
            f"missing_rate={meta.get('missing_rate')}. "
            f"[Causal Analysis]: {toolchain.name} aggregates those stage tools (score={score:.2f})."
        )
    return (
        f"{toolchain.name} is selected for seasonality={meta.get('seasonality_level')}, "
        f"trend={meta.get('trend_level')}, missing_rate={meta.get('missing_rate')} (score={score:.2f})."
    )
