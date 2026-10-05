from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import TOOLCHAINS, ToolchainSpec, tool_library_as_dict
from .lessons import (
    Lesson,
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


def _preprocess_extras_unjustified(
    meta: dict[str, Any],
    shallow_name: str,
    deep_name: str,
) -> bool:
    """True when deep adds preprocess stages that this dataset's meta does not support."""
    if shallow_name not in TOOLCHAINS or deep_name not in TOOLCHAINS:
        return False
    s = TOOLCHAINS[shallow_name].stages
    d = TOOLCHAINS[deep_name].stages
    missing = float(meta.get("missing_rate") or 0.0)
    anomaly_ratio = float(meta.get("anomaly_ratio") or 0.0)
    seasonality = str(meta.get("seasonality_level") or "")
    dominant = meta.get("dominant_period")

    if s.get("imputation") in {None, "none"} and d.get("imputation") not in {None, "none"}:
        if missing < 1e-6:
            return True
    if s.get("anomaly_handling") in {None, "none"} and d.get("anomaly_handling") not in {
        None,
        "none",
    }:
        # Low anomaly mass: prefer evidence that omits anomaly handling.
        if anomaly_ratio < 0.05:
            return True
    if s.get("decomposition") in {None, "none"} and d.get("decomposition") not in {
        None,
        "none",
    }:
        # No clear period / weak seasonality: FFT/classical not meta-justified.
        if dominant in {None, "", 0, 0.0} and seasonality in {"", "weak", "none"}:
            return True
    return False


def _strict_forecasting_evidence(
    lessons: list[Lesson],
    meta: dict[str, Any],
    pred_len: int,
) -> list[str]:
    """Evidence toolchains whose Forecasting lesson strictly activates for this meta."""
    names: list[str] = []
    seen: set[str] = set()
    for lesson in lessons:
        if lesson.task_category != "Forecasting":
            continue
        name = lesson.toolchain
        if not name or name not in TOOLCHAINS or name in seen:
            continue
        if lesson_matches(meta, lesson, pred_len):
            names.append(name)
            seen.add(name)
    return names


def _parsimony_snap_to_evidence(
    resolved: str,
    meta: dict[str, Any],
    evidence_toolchains: list[str] | None,
    *,
    matched_lessons: list[Lesson] | None = None,
) -> str:
    """Remap LLM proposals toward shallow, mechanism-fitting retrieved evidence.

    Paper Alg. 5 still plans over the full library; this only remaps when:
    1) same forecasting model already appears in retrieved evidence with a shallower
       preprocess, and the extra stages are not supported by this dataset's meta; or
    2) some evidence toolchain has a *strictly matching* Forecasting lesson for this
       meta, but the proposal does not — then prefer the shallowest such evidence
       (prevents Normalization/Strategy-only or non-activating Forecasting support
       from nominating a forecast model when a better-fitting one is available).
    No dataset- or model-name hardcoding.
    """
    from .tools import evidence_toolchain_preprocessing_depth

    if resolved not in TOOLCHAINS:
        return resolved

    snapped = resolved
    evidence = [e for e in (evidence_toolchains or []) if e in TOOLCHAINS]
    pred_len = int(meta.get("pred_len") or 0)

    # (1) Same-model parsimony snap when unjustified preprocess depth is added.
    if evidence:
        model = TOOLCHAINS[resolved].forecasting
        same_model = [e for e in evidence if TOOLCHAINS[e].forecasting == model]
        if same_model:
            missing = float(meta.get("missing_rate") or 0.0)
            if missing < 1e-6:
                no_imp = [
                    e for e in same_model if TOOLCHAINS[e].imputation in {None, "none"}
                ]
                if no_imp:
                    same_model = no_imp

            pool = list(same_model)
            if resolved in evidence and resolved not in pool:
                pool.append(resolved)

            def ok_candidate(name: str) -> bool:
                if name == resolved:
                    return True
                return (
                    evidence_toolchain_preprocessing_depth(name)
                    < evidence_toolchain_preprocessing_depth(resolved)
                    and _preprocess_extras_unjustified(meta, name, resolved)
                ) or (
                    resolved not in evidence
                    and evidence_toolchain_preprocessing_depth(name)
                    <= evidence_toolchain_preprocessing_depth(resolved)
                )

            candidates = [e for e in pool if ok_candidate(e)]
            if not candidates:
                candidates = list(same_model) if resolved not in evidence else [resolved]

            candidate = min(
                candidates,
                key=lambda e: (evidence_toolchain_preprocessing_depth(e), e),
            )
            if candidate != snapped:
                print(
                    f"[paper-plan] parsimony-snap {snapped!r} -> {candidate!r} "
                    f"(same forecasting model in retrieved evidence)"
                )
                snapped = candidate

    # (2) Mechanism-fit snap: prefer evidence with strict Forecasting activation.
    lessons = list(matched_lessons or [])
    strict_fc = _strict_forecasting_evidence(lessons, meta, pred_len)
    if strict_fc and snapped not in strict_fc:
        best = min(
            strict_fc,
            key=lambda e: (evidence_toolchain_preprocessing_depth(e), e),
        )
        if best != snapped:
            print(
                f"[paper-plan] mechanism-snap {snapped!r} -> {best!r} "
                f"(strict Forecasting evidence activates; proposal does not)"
            )
            snapped = best

    # (3) Cross-model parsimony: if a shallower strict-Forecasting evidence chain
    # exists and the proposal's extra preprocess stages are not meta-justified,
    # snap to that shallow evidence (preprocess must earn its place).
    if strict_fc:
        best = min(
            strict_fc,
            key=lambda e: (evidence_toolchain_preprocessing_depth(e), e),
        )
        if (
            best != snapped
            and evidence_toolchain_preprocessing_depth(best)
            < evidence_toolchain_preprocessing_depth(snapped)
            and _preprocess_extras_unjustified(meta, best, snapped)
        ):
            print(
                f"[paper-plan] parsimony-snap {snapped!r} -> {best!r} "
                f"(shallower strict Forecasting evidence; extra preprocess unjustified)"
            )
            snapped = best
    return snapped


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
    """Lesson-vote scoring only (paper deployment fallback; no meta heuristics).

    Uses mean confidence over supporting lessons so explore/induce frequency cannot
    dominate (aligned with Alg. 4 prompt: do not prefer by lesson count).
    """
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
    out: dict[str, tuple[float, list[Lesson]]] = {}
    for name in names:
        support = matched[name]
        if support:
            out[name] = (scores[name] / len(support), support)
        else:
            out[name] = (scores[name], support)
    return out


def _score_toolchain_candidates(
    meta: dict[str, Any],
    active_lessons: list[Lesson],
    candidate_names: set[str] | list[str],
) -> dict[str, tuple[float, list[Lesson]]]:
    """Lesson votes + optional meta heuristics."""
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
                planning_context=ctx,
            )
            if pick is not None:
                evidence_names = list(ctx.get("evidence_toolchains") or [])
                snapped_name = _parsimony_snap_to_evidence(
                    pick.toolchain.name,
                    meta,
                    evidence_names,
                    # Use full strict-matched set (not only the top-k inject budget)
                    # so mechanism-fit snap sees every activating Forecasting lesson.
                    matched_lessons=matched,
                )
                if snapped_name != pick.toolchain.name and snapped_name in TOOLCHAINS:
                    pick = PlanningResult(
                        TOOLCHAINS[snapped_name],
                        pick.score,
                        pick.matched_lessons,
                        f"{pick.reason}; evidence_snap->{snapped_name}",
                    )
                sample_picks.append(pick)
        llm_pick: PlanningResult | None = None
        if sample_picks:
            counts: dict[str, int] = {}
            for pick in sample_picks:
                counts[pick.toolchain.name] = counts.get(pick.toolchain.name, 0) + 1

            def _tiebreak(name: str) -> tuple[int, int, int, int, int]:
                """When votes tie: strict Forecasting activation, control winners,
                avoid unjustified imputation, shallower preprocess, then any
                Forecasting lesson (mechanism-level evidence beats normalization-only)."""
                from .tools import evidence_toolchain_preprocessing_depth

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
                # Negated depth so shallower ranks higher under max().
                shallow = -evidence_toolchain_preprocessing_depth(name)
                strict_fc = int(
                    any(
                        lesson.toolchain == name
                        and lesson.task_category == "Forecasting"
                        and lesson_matches(meta, lesson, pred_len)
                        for lesson in llm_lessons
                    )
                )
                has_forecasting_lesson = int(
                    any(
                        lesson.toolchain == name and lesson.task_category == "Forecasting"
                        for lesson in llm_lessons
                    )
                )
                return (
                    strict_fc,
                    ctrl_wins,
                    -unjustified_impute,
                    shallow,
                    has_forecasting_lesson,
                )

            best_votes = max(counts.values())
            # Near-ties (within 1 vote): prioritize Alg. 4 parsimony / mechanism-fit
            # tie-breaks over a single-sample vote gap.
            contenders = [
                name for name, votes in counts.items() if votes >= max(1, best_votes - 1)
            ]
            modal_name = max(
                contenders,
                key=lambda name: (
                    _tiebreak(name),
                    counts[name],
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


def plan_toolchain(
    meta: dict[str, Any],
    lessons: list[Lesson] | None = None,
    *,
    relaxed_lesson_match: bool = False,
    skip_meta_filter: bool = False,
) -> PlanningResult:
    lesson_pool = [] if lessons is None else list(lessons)
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


def llm_plan_toolchain(
    meta: dict[str, Any],
    lessons: list[Lesson] | None,
    client: Any,
    retries: int = 3,
    *,
    relaxed_lesson_match: bool = False,
    skip_meta_filter: bool = False,
    planning_context: dict[str, Any] | None = None,
) -> PlanningResult | None:
    """Stage 4: LLM picks one tool per stage, resolved to a registered toolchain."""
    if not getattr(client, "available", False):
        return None
    lesson_pool = [] if lessons is None else list(lessons)
    if skip_meta_filter:
        matched = list(lesson_pool)
    else:
        matched = _filter_lessons_for_planning(
            lesson_pool, meta, for_verify=False, relaxed=relaxed_lesson_match
        )
    lesson_dicts = [lesson.to_dict() for lesson in matched]
    if planning_context is None:
        planning_context = paper_planning_context()
    prompt = lesson_guided_planning_prompt(meta, lesson_dicts, planning_context)
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
