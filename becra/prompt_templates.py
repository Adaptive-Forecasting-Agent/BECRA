from __future__ import annotations

import json
from typing import Any


_VERIFY_PLAN_LESSON_EXAMPLE = """  1. Lesson1: [Phenomenon] KNN belongs to the Imputation tool category. When the dataset has a low missing rate, such as missing_rate=0.06, KNN tends to support strong forecasting performance. [Analysis] KNN estimates missing values from similar neighboring samples. With low missingness, most local references remain valid, so imputation noise is limited.
  2. Lesson2: [Phenomenon] FFT belongs to the Decomposition tool category. When the dataset has a clear 24-hour seasonal peak, FFT tends to improve forecasting performance. [Analysis] FFT extracts dominant frequency components. A 24-hour seasonal peak indicates stable daily periodicity, allowing FFT to provide clearer seasonal signals for downstream forecasting."""


def _format_meta_block(meta: dict[str, Any]) -> str:
    """Compact meta line: {key=value, ...}."""
    skip = {"dataset", "n_runs", "mean_reward"}
    parts = [f"{k}={meta[k]}" for k in sorted(meta) if k not in skip and meta.get(k) is not None]
    return "{" + ", ".join(parts) + "}"


def _lessons_description(lessons: list[dict[str, Any]]) -> str:
    """Format lessons for prompt injection."""
    lines: list[str] = []
    for idx, lesson in enumerate(lessons, start=1):
        phen = str(lesson.get("phenomenon") or "").strip()
        anal = str(lesson.get("analysis") or "").strip()
        tool = str(lesson.get("tool") or "").strip()
        category = str(lesson.get("task_category") or "").strip()
        if phen or anal:
            text = phen
            if anal and anal not in phen:
                text = f"{phen} {anal}" if phen else anal
            lines.append(f"  {idx}. Lesson{idx}: {text}")
            continue
        if tool and category and category != "Strategy":
            lines.append(
                f"  {idx}. Lesson{idx}: [Phenomenon] {tool} ({category}). "
                f"{lesson.get('important_features') or lesson.get('description') or ''}"
            )
            continue
        desc = str(lesson.get("description") or lesson.get("important_features") or "")
        lines.append(f"  {idx}. Lesson{idx}: {desc}")
    return "\n".join(lines) if lines else _VERIFY_PLAN_LESSON_EXAMPLE


VERIFIED_LESSON_EXAMPLES = [
    {
        "title": "Verified Lesson 1 (TimesNet -- Imputation)",
        "phenomenon": (
            "TimesNet belongs to the Imputation tool category. When the dataset exhibits continuous "
            "point-wise missing data, the observed average imputation accuracy for TimesNet is 30%. "
            "When the dataset exhibits continuous and uniformly dispersed block-wise missing data, "
            "the observed average accuracy is 70%."
        ),
        "analysis": (
            "TimesNet is a neural network model that requires training. If the missing values are too "
            "numerous and dispersed (point-wise), it is impossible to effectively slice complete and "
            "valid training samples for TimesNet to learn temporal variations, resulting in poor "
            "performance. Conversely, if the missing distribution is block-wise or concentrated, "
            "satisfying the condition for constructing a valid training set, then TimesNet performs well."
        ),
    },
    {
        "title": "Verified Lesson 2 (Linear Interpolation -- Imputation)",
        "phenomenon": (
            "Linear Interpolation belongs to the Imputation tool category. When the dataset exhibits "
            "continuous point-wise missing data, the observed average imputation accuracy is 85%. "
            "When the dataset exhibits large-scale block-wise missing data, the observed average "
            "accuracy is 40%."
        ),
        "analysis": (
            "Linear Interpolation is a non-parametric, training-free statistical method that relies on "
            "local neighborhood information for inference. When data presents point-wise missing "
            "patterns, the local context is well-preserved, and the method is not constrained by the "
            "inability to construct a training set, leading to superior performance. Under block-wise "
            "missing patterns, the method fails due to the lack of local reference information."
        ),
    },
]


FILTERED_LESSON_EXAMPLES = [
    {
        "title": "Filtered Lesson A (Spurious Attribution)",
        "phenomenon": (
            "PatchTST belongs to the Forecasting tool category. When the dataset has a series length "
            "> 10000, the observed average forecasting accuracy is high. When the series length < 5000, "
            "the accuracy drops significantly."
        ),
        "analysis": (
            "PatchTST benefits from longer series because more data provides richer patch representations "
            "for self-attention."
        ),
        "filter_reason": (
            "Series length is confounded with volatility, seasonality, and domain. Reject unless controlled "
            "verification shows the lesson improves planning under fixed comparable conditions."
        ),
    },
    {
        "title": "Filtered Lesson B (Over-Generalization)",
        "phenomenon": (
            "STL decomposition belongs to the Decomposition tool category. Using STL always improves "
            "forecasting performance by separating trend and seasonal components, making the residual easier to model."
        ),
        "analysis": "STL separates trend and seasonal components, which universally simplifies the forecasting task.",
        "filter_reason": "The claim is universal and lacks activation conditions. Reject lessons that do not state when the strategy helps or hurts.",
    },
]


PAPER_FORECASTING_LESSON_EXAMPLE = {
    "lesson_id": "none_none_none_none_standard_timexer_TimeXer_L1",
    "lesson_type": "positive",
    "lesson_level": "stage",
    "task_category": "Forecasting",
    "tool": "TimeXer",
    "evidence_toolchain": "none_none_none_none_standard_timexer",
    "activation_conditions": {
        "mean_abs_correlation": ">0.25",
        "series_length": ">15000",
    },
    "phenomenon": (
        "[Phenomenon]: On evidence strategy none_none_none_none_standard_timexer, TimeXer forecasting "
        "tends to succeed when mean absolute correlation is moderate and series length is long."
    ),
    "analysis": (
        "[Analysis]: Within this minimal-preprocessing chain, standard scaler stabilizes cross-variate scale "
        "so TimeXer can model dependence directly; failures on this strategy correlate with very high "
        "volatility where extra preprocessing helps."
    ),
    "confidence": 0.82,
}

PAPER_TRANSFERABLE_STAGE_BAD_EXAMPLE = {
    "lesson_id": "standard_scaler_normalization_L1",
    "lesson_type": "positive",
    "task_category": "Normalization",
    "tool": "standard_scaler",
    "activation_conditions": {
        "mean_abs_correlation": ">0.3",
        "series_length": ">20000",
    },
    "phenomenon": (
        "[Phenomenon]: Standard Scaler normalization tends to improve forecasting performance when "
        "mean absolute correlation is above 0.3 and series length is greater than 20,000."
    ),
    "filter_reason": (
        "Reject generic transferable preprocessing lessons: no evidence_toolchain, no forecasting model, "
        "and activation keys that do not separate positive vs negative runs on the fixed strategy."
    ),
}

PAPER_STRATEGY_LESSON_EXAMPLE = {
    "lesson_id": "linear_iqr_none_fft_standard_itransformer_strategy_L1",
    "lesson_type": "positive",
    "lesson_level": "strategy",
    "task_category": "Strategy",
    "tool": "",
    "evidence_toolchain": "linear_iqr_none_fft_standard_itransformer",
    "activation_conditions": {
        "seasonality_level": ["moderate", "strong"],
        "series_length": ">15000",
    },
    "phenomenon": (
        "[Phenomenon]: Strategy linear_iqr_none_fft_standard_itransformer (linear imputation, IQR, FFT, iTransformer) "
        "tends to succeed when seasonality is moderate-to-strong and variate count is moderate."
    ),
    "analysis": (
        "[Analysis]: Under these meta-features the preprocessing chain stabilizes periodic structure before "
        "iTransformer models cross-variate dependence; poor outcomes on this strategy correlate with weak seasonality "
        "or very long horizons where FFT gain fades."
    ),
    "confidence": 0.8,
}


def lesson_induction_paper_prompt(
    toolchain_name: str,
    toolchain_stages: dict[str, str],
    positive_cases: list[dict[str, Any]],
    negative_cases: list[dict[str, Any]],
    contrast_summary: dict[str, Any] | None = None,
    *,
    max_strategy_lessons: int = 1,
    max_stage_lessons: int = 6,
) -> str:
    """Induce on one fixed strategy: contrast (dataset, pred_len) outcomes; strategy- and stage-level lessons."""
    group_id = toolchain_name
    summary_block = ""
    if contrast_summary:
        summary_block = (
            "\nContrastive summary (positive vs negative runs on this fixed strategy):\n"
            f"{json.dumps(contrast_summary, indent=2, ensure_ascii=False)}\n"
        )
    stage_tools = [
        {"stage": s, "task_category": c, "tool": t}
        for s, c, t in [
            ("imputation", "Imputation", toolchain_stages.get("imputation")),
            ("anomaly_handling", "Anomaly Handling", toolchain_stages.get("anomaly_handling")),
            ("transformation", "Transformation", toolchain_stages.get("transformation")),
            ("decomposition", "Decomposition", toolchain_stages.get("decomposition")),
            ("normalization", "Normalization", toolchain_stages.get("normalization")),
            ("forecasting", "Forecasting", toolchain_stages.get("forecasting")),
        ]
        if t and str(t).lower() != "none"
    ]
    return f"""You are an expert in causal analysis for time series forecasting.

Given contrastive pairs under the same toolchain strategy:
- Fixed strategy (toolchain slug): {toolchain_name}
- Stages: {json.dumps(toolchain_stages, indent=2, ensure_ascii=False)}
- Stage tools to cover in stage-level lessons: {json.dumps(stage_tools, indent=2, ensure_ascii=False)}

Positive runs (higher reward on this strategy):
{json.dumps(positive_cases, indent=2, ensure_ascii=False)}

Negative runs (lower reward on this strategy):
{json.dumps(negative_cases, indent=2, ensure_ascii=False)}
{summary_block}
Analyze the experimental results to extract causal lessons that explain why this forecasting strategy succeeds or fails under specific dataset characteristics.

Focus on identifying:
1. Causal relationships between dataset features and strategy performance
2. Necessary and sufficient conditions for strategy success
3. Causal mechanisms that explain the relationships
4. Confidence levels based on evidence strength

Write JSON only with:
1) At most {max_strategy_lessons} **strategy-level** lesson(s) (`lesson_level`: "strategy", `task_category`: "Strategy") about when the **whole toolchain** helps or hurts. Refer to the slug and its stage composition; do not claim a different toolchain.
2) Up to {max_stage_lessons} **stage-level** lesson(s) (`lesson_level`: "stage") — **one lesson per stage tool** listed above (including **Forecasting** model). Each names exactly one `tool` that appears in the stages for that `task_category`.
3) **Every** `lesson_id` MUST start with `{group_id}_` (toolchain-scoped). Never reuse generic ids like `timexer_forecasting_L1` or `standard_scaler_normalization_L1` across strategies.
4) Set `evidence_toolchain` to `{toolchain_name}` on every lesson. For **Forecasting** stage lessons, `phenomenon` MUST name the forecasting model **and** the evidence strategy slug (e.g. "On evidence strategy none_..._timexer, TimeXer ..."). Do NOT write model-agnostic preprocessing claims that could apply to any toolchain.
5) Use at most **2** meta-feature keys in `activation_conditions` (broad rules). You MAY include `pred_len` as a third key (list of integers) when the pattern is horizon-specific. Prefer keys from `categorical_differences` / `numeric_differences` in the contrastive summary — do NOT use features that are similar across positive and negative runs on this strategy.
   **Strategy-level only:** write activation rules that **all or most positive runs** satisfy (strict match on multiple source datasets). Prefer categorical **lists** (e.g. `"seasonality_level": ["moderate", "strong"]`) over a single narrow level; for numeric thresholds, stay slightly below the smallest positive-run value. Avoid brittle pairs like a single volatility+trend combo that only one dataset matches.
6) `phenomenon` / `analysis` use [Phenomenon]: and [Analysis]: prefixes. Attribute outcomes to agent strategy choices under meta-feature differences, not data-generating causality.
7) Prefer conditions satisfied by several positive runs; avoid universal claims. The **Forecasting** lesson is the primary lever for model choice — make it specific to this evidence toolchain.

Strategy-level example:
{json.dumps(PAPER_STRATEGY_LESSON_EXAMPLE, indent=2, ensure_ascii=False)}

Forecasting stage example (toolchain-specific — follow this pattern):
{json.dumps(PAPER_FORECASTING_LESSON_EXAMPLE, indent=2, ensure_ascii=False)}

Bad stage lesson (reject this pattern):
{json.dumps(PAPER_TRANSFERABLE_STAGE_BAD_EXAMPLE, indent=2, ensure_ascii=False)}

Return:
{{
  "strategy_lessons": [ {{ ... }} ],
  "stage_lessons": [ {{ ... }}, ... ]
}}"""


def lesson_induction_group_prompt(
    toolchain_name: str,
    toolchain_stages: dict[str, str],
    pred_len: int,
    positive_cases: list[dict[str, Any]],
    negative_cases: list[dict[str, Any]],
    contrast_summary: dict[str, Any] | None = None,
    max_lessons: int = 2,
) -> str:
    group_id = f"{toolchain_name}_pl{pred_len}"
    summary_block = ""
    if contrast_summary:
        summary_block = (
            "\nContrastive meta-feature summary (positive group vs negative group at this pred_len):\n"
            f"{json.dumps(contrast_summary, indent=2, ensure_ascii=False)}\n"
        )
    return f"""You are an expert in time-series forecasting, tool planning, and agent-level causal reasoning.

BECRA induces **single-tool** causal lessons for Lesson-Guided Planning. Each lesson names ONE tool in ONE stage category.
The evidence toolchain below is fixed context for this contrast — do NOT write lessons about the whole toolchain slug.

Evidence toolchain:
{toolchain_name}
Stages: {json.dumps(toolchain_stages, indent=2, ensure_ascii=False)}

Forecast horizon (fixed): pred_len = {pred_len}

Positive group (same toolchain + pred_len; higher rewards):
{json.dumps(positive_cases, indent=2, ensure_ascii=False)}

Negative group (same toolchain + pred_len; lower rewards):
{json.dumps(negative_cases, indent=2, ensure_ascii=False)}
{summary_block}
Write at most {max_lessons} lessons (prefer 1 strong lesson if sufficient). Each lesson MUST:
- Use a unique lesson_id prefixed with "{group_id}_".
- Name exactly one **preprocessing** `tool` from imputation, anomaly_handling, or decomposition in the stages above.
- FORBIDDEN: lessons whose `tool` is a forecasting model (PatchTST, iTransformer, TimeMixer, TimesNet, DLinear, Autoformer, TiDE, etc.) or `task_category` = Forecasting.
- Set `task_category` to Imputation | Anomaly Handling | Decomposition only.
- Write `phenomenon` and `analysis` about that single preprocessing tool (e.g. knn, iqr, fft) — NOT the whole toolchain slug.
- Use at most **2** activation_conditions keys (broad meta-features only), e.g. {{"seasonality_level": ["moderate", "strong"], "missing_rate": "<0.1"}} (NOT nested feature_name/operator objects). Choose keys/values so several source datasets in the positive group would satisfy **all** conditions at verify (strict match); pred_len is added by the system.
- Do NOT add pred_len to activation_conditions; the system binds this lesson to pred_len={pred_len} automatically for verify/planning.

Good examples:
{json.dumps(VERIFIED_LESSON_EXAMPLES, indent=2, ensure_ascii=False)}

Bad examples:
{json.dumps(FILTERED_LESSON_EXAMPLES, indent=2, ensure_ascii=False)}

Return JSON only:
{{
  "lessons": [
    {{
      "lesson_id": "{group_id}_L1",
      "lesson_type": "positive|negative",
      "tool": "exact_tool_name",
      "task_category": "Imputation|Anomaly Handling|Decomposition",
      "evidence_toolchain": "{toolchain_name}",
      "activation_conditions": {{}},
      "phenomenon": "[Phenomenon]: ...",
      "analysis": "[Analysis]: ...",
      "confidence": 0.0
    }}
  ]
}}"""


def lesson_induction_pair_prompt(
    toolchain_name: str,
    toolchain_stages: dict[str, str],
    pred_len: int,
    positive_case: dict[str, Any],
    negative_case: dict[str, Any],
    pair_index: int,
) -> str:
    pair_id = f"{toolchain_name}_pl{pred_len}_pair{pair_index}"
    return f"""You are an expert in time-series forecasting, tool planning, and agent-level causal reasoning.

BECRA induces **single-tool** causal lessons for Lesson-Guided Planning. Each lesson must attribute success or failure
to ONE tool in ONE stage category (Imputation, Anomaly Handling, Transformation, Decomposition, Normalization, or Forecasting).
The full six-stage evidence toolchain is context only — do NOT write phenomenon/analysis about the whole toolchain slug.

Evidence toolchain (fixed context for this contrast):
{toolchain_name}
Stages: {json.dumps(toolchain_stages, indent=2, ensure_ascii=False)}

Forecast horizon (fixed for this pair): pred_len = {pred_len}

Positive case (same toolchain + pred_len, higher reward):
{json.dumps(positive_case, indent=2, ensure_ascii=False)}

Negative case (same toolchain + pred_len, lower reward):
{json.dumps(negative_case, indent=2, ensure_ascii=False)}

Write exactly ONE candidate lesson (a single object in the lessons array). Each lesson MUST:
- Name exactly one `tool` from the stages above (e.g. TimeMixer, iqr, linear_interpolation, classical_decomposition).
- Set `task_category` to that tool's stage (Forecasting, Imputation, Anomaly Handling, etc.).
- Write `phenomenon` and `analysis` about that single tool's mechanism under the meta-feature difference between the two cases.
- Be conditional (when to use / avoid), not universal.
- Use activation_conditions checkable from meta-features.

Good verified lesson style (single-tool):
{json.dumps(VERIFIED_LESSON_EXAMPLES, indent=2, ensure_ascii=False)}

Bad lessons:
{json.dumps(FILTERED_LESSON_EXAMPLES, indent=2, ensure_ascii=False)}

Return JSON only:
{{
  "lessons": [
    {{
      "lesson_id": "{pair_id}_L1",
      "lesson_type": "positive|negative",
      "tool": "exact_tool_name_from_stages",
      "task_category": "Imputation|Anomaly Handling|Decomposition",
      "evidence_toolchain": "{toolchain_name}",
      "activation_conditions": {{
        "feature_name": "operator and threshold or categorical value"
      }},
      "phenomenon": "[Phenomenon]: <Tool> belongs to the <task_category> tool category. When ...",
      "analysis": "[Analysis]: mechanism for this tool only ...",
      "confidence": 0.0,
      "verification_notes": "what controlled intervention should test"
    }}
  ]
}}"""


def lesson_induction_prompt(
    toolchain_name: str,
    positive_meta: list[dict[str, Any]],
    negative_meta: list[dict[str, Any]],
    positive_scores: list[dict[str, Any]] | None = None,
    negative_scores: list[dict[str, Any]] | None = None,
    contrast_summary: dict[str, Any] | None = None,
    toolchain_stages: dict[str, str] | None = None,
    pred_len: int | None = None,
) -> str:
    """Legacy bulk prompt; prefer lesson_induction_pair_prompt for new runs."""
    if pred_len is not None and len(positive_meta) == 1 and len(negative_meta) == 1:
        return lesson_induction_pair_prompt(
            toolchain_name,
            toolchain_stages or {},
            pred_len,
            positive_meta[0],
            negative_meta[0],
            pair_index=0,
        )
    summary_block = ""
    if contrast_summary:
        summary_block = (
            "\nContrastive meta-feature summary:\n"
            f"{json.dumps(contrast_summary, indent=2, ensure_ascii=False)}\n"
        )
    stages_block = ""
    if toolchain_stages:
        stages_block = f"\nEvidence toolchain stages:\n{json.dumps(toolchain_stages, indent=2, ensure_ascii=False)}\n"
    return f"""You are an expert in time-series forecasting, tool planning, and agent-level causal reasoning.

Induce **single-tool** lessons (not whole-toolchain claims). Evidence toolchain: {toolchain_name}
{stages_block}{summary_block}
Positive outcomes:
{json.dumps(positive_meta, indent=2, ensure_ascii=False)}

Negative outcomes:
{json.dumps(negative_meta, indent=2, ensure_ascii=False)}

Return JSON with fields: lesson_id, lesson_type, tool, task_category, evidence_toolchain, activation_conditions, phenomenon, analysis, confidence.
Each phenomenon/analysis must refer to one tool only.

Good examples:
{json.dumps(VERIFIED_LESSON_EXAMPLES, indent=2, ensure_ascii=False)}

Bad examples:
{json.dumps(FILTERED_LESSON_EXAMPLES, indent=2, ensure_ascii=False)}

Return JSON only: {{ "lessons": [ ... ] }}"""


def _lessons_description_with_evidence(lessons: list[dict[str, Any]]) -> str:
    """Paper planning prompt asks the LLM to weigh 'strongest verified causal effects';
    append each lesson's evidence toolchain and verification statistics so it can."""
    lines: list[str] = []
    for idx, lesson in enumerate(lessons, start=1):
        phen = str(lesson.get("phenomenon") or "").strip()
        anal = str(lesson.get("analysis") or "").strip()
        text = phen if phen else ""
        if anal and anal not in text:
            text = f"{text} {anal}".strip()
        if not text:
            text = str(lesson.get("description") or lesson.get("important_features") or "")
        evidence_bits = [f"evidence_toolchain={lesson.get('toolchain')}"]
        if lesson.get("lesson_type"):
            evidence_bits.append(f"type={lesson['lesson_type']}")
        conditions = lesson.get("activation_conditions") or {}
        if conditions:
            cond_txt = ", ".join(f"{k}{v}" if str(v).startswith((">", "<", "=")) else f"{k}={v}" for k, v in conditions.items())
            evidence_bits.append(f"activation_conditions[{cond_txt}]")
        if lesson.get("confidence") is not None:
            evidence_bits.append(f"confidence={float(lesson['confidence']):.2f}")
        if lesson.get("verification_effect") is not None:
            evidence_bits.append(
                f"verified_causal_effect={float(lesson['verification_effect']):.2f}"
            )
        if lesson.get("verification_support") is not None:
            evidence_bits.append(f"verify_episodes={int(lesson['verification_support'])}")
        episodes = lesson.get("verification_episodes") or []
        if episodes:
            wins = sum(
                1
                for ep in episodes
                if float(ep.get("y_treat") or 0) > float(ep.get("y_ctrl") or 0)
            )
            evidence_bits.append(f"paired_rollout_wins={wins}/{len(episodes)}")
            lost_to = sorted(
                {
                    str(ep.get("a_ctrl"))
                    for ep in episodes
                    if float(ep.get("y_treat") or 0) < float(ep.get("y_ctrl") or 0)
                    and ep.get("a_ctrl")
                }
            )
            if lost_to:
                evidence_bits.append(f"lost_paired_rollouts_to=[{', '.join(lost_to)}]")
        lines.append(f"  {idx}. Lesson{idx}: {text} ({', '.join(evidence_bits)})")
    return "\n".join(lines) if lines else _lessons_description(lessons)


def lesson_guided_planning_prompt(
    meta_features: dict[str, Any],
    lessons: list[dict[str, Any]],
    planning_context: Any,
    *,
    champion_pool_only: bool = False,
) -> str:
    if champion_pool_only:
        return f"""You are an expert in time series forecasting and tool planning.
Your task is to plan the best forecasting strategy for the new dataset by meta features.

You are given:
- Meta features of the new dataset: {_format_meta_block(meta_features)}

**No causal lessons matched** this meta-profile at this horizon. Do NOT invent lesson-based preprocessing claims.

Plan a **six-stage** toolchain by choosing one executable tool per stage.
Stages (in order): Imputation -> Anomaly Handling -> Transformation -> Decomposition -> Normalization -> Forecasting.

Champion pool (the resolved toolchain name MUST be exactly one of these registered slugs):
{json.dumps(planning_context.get("champion_toolchains_only", []), indent=2, ensure_ascii=False)}

Reference stage patterns for each champion (pick stages that resolve to one slug above):
{json.dumps(planning_context.get("stages_by_champion", {}), indent=2, ensure_ascii=False)}

Allowed tools per stage (union over the champion pool only):
{json.dumps(planning_context.get("stage_tools", {}), indent=2, ensure_ascii=False)}

Planning requirements:
1. Pick **one tool per stage** — do NOT return a toolchain slug directly in a single field; return a ``stages`` object.
2. After stage choices are resolved, the toolchain must be **exactly one** name from the champion pool list.
3. Prefer ``none`` for imputation, anomaly_handling, and decomposition when using TimeXer or MultiPatchFormer.
4. Return concrete stage tools aligned with one champion pattern.

Return JSON only:
{{
  "stages": {{
    "imputation": "...",
    "anomaly_handling": "...",
    "transformation": "...",
    "decomposition": "...",
    "normalization": "...",
    "forecasting": "..."
  }},
  "matched_lessons": [],
  "reason": "[Phenomenon]: ... [Causal Analysis]: ..."
}}"""

    lessons_block = (
        _lessons_description_with_evidence(lessons)
        if lessons
        else "  (no lessons matched this meta-profile)"
    )
    ctx = planning_context if isinstance(planning_context, dict) else {}
    compact_ctx = {
        key: value
        for key, value in ctx.items()
        if key not in {"stage_tools", "paper_tool_library"}
    }
    verification = ctx.get("evidence_toolchain_verification") or {}
    scoreboard_lines: list[str] = []
    if verification:
        ranked = sorted(
            verification.items(),
            key=lambda item: (
                -int(item[1].get("won_as_control") or 0),
                -int(str(item[1].get("verified_rollout_record") or "0W").split("W")[0] or 0),
            ),
        )
        for name, stats in ranked:
            bits = [f"toolchain={name}"]
            if stats.get("won_as_control"):
                bits.append(f"won_as_control={stats['won_as_control']}")
            if stats.get("verified_rollout_record"):
                bits.append(f"as_treatment={stats['verified_rollout_record']}")
            bits.append(f"matched_lessons={stats.get('matched_lessons', 0)}")
            scoreboard_lines.append("  - " + ", ".join(bits))
    scoreboard = "\n".join(scoreboard_lines) if scoreboard_lines else "  (no verification digest)"
    primary_rule = (
        "it is the evidence toolchain with the highest verified causal support among matched lessons "
        "(informational hint only); weigh it against the other injected lessons and choose the strategy "
        "with the strongest verified causal fit to this dataset's meta-features"
        if ctx.get("planning_inject_scope") == "matched"
        else "stage choices should resolve to that slug unless a matched **Strategy** lesson explicitly "
        "names a different evidence toolchain with higher confidence"
    )
    return f"""You are an expert in time series forecasting and tool planning.
Your task is to plan the best forecasting strategy for the new dataset by meta features and lessons learned. You should adaptively match the meta-features of the dataset with lessons learned and select the most appropriate strategy based on these lessons learned.

You have the memory of causal lessons, each of which describes the conditions for the success or failure of certain prediction strategies.

You are given:
- Meta features of the new dataset: {_format_meta_block(meta_features)}
- Candidate evidence scoreboard (aggregated over ALL matched lessons, not just the injected subset; higher `won_as_control` means this strategy repeatedly beat other evaluated strategies in verification):
{scoreboard}
- Causal lessons (retrieval budget; use together with the scoreboard above):
{lessons_block}
- Executable tool library (pick one tool per stage): {json.dumps(planning_context.get("stage_tools", planning_context), indent=2, ensure_ascii=False)}

Where causal lessons describe the conditions for the success or failure of certain prediction strategies.

Select the most appropriate tool at each stage from the executable tool library for this dataset based on lessons. Consider:
1. Which lessons best match the dataset's characteristics?
2. Which strategies have the strongest verified causal effects?
3. How do the meta-features align with the lesson conditions?

Planning context (for ranking / evidence toolchains):
{json.dumps(compact_ctx, indent=2, ensure_ascii=False)}

Planning requirements:
1. Pick **one tool per stage** from the lists above (do NOT return a pre-baked toolchain slug; the system resolves stages to a registered toolchain).
2. classical_decomposition is only valid with forecasting DLinear or TimeMixer; otherwise use fft or none for decomposition.
3. Each lesson's `toolchain` field is the **evidence strategy slug** from explore. When `primary_evidence_toolchain` is set in the context below, {primary_rule}.
4. Weigh lessons by their **verified causal effect** and confidence (shown per lesson) together with how well their `activation_conditions` match this dataset's meta-features. Do NOT choose a strategy merely because more lessons mention it — lesson counts reflect exploration frequency, not superiority. The scoreboard's `won_as_control` is especially important: a strategy that repeatedly beat other evidence strategies as the control arm is strong causal evidence, even if fewer narrative lessons about it were retrieved.
5. **Reason causally about mechanism fit**: identify what is distinctive about this dataset's meta-features (e.g., variate count, cross-variate correlation, seasonality strength and period, sampling frequency, series length, missingness, anomaly profile, stationarity) and select the forecasting model whose inductive bias best fits those distinctive characteristics — combining the matched lessons' causal analyses with your own knowledge of how these architectures work. For example: architectures that explicitly model cross-variate/channel dependence suit datasets whose variate count and mean_abs_correlation are high, channel-independent patching suits few or weakly-coupled variates with clean periodic motifs, and multi-scale mixing or trend-seasonal decomposition suits strong interacting trend/seasonal components. A lesson whose causal mechanism specifically explains this dataset's profile outranks several generic lessons that would apply to almost any dataset.
6. **Preprocessing stages must earn their place**: adopt an imputation/anomaly/decomposition tool only when a matched lesson's mechanism applies to this dataset's actual meta-features. If `missing_rate` is ~0, do not add imputation. If a high-`won_as_control` candidate already uses `none` for anomaly_handling, do not prefer a heavier anomaly stage merely because some lessons mention it. Conversely, do adopt a stage when lessons show it causally helps under conditions this dataset satisfies.
7. Prefer lessons whose tool you adopt in the matching stage when evidence supports it. `evidence_toolchains` and `evidence_toolchain_stages` are observed evidence patterns, **not a closed candidate list**. Plan over the full executable tool library and compose stage tools from different matched lessons when their causal mechanisms are compatible with each other and with this dataset's meta-features. Every non-`none` stage must be justified by matched causal evidence or a directly relevant meta-feature, and the final combination must resolve to a registered executable toolchain.
8. **Deliberate before deciding**: first list the 2-4 meta-features that most distinguish this dataset, then shortlist the 3-5 strongest evidence patterns or causally supported stage compositions using the scoreboard and lessons, and only then choose. Fill `distinctive_meta` and `candidates` in the output with this deliberation.
9. Return concrete stage tools — not a vague model family essay.

Return JSON only:
{{
  "distinctive_meta": ["the 2-4 most distinctive meta-features of this dataset and their values"],
  "candidates": [
    {{"toolchain": "candidate evidence slug or proposed registered composition", "mechanism_fit": "one sentence: why its forecasting model fits or does not fit the distinctive meta-features", "preprocessing_audit": "one sentence per non-none imputation/anomaly/decomposition stage: is it justified by THIS dataset's missing_rate / anomaly_ratio / periodic structure, or is it unnecessary here?"}}
  ],
  "stages": {{
    "imputation": "...",
    "anomaly_handling": "...",
    "transformation": "...",
    "decomposition": "...",
    "normalization": "...",
    "forecasting": "..."
  }},
  "matched_lessons": ["lesson_id"],
  "reason": "[Phenomenon]: ... [Causal Analysis]: ..."
}}"""
