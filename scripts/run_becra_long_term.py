#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from becra.config import LONG_TERM_HORIZONS, OUTPUT_ROOT, TOOLCHAINS, all_dataset_names, all_toolchain_names, get_dataset, get_toolchain, tool_library_as_dict
from becra.induction import (
    build_induction_prompts,
    heuristic_induce_lessons,
    induce_lessons_with_llm,
    verify_lessons_from_results,
    verify_lessons_paired_intervention,
    verify_lessons_real_paired_rollout,
)
from becra.lessons import (
    lesson_matches,
    lesson_matches_for_planning,
    rank_lessons_by_meta_overlap,
    lessons_for_paper_planning,
    top_k_lessons_for_planning,
    load_lessons_with_seed,
    read_lessons,
    write_lessons,
)
from becra.llm import OpenAICompatibleClient
from becra.meta_features import compute_meta_features, summarize_meta
from becra.planner import (
    llm_plan_toolchain,
    paper_plan_toolchain,
    paper_planning_context,
    plan_toolchain,
    rank_toolchains,
    _pick_by_lesson_votes,
)
from becra.prompt_templates import lesson_guided_planning_prompt
from becra.runner import run_forecast, shell_command
from becra.ucb import ContrastAwareUCB, reward_from_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="BECRA long-term forecasting reconstruction")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_profile = sub.add_parser("profile", help="Compute dataset meta-features")
    add_dataset_args(p_profile)
    p_profile.add_argument("--output", type=Path, default=OUTPUT_ROOT / "meta_features.json")

    p_plan = sub.add_parser("plan", help="Plan BECRA toolchains from meta-features and seed lessons")
    add_dataset_args(p_plan)
    add_lesson_args(p_plan)
    add_planner_args(p_plan)
    _add_lesson_match_args(p_plan)
    p_plan.add_argument("--show-command", action="store_true")
    p_plan.add_argument("--pred-len", type=int, default=96)
    p_plan.add_argument("--cpu", action="store_true")
    p_plan.add_argument("--quick", action="store_true")

    p_run = sub.add_parser("run", help="Run planned or selected toolchains")
    add_dataset_args(p_run)
    add_run_args(p_run)
    add_lesson_args(p_run)
    add_planner_args(p_run)
    _add_lesson_match_args(p_run)
    p_run.add_argument("--toolchain", choices=all_toolchain_names(), default=None)

    p_sweep = sub.add_parser("sweep", help="Run multiple toolchains and horizons")
    add_dataset_args(p_sweep)
    add_run_args(p_sweep)
    p_sweep.add_argument("--toolchains", nargs="+", choices=all_toolchain_names(), default=all_toolchain_names())

    p_explore = sub.add_parser("explore", help="Contrast-aware UCB exploration over toolchains")
    add_dataset_args(p_explore)
    add_run_args(p_explore)
    p_explore.add_argument("--rounds", type=int, default=2)
    p_explore.add_argument("--k-pos", type=int, default=1)
    p_explore.add_argument("--k-neg", type=int, default=1)
    p_explore.add_argument("--ucb-lambda", type=float, default=0.5)
    p_explore.add_argument(
        "--ucb-unpulled-bonus",
        type=float,
        default=0.0,
        help="Score boost for arms with zero updates (paper Alg. 1 uses 0).",
    )
    p_explore.add_argument(
        "--ucb-exploration-c",
        type=float,
        default=0.0,
        help="Optional UCB1 sqrt(log(t)/(n+1)) exploration term (e.g. 0.3).",
    )
    p_explore.add_argument(
        "--explore-stratify-forecasting",
        action="store_true",
        help="Round-robin positive picks across forecast-model families (TimeXer, MultiPatchFormer, ...).",
    )
    p_explore.add_argument("--T-noise", type=int, default=1, help="Number of leading rounds in which diversity noise is added to S(a)")
    p_explore.add_argument("--log-path", type=Path, default=OUTPUT_ROOT / "explore_log.jsonl")

    p_prompts = sub.add_parser("make-prompt", help="Write lesson-guided planning prompts for selected datasets")
    add_dataset_args(p_prompts)
    add_lesson_args(p_prompts)
    p_prompts.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "prompts")

    p_induce = sub.add_parser("induce", help="Build contrastive lesson prompts and induce candidate lessons from results CSV")
    add_dataset_args(p_induce)
    p_induce.add_argument("--results-csv", type=Path, default=OUTPUT_ROOT / "long_term_results.csv")
    p_induce.add_argument("--prompt-dir", type=Path, default=OUTPUT_ROOT / "lesson_induction_prompts")
    p_induce.add_argument(
        "--llm-raw-dir",
        type=Path,
        default=OUTPUT_ROOT / "lesson_induction_llm_raw",
        help="Save per-pair LLM prompts/responses and dedupe report (induce --use-llm)",
    )
    p_induce.add_argument("--lesson-json", type=Path, default=OUTPUT_ROOT / "candidate_lessons.json")
    p_induce.add_argument("--threshold-quantile", type=float, default=0.5)
    p_induce.add_argument(
        "--induction-mode",
        choices=["paper"],
        default="paper",
        help="Strategy-level contrastive lesson induction.",
    )
    p_induce.add_argument(
        "--max-induction-groups",
        "--max-induction-pairs",
        type=int,
        default=10,
        dest="max_induction_groups",
        help="Max strategy contrasts (paper) or (toolchain,pred_len) groups (legacy) sent to LLM.",
    )
    p_induce.add_argument(
        "--max-groups-per-toolchain",
        "--max-pairs-per-toolchain",
        type=int,
        default=4,
        dest="max_groups_per_toolchain",
    )
    p_induce.add_argument(
        "--max-lessons-per-group",
        type=int,
        default=7,
        help="Legacy: max lessons per group. Paper: min stage-lesson budget (1 strategy + up to this many stage tools).",
    )
    p_induce.add_argument("--use-llm", action="store_true", help="Call the configured LLM (BECRA_LLM_API_KEY) to induce lessons; falls back to heuristic if unavailable")
    p_induce.add_argument("--no-heuristic", action="store_true", help="Skip the heuristic induction even when LLM induction succeeds")
    p_induce.add_argument("--overwrite", action="store_true", help="Overwrite the lesson JSON instead of appending+deduping")

    p_verify = sub.add_parser("verify", help="Filter candidate lessons using completed sweep/explore results")
    add_dataset_args(p_verify)
    p_verify.add_argument("--results-csv", type=Path, default=OUTPUT_ROOT / "long_term_results.csv")
    p_verify.add_argument(
        "--lesson-json",
        type=Path,
        default=OUTPUT_ROOT / "candidate_lessons.json",
        help="Candidate lessons to verify (not injected into the planner pool together).",
    )
    p_verify.add_argument(
        "--background-lessons-json",
        nargs="+",
        type=Path,
        default=[],
        help="Optional verified lessons merged into the background pool with seed (default: seed only).",
    )
    p_verify.add_argument("--no-seed", action="store_true", help="Exclude seed lessons from the background pool.")
    p_verify.add_argument(
        "--verify-mode",
        choices=["paper"],
        default="paper",
        help="Paired verification with lesson pool L vs L\\{phi_k}.",
    )
    p_verify.add_argument(
        "--verify-success-metric",
        choices=["paired", "top_alpha", "top_alpha_paired"],
        default="top_alpha_paired",
        help=(
            "top_alpha_paired (default): top-alpha with paired tie-break when chains differ but "
            "both in top quantile. top_alpha: paper quantile only. paired: direct reward compare."
        ),
    )
    p_verify.add_argument(
        "--verify-planner",
        choices=["rule", "llm"],
        default="rule",
        help="Planner for Alg. 3 verify episodes (default rule for distinguishable L vs L\\{phi_k}).",
    )
    p_verify.add_argument(
        "--verify-planning-top-k",
        type=int,
        default=8,
        help="Inject only top-K strict-matched lessons per verify episode (0 = legacy full pool).",
    )
    p_verify.add_argument(
        "--verify-pool-scope",
        choices=["all", "toolchain"],
        default="all",
        help="all: full candidate library L. toolchain: restrict L to the candidate's toolchain.",
    )
    p_verify.add_argument(
        "--verify-allow-identical-plans",
        action="store_true",
        help="Count episodes where treat and ctrl plan the same toolchain (legacy behavior).",
    )
    p_verify.add_argument("--verified-json", type=Path, default=OUTPUT_ROOT / "verified_lessons.json")
    p_verify.add_argument("--top-alpha", type=float, default=0.35)
    p_verify.add_argument("--min-effect", type=float, default=0.0)
    p_verify.add_argument(
        "--min-verify-support",
        type=int,
        default=2,
        help="Minimum paired episodes required for PASS (uses all episodes run when fewer than this target).",
    )
    p_verify.add_argument(
        "--target-verify-datasets",
        type=int,
        default=4,
        help="Run paired verify on up to this many meta-matched datasets per lesson (all available if fewer).",
    )
    p_verify.add_argument("--paired-rollout", action="store_true", help="Use paper Alg.3 paired controlled intervention (with vs without phi_k; Contradict for negative lessons)")
    p_verify.add_argument("--rollout-mode", choices=["csv", "execute"], default="csv", help="csv: look up y from the reference results CSV (cheap). execute: actually run forecasts for the planner's chosen strategies (real paired rollout).")
    p_verify.add_argument("--rollout-cache", type=Path, default=OUTPUT_ROOT / "paired_rollout_cache.json", help="JSON cache for executed (dataset, pred_len, toolchain) -> reward")
    p_verify.add_argument("--pred-lens", nargs="+", type=int, default=LONG_TERM_HORIZONS, help="Horizons used only in --rollout-mode execute")
    p_verify.add_argument("--seq-len", type=int, default=96)
    p_verify.add_argument("--cpu", action="store_true")
    p_verify.add_argument("--gpu", type=int, default=0)
    p_verify.add_argument("--gpus", nargs="+", type=int, default=None)
    p_verify.add_argument("--max-parallel", type=int, default=None)
    p_verify.add_argument("--quick", action="store_true")
    p_verify.add_argument("--timeout", type=int, default=None)
    p_verify.add_argument("--python-bin", type=str, default=None)
    p_verify.add_argument("--overwrite", action="store_true", help="Overwrite the verified lesson JSON instead of appending+deduping")
    p_verify.add_argument(
        "--llm-strict",
        action="store_true",
        help="With --planner llm: skip episodes when the API fails (default: fall back to rule planner).",
    )
    add_planner_args(p_verify)

    args = parser.parse_args()
    if args.mode == "profile":
        cmd_profile(args)
    elif args.mode == "plan":
        cmd_plan(args)
    elif args.mode == "run":
        cmd_run(args)
    elif args.mode == "sweep":
        cmd_sweep(args)
    elif args.mode == "explore":
        cmd_explore(args)
    elif args.mode == "make-prompt":
        cmd_make_prompt(args)
    elif args.mode == "induce":
        cmd_induce(args)
    elif args.mode == "verify":
        cmd_verify(args)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", default=all_dataset_names(), choices=all_dataset_names())


def add_lesson_args(
    parser: argparse.ArgumentParser,
    default_paths: list[Path] | None = None,
    help_pool: str = "Lesson JSON file(s) to merge on top of the seed lessons. Later files override earlier on lesson_id collision.",
) -> None:
    parser.add_argument("--lessons-json", nargs="+", type=Path, default=default_paths or [], help=help_pool)
    parser.add_argument("--no-seed", action="store_true", help="Exclude becra.lessons.SEED_VERIFIED_LESSONS from the planner pool.")


def add_planner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--planner",
        choices=["rule", "llm"],
        default="rule",
        help="rule: deterministic BECRA planner (heuristics + lesson scores). llm: lesson-guided in-context LLM planner (paper Alg.4). Falls back to rule when no API key.",
    )
    parser.add_argument(
        "--no-rewiden-strategy",
        action="store_true",
        help="Disable explore-based widening of strategy-lesson activation before planning.",
    )
    parser.add_argument(
        "--explore-rewiden-csv",
        type=Path,
        default=None,
        help="Explore CSV for strategy activation widening (default: infer from <holdout>_zeroshot.csv sibling).",
    )
    parser.add_argument(
        "--rewiden-source-datasets",
        nargs="+",
        default=None,
        help="Source datasets used in explore when re-widening strategy lessons (default: infer from explore CSV name).",
    )


_HOLDOUT_EXPLORE_SOURCES: dict[str, list[str]] = {
    "weather": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity"],
    "electricity": ["Weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    "etth1": ["Weather", "ETTh2", "ETTm1", "ETTm2", "Electricity"],
    "etth2": ["Weather", "ETTh1", "ETTm1", "ETTm2", "Electricity"],
    "ettm1": ["Weather", "ETTh1", "ETTh2", "ETTm2", "Electricity"],
    "ettm2": ["Weather", "ETTh1", "ETTh2", "ETTm1", "Electricity"],
}


def _infer_rewiden_sources(explore_csv: Path) -> list[str]:
    key = explore_csv.name.replace("_explore_results.csv", "").lower()
    return list(_HOLDOUT_EXPLORE_SOURCES.get(key, []))


def _load_lesson_pool(args: argparse.Namespace) -> list:
    include_seed = not getattr(args, "no_seed", False)
    paths = getattr(args, "lessons_json", []) or []
    return load_lessons_with_seed(paths=paths, include_seed=include_seed)


def _planning_rewiden_context(args: argparse.Namespace) -> tuple[Path | None, list[str]]:
    if getattr(args, "no_rewiden_strategy", False):
        return None, []
    explore_csv = getattr(args, "explore_rewiden_csv", None)
    if explore_csv is None:
        results_csv = getattr(args, "results_csv", None)
        if results_csv:
            p = Path(results_csv)
            if p.name.endswith("_zeroshot.csv"):
                cand = p.parent / p.name.replace("_zeroshot.csv", "_explore_results.csv")
                if cand.is_file():
                    explore_csv = cand
    src = list(getattr(args, "rewiden_source_datasets", None) or [])
    if explore_csv and not src:
        src = _infer_rewiden_sources(Path(explore_csv))
    return (Path(explore_csv) if explore_csv else None), src


def _lessons_for_planning(
    lesson_pool: list,
    meta: dict,
    pred_len: int,
    *,
    relaxed: bool = False,
) -> list:
    if relaxed:
        return [
            lesson
            for lesson in lesson_pool
            if lesson_matches_for_planning(meta, lesson, int(pred_len), min_meta_overlap=1)
        ]
    return [lesson for lesson in lesson_pool if lesson_matches(meta, lesson, int(pred_len))]


def _plan_for(
    meta: dict,
    lesson_pool: list,
    planner_mode: str,
    *,
    relaxed_lesson_match: bool = False,
    skip_meta_filter: bool = False,
    llm_retries: int = 3,
):
    if planner_mode == "llm":
        client = OpenAICompatibleClient()
        if not client.available:
            print("[planner] --planner=llm but no API key (BECRA_LLM_API_KEY/OPENAI_API_KEY) found; falling back to rule planner.")
        else:
            result = llm_plan_toolchain(
                meta,
                lessons=lesson_pool,
                client=client,
                retries=llm_retries,
                relaxed_lesson_match=relaxed_lesson_match,
                skip_meta_filter=skip_meta_filter,
            )
            if result is not None:
                return result, "llm"
            print("[planner] LLM planner failed to return a valid strategy; falling back to rule planner.")
    return plan_toolchain(
        meta,
        lessons=lesson_pool,
        relaxed_lesson_match=relaxed_lesson_match,
        skip_meta_filter=skip_meta_filter,
    ), "rule"


def _add_lesson_match_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--relaxed-lesson-match",
        action="store_true",
        help="Planning only: inject lessons with ≥1 overlapping activation key (not full strict match).",
    )
    parser.add_argument(
        "--inject-all-lessons",
        action="store_true",
        help="Planning only: skip meta matching and pass the full lesson pool to the planner (deprecated; prefer --planning-top-k).",
    )
    parser.add_argument(
        "--planning-top-k",
        type=int,
        default=5,
        help="Planning: inject only the top-K lessons by meta activation overlap (default 5). Use 0 to disable top-K selection.",
    )
    parser.add_argument(
        "--planning-llm-retries",
        type=int,
        default=3,
        help="LLM planning JSON retries before rule-planner fallback (default 3).",
    )
    parser.add_argument(
        "--planning-min-overlap",
        type=int,
        default=1,
        help="Require at least this many activation-key matches to use a lesson (champion / top-k modes).",
    )
    parser.add_argument(
        "--planning-mode",
        choices=("paper",),
        default="paper",
        help="Stage-wise lesson-guided planning over the full toolchain library.",
    )
    parser.add_argument(
        "--planning-inject-scope",
        choices=("primary", "matched"),
        default="primary",
        help=(
            "primary: inject only lessons on primary_evidence_toolchain (default). "
            "matched: inject top-k diverse lessons from all strict matches, always including "
            "≥1 primary lesson; LLM may choose a non-primary toolchain."
        ),
    )


def _lessons_for_run(
    lesson_pool: list,
    meta: dict,
    pred_len: int,
    args: argparse.Namespace,
) -> tuple[list, str, bool]:
    """Return (lessons_for_planner, mode_label, skip_meta_filter)."""
    top_k = int(getattr(args, "planning_top_k", 0) or 0)
    min_ov = int(getattr(args, "planning_min_overlap", 1) or 0)
    if top_k > 0:
        selected = top_k_lessons_for_planning(
            lesson_pool, meta, pred_len, top_k, min_overlap=min_ov
        )
        label = f"top-{top_k} overlap>={min_ov}" if min_ov else f"top-{top_k} overlap"
        return selected, label, True
    if getattr(args, "inject_all_lessons", False):
        return list(lesson_pool), "all (no meta filter)", True
    relaxed = getattr(args, "relaxed_lesson_match", False)
    mode = "relaxed (≥1 key)" if relaxed else "strict (all keys)"
    return _lessons_for_planning(lesson_pool, meta, pred_len, relaxed=relaxed), mode, False


def _make_plan_fn(
    planner_mode: str,
    strict: bool = True,
):
    """Return a callable(meta, lessons) -> toolchain_name|None used by verify functions."""
    if planner_mode == "llm":
        client = OpenAICompatibleClient()
        if not client.available:
            raise SystemExit("[verify] --planner=llm requested but no API key (BECRA_LLM_API_KEY/OPENAI_API_KEY) found.")

        def _plan(meta: dict, lessons: list):
            res = llm_plan_toolchain(
                meta,
                lessons=lessons,
                client=client,
                skip_meta_filter=True,
                planning_context=paper_planning_context(),
            )
            if res is not None:
                return res.toolchain.name
            if strict:
                print("[verify] LLM planner returned no valid strategy; skipping episode (strict mode).")
                return None
            print("[verify] LLM planner returned no valid strategy; falling back to lesson-vote rule.")
            return _pick_by_lesson_votes(
                meta,
                lessons,
                set(TOOLCHAINS.keys()),
                note="verify_rule",
            ).toolchain.name

        return _plan

    def _plan(meta: dict, lessons: list):
        return _pick_by_lesson_votes(
            meta,
            lessons,
            set(TOOLCHAINS.keys()),
            note="verify_rule",
        ).toolchain.name

    return _plan


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pred-lens", nargs="+", type=int, default=LONG_TERM_HORIZONS)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="Round-robin GPU ids for parallel training jobs (e.g. --gpus 0 1). Overrides --gpu when set.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max concurrent training subprocesses (default: number of --gpus, or 1).",
    )
    parser.add_argument("--quick", action="store_true", help="Use tiny hyperparameters for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--python-bin", type=str, default=None)
    parser.add_argument("--results-csv", type=Path, default=OUTPUT_ROOT / "long_term_results.csv")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep TSL checkpoint/results/test_results dirs after each run (default: prune once mse/mae are recorded).",
    )


def cmd_profile(args: argparse.Namespace) -> None:
    profiles = {}
    for name in args.datasets:
        spec = get_dataset(name)
        meta = compute_meta_features(spec)
        profiles[name] = meta.to_dict()
        print(f"{name}: {summarize_meta(meta)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profiles, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


def cmd_plan(args: argparse.Namespace) -> None:
    lesson_pool = _load_lesson_pool(args)
    extra = len(args.lessons_json) if args.lessons_json else 0
    print(f"[plan] using {len(lesson_pool)} lesson(s) (seed+{extra} extra file(s)) via planner={args.planner}")
    for name in args.datasets:
        spec = get_dataset(name)
        meta = compute_meta_features(spec).to_dict()
        meta["pred_len"] = int(args.pred_len)
        planning_mode = getattr(args, "planning_mode", "paper")
        top_k = int(getattr(args, "planning_top_k", 5) or 0)
        inject_scope = getattr(args, "planning_inject_scope", "primary")
        llm_lessons, matched = lessons_for_paper_planning(
            lesson_pool, meta, args.pred_len, top_k=top_k, inject_scope=inject_scope
        )
        print(
            f"[plan] pred_len={args.pred_len} paper "
            f"strict_matched={len(matched)} llm_inject={len(llm_lessons)} "
            f"inject_scope={inject_scope}"
        )
        client = OpenAICompatibleClient() if args.planner == "llm" else None
        explore_csv, rewiden_src = _planning_rewiden_context(args)
        selected = paper_plan_toolchain(
            meta,
            lesson_pool,
            client,
            top_k=top_k,
            llm_retries=int(getattr(args, "planning_llm_retries", 3)),
            use_llm=(args.planner == "llm"),
            explore_csv=explore_csv,
            rewiden_source_datasets=rewiden_src,
            inject_scope=inject_scope,
        )
        planner_used = "paper+llm" if args.planner == "llm" else "paper+rule"
        payload = selected.to_dict()
        payload["planner"] = planner_used
        print(json.dumps({"dataset": name, "meta": summarize_meta(meta), "plan": payload}, indent=2, ensure_ascii=False))
        if args.show_command:
            run = run_forecast(
                spec,
                selected.toolchain,
                pred_len=args.pred_len,
                use_gpu=not args.cpu,
                quick=args.quick,
                dry_run=True,
            )
            print(shell_command(run.command))


def cmd_run(args: argparse.Namespace) -> None:
    rows = []
    lesson_pool = _load_lesson_pool(args)
    extra = len(args.lessons_json) if args.lessons_json else 0
    if not args.toolchain:
        mode = getattr(args, "planning_mode", "paper")
        top_k = int(getattr(args, "planning_top_k", 5) or 0)
        print(
            f"[run] planner={args.planner} planning_mode={mode} pool={len(lesson_pool)} "
            f"(seed+{extra} files) top_k={top_k}"
        )
    for name in args.datasets:
        spec = get_dataset(name)
        if args.toolchain:
            toolchain = get_toolchain(args.toolchain)
            tasks = [(spec, toolchain, pred_len) for pred_len in args.pred_lens]
        else:
            tasks = []
            for pred_len in args.pred_lens:
                meta = compute_meta_features(spec).to_dict()
                meta["pred_len"] = int(pred_len)
                top_k = int(getattr(args, "planning_top_k", 5) or 0)
                inject_scope = getattr(args, "planning_inject_scope", "primary")
                llm_lessons, matched = lessons_for_paper_planning(
                    lesson_pool, meta, pred_len, top_k=top_k, inject_scope=inject_scope
                )
                print(
                    f"[run] dataset={name} pred_len={pred_len} "
                    f"strict_matched={len(matched)} llm_inject={len(llm_lessons)} "
                    f"inject_scope={inject_scope}"
                )
                for lesson, overlap, strict in rank_lessons_by_meta_overlap(
                    matched or llm_lessons, meta, pred_len
                ):
                    print(
                        f"[run]   lesson={lesson.lesson_id} overlap={overlap} strict={strict}"
                    )
                client = OpenAICompatibleClient() if args.planner == "llm" else None
                explore_csv, rewiden_src = _planning_rewiden_context(args)
                plan = paper_plan_toolchain(
                    meta,
                    lesson_pool,
                    client,
                    top_k=top_k,
                    llm_retries=int(getattr(args, "planning_llm_retries", 3)),
                    use_llm=(args.planner == "llm"),
                    explore_csv=explore_csv,
                    rewiden_source_datasets=rewiden_src,
                    inject_scope=inject_scope,
                )
                planner_used = "paper+llm" if args.planner == "llm" else "paper+rule"
                toolchain = plan.toolchain
                print(
                    f"[run] dataset={name} pred_len={pred_len} planner={planner_used} "
                    f"-> {toolchain.name} ({plan.reason[:120]}...)"
                )
                tasks.append((spec, toolchain, pred_len))
        rows.extend(_run_tasks_parallel(args, tasks))
    _write_results(args.results_csv, rows)


def cmd_sweep(args: argparse.Namespace) -> None:
    tasks: list[tuple[Any, Any, int]] = []
    for name in args.datasets:
        spec = get_dataset(name)
        for toolchain_name in args.toolchains:
            toolchain = get_toolchain(toolchain_name)
            for pred_len in args.pred_lens:
                tasks.append((spec, toolchain, pred_len))
    rows = _run_tasks_parallel(args, tasks)
    _write_results(args.results_csv, rows)


def _ucb_explore_kwargs(args: argparse.Namespace) -> dict[str, float | bool]:
    return {
        "unpulled_bonus": float(getattr(args, "ucb_unpulled_bonus", 1.0) or 0.0),
        "exploration_c": float(getattr(args, "ucb_exploration_c", 0.0) or 0.0),
        "stratify_forecasting": bool(getattr(args, "explore_stratify_forecasting", False)),
    }


def cmd_explore(args: argparse.Namespace) -> None:
    rows = []
    state_path = args.results_csv.with_suffix(".ucb.json")
    ucb_kw = _ucb_explore_kwargs(args)
    if state_path.exists():
        try:
            ucb = ContrastAwareUCB.from_dict(
                json.loads(state_path.read_text()),
                lam=args.ucb_lambda,
                **ucb_kw,
            )
            print(f"[explore] resumed UCB state from {state_path}")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[explore] could not load {state_path} ({exc}); starting fresh UCB")
            ucb = ContrastAwareUCB(all_toolchain_names(), lam=args.ucb_lambda, **ucb_kw)
    else:
        ucb = ContrastAwareUCB(all_toolchain_names(), lam=args.ucb_lambda, **ucb_kw)
    print(
        f"[explore] UCB arms={len(ucb.arms)} "
        f"unpulled_bonus={ucb.unpulled_bonus} exploration_c={ucb.exploration_c} "
        f"stratify_forecasting={ucb.stratify_forecasting}"
    )
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = args.log_path.open("a")
    ucb_lock = threading.Lock()
    try:
        for round_idx in range(args.rounds):
            selected_names = ucb.select_batch(k_pos=args.k_pos, k_neg=args.k_neg, T_noise=args.T_noise)
            print(f"Round {round_idx + 1}: selected {selected_names}")
            tasks: list[tuple[Any, Any, int]] = []
            for name in args.datasets:
                spec = get_dataset(name)
                for toolchain_name in selected_names:
                    toolchain = get_toolchain(toolchain_name)
                    for pred_len in args.pred_lens:
                        tasks.append((spec, toolchain, pred_len))

            def _after_row(row: dict[str, Any], spec, toolchain, pred_len: int) -> None:
                toolchain_name = toolchain.name
                reward = None
                if row.get("mse") is not None and row.get("mae") is not None:
                    reward = reward_from_metrics(float(row["mse"]), float(row["mae"]))
                    with ucb_lock:
                        ucb.update(toolchain_name, reward)
                        stats = ucb.stats[toolchain_name]
                else:
                    with ucb_lock:
                        stats = ucb.stats[toolchain_name]
                log_fp.write(json.dumps({
                    "round": round_idx + 1,
                    "dataset": spec.name,
                    "toolchain": toolchain_name,
                    "pred_len": pred_len,
                    "mse": row.get("mse"),
                    "mae": row.get("mae"),
                    "reward": reward,
                    "score": stats.mean + args.ucb_lambda * stats.std,
                    "mu": stats.mean,
                    "sigma": stats.std,
                }, ensure_ascii=False) + "\n")
                log_fp.flush()

            round_rows = _run_tasks_parallel(args, tasks, on_complete=_after_row)
            rows.extend(round_rows)
            ok_rows = [r for r in round_rows if r.get("mse") is not None and r.get("mae") is not None]
            if ok_rows:
                _write_results(args.results_csv, ok_rows)
            failed = len(round_rows) - len(ok_rows)
            if failed:
                print(f"[explore] round {round_idx + 1}: {failed} job(s) failed (see log / error column)")
            state_path.write_text(json.dumps(ucb.to_dict(), indent=2))
            print(f"[explore] checkpoint UCB -> {state_path}")
    finally:
        log_fp.close()
    print(f"Wrote {state_path}")
    print(f"Wrote {args.log_path}")


def cmd_make_prompt(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    toolchains = [get_toolchain(name).to_dict() for name in all_toolchain_names()]
    lesson_pool = _load_lesson_pool(args)
    lessons = [lesson.to_dict() for lesson in lesson_pool]
    print(f"[make-prompt] embedding {len(lessons)} lesson(s) in each planning prompt")
    for name in args.datasets:
        meta = compute_meta_features(get_dataset(name)).to_dict()
        prompt = lesson_guided_planning_prompt(meta, lessons, {"paper_tool_library": tool_library_as_dict(), "candidate_toolchains": toolchains})
        out = args.output_dir / f"{name}_planning_prompt.txt"
        out.write_text(prompt)
        print(f"Wrote {out}")


def _load_background_pool(args: argparse.Namespace) -> list:
    return load_lessons_with_seed(
        paths=getattr(args, "background_lessons_json", []) or [],
        include_seed=not getattr(args, "no_seed", False),
    )


def cmd_induce(args: argparse.Namespace) -> None:
    collected = []
    llm_lessons: list = []
    llm_attempted = False

    group_kw = dict(
        max_total_groups=args.max_induction_groups,
        max_groups_per_toolchain=args.max_groups_per_toolchain,
        max_lessons_per_group=args.max_lessons_per_group,
        induction_mode=getattr(args, "induction_mode", "paper"),
    )
    prompt_paths = build_induction_prompts(
        args.results_csv,
        args.prompt_dir,
        dataset_names=args.datasets,
        threshold_quantile=args.threshold_quantile,
        **group_kw,
    )
    print(f"[induce] mode={group_kw['induction_mode']}; wrote {len(prompt_paths)} prompt(s)")

    if args.use_llm:
        llm_attempted = True
        client = OpenAICompatibleClient()
        if not client.available:
            msg = "[induce] --use-llm requested but no API key (BECRA_LLM_API_KEY/OPENAI_API_KEY) found."
            if args.no_heuristic:
                raise SystemExit(msg + " --no-heuristic is set, refusing to silently substitute heuristic lessons.")
            print(msg + " Falling back to heuristic.")
        else:
            llm_lessons, llm_summary = induce_lessons_with_llm(
                args.results_csv,
                dataset_names=args.datasets,
                client=client,
                threshold_quantile=args.threshold_quantile,
                llm_raw_dir=args.llm_raw_dir,
                max_total_groups=args.max_induction_groups,
                max_groups_per_toolchain=args.max_groups_per_toolchain,
                max_lessons_per_group=args.max_lessons_per_group,
                induction_mode=getattr(args, "induction_mode", "paper"),
            )
            print(
                f"[induce] LLM produced {llm_summary.get('raw_lessons', len(llm_lessons))} raw lesson(s), "
                f"accepted {llm_summary.get('accepted', len(llm_lessons))} after dedupe "
                f"({llm_summary.get('requests', 0)} group requests)"
            )
            dropped = llm_summary.get("dropped") or []
            if dropped:
                print(f"[induce] dedupe dropped {len(dropped)} lesson(s); see {args.llm_raw_dir}/dedupe_report.json")
            collected.extend(llm_lessons)

    if args.no_heuristic:
        if llm_attempted and not llm_lessons:
            print("[induce] LLM produced 0 lessons and --no-heuristic is set; writing an empty candidate set instead of silently using heuristic.")
    else:
        heuristic_lessons = heuristic_induce_lessons(
            args.results_csv,
            dataset_names=args.datasets,
            threshold_quantile=args.threshold_quantile,
        )
        print(f"[induce] heuristic produced {len(heuristic_lessons)} candidate lesson(s)")
        collected.extend(heuristic_lessons)

    merged = write_lessons(args.lesson_json, collected, append=not args.overwrite)
    print(f"Wrote {len(prompt_paths)} induction prompt(s) to {args.prompt_dir}")
    mode = "overwrite" if args.overwrite else "append+dedupe by lesson_id"
    print(f"Wrote {len(merged)} candidate lesson(s) (mode={mode}) to {args.lesson_json}")


def cmd_verify(args: argparse.Namespace) -> None:
    candidate_lessons = read_lessons(args.lesson_json)
    verify_mode = getattr(args, "verify_mode", "paper")
    success_metric = getattr(args, "verify_success_metric", "top_alpha")
    if args.paired_rollout:
        pool = _load_background_pool(args)
        llm_strict = getattr(args, "llm_strict", False)
        verify_planner = getattr(args, "verify_planner", "rule")
        if verify_planner == "rule":
            from becra.induction import verify_rule_plan_fn

            plan_fn = verify_rule_plan_fn
        else:
            plan_fn = _make_plan_fn(
                verify_planner,
                strict=llm_strict,
            )
        verify_top_k = int(getattr(args, "verify_planning_top_k", 8) or 8)
        verify_pool_scope = getattr(args, "verify_pool_scope", "all")
        skip_identical = not getattr(args, "verify_allow_identical_plans", False)
        print(
            f"[verify] verify_mode={verify_mode} success_metric={success_metric} "
            f"background_pool={len(pool)} verify_planner={verify_planner} "
            f"plan_top_k={verify_top_k} pool_scope={verify_pool_scope} "
            f"skip_identical={skip_identical}"
        )
        if verify_planner == "llm":
            mode_note = "strict (skip on API failure)" if llm_strict else "fallback to rule on API failure"
            print(f"[verify] planner=llm ({mode_note}); paired rollout mode={args.rollout_mode}")
        else:
            print(f"[verify] planner={verify_planner}; paired rollout mode={args.rollout_mode}")
        verify_kwargs = dict(
            plan_fn=plan_fn,
            verify_mode=verify_mode,
            success_metric=success_metric,
            all_candidates=candidate_lessons,
            verify_planning_top_k=verify_top_k,
            verify_pool_scope=verify_pool_scope,
            skip_identical_plans=skip_identical,
        )
        if args.rollout_mode == "execute":
            verified = verify_lessons_real_paired_rollout(
                candidate_lessons,
                dataset_names=args.datasets,
                pred_lens=args.pred_lens,
                reference_results_csv=args.results_csv,
                candidate_pool=pool,
                top_alpha=args.top_alpha,
                min_effect=args.min_effect,
                min_verify_support=args.min_verify_support,
                target_verify_datasets=args.target_verify_datasets,
                seq_len=args.seq_len,
                use_gpu=not args.cpu,
                gpu=args.gpu,
                quick=args.quick,
                python_bin=args.python_bin,
                timeout=args.timeout,
                cache_path=args.rollout_cache,
                **verify_kwargs,
            )
            mode_label = "paired_real_rollout"
        else:
            verified = verify_lessons_paired_intervention(
                candidate_lessons,
                results_csv=args.results_csv,
                dataset_names=args.datasets,
                candidate_pool=pool,
                top_alpha=args.top_alpha,
                min_effect=args.min_effect,
                min_verify_support=args.min_verify_support,
                target_verify_datasets=args.target_verify_datasets,
                **verify_kwargs,
            )
            mode_label = "paired_csv_lookup"
    else:
        verified = verify_lessons_from_results(
            candidate_lessons,
            args.results_csv,
            dataset_names=args.datasets,
            top_alpha=args.top_alpha,
            min_effect=args.min_effect,
        )
        mode_label = "post_hoc_top_alpha"
    merged = write_lessons(args.verified_json, verified, append=not args.overwrite)
    write_mode = "overwrite" if args.overwrite else "append+dedupe by lesson_id"
    print(f"[verify] mode={mode_label}; new={len(verified)}; total after merge={len(merged)} -> {args.verified_json} ({write_mode})")


def _resolve_gpus(args: argparse.Namespace) -> list[int]:
    if args.cpu:
        return [0]
    if getattr(args, "gpus", None):
        return list(args.gpus)
    return [args.gpu]


def _run_one_on_gpu(args: argparse.Namespace, spec, toolchain, pred_len: int, gpu: int) -> dict[str, Any]:
    local = copy.copy(args)
    local.gpu = gpu
    return _run_one(local, spec, toolchain, pred_len)


def _run_one_safe(args: argparse.Namespace, spec, toolchain, pred_len: int, gpu: int) -> dict[str, Any]:
    try:
        return _run_one_on_gpu(args, spec, toolchain, pred_len, gpu)
    except Exception as exc:
        err = str(exc).strip().splitlines()[-1][:500]
        print(
            f"[run] FAILED dataset={spec.name} toolchain={toolchain.name} pred_len={pred_len} gpu={gpu}: {err}",
            flush=True,
        )
        return {
            "dataset": spec.name,
            "pred_len": pred_len,
            "toolchain": toolchain.name,
            "model": toolchain.model,
            "mse": None,
            "mae": None,
            "command": "",
            "error": err,
        }


def _run_tasks_parallel(
    args: argparse.Namespace,
    tasks: list[tuple[Any, Any, int]],
    on_complete: Callable[[dict[str, Any], Any, Any, int], None] | None = None,
) -> list[dict[str, Any]]:
    gpus = _resolve_gpus(args)
    max_workers = getattr(args, "max_parallel", None) or (len(gpus) if not args.cpu else 1)
    if max_workers <= 1 or len(tasks) <= 1:
        rows = []
        for idx, (spec, toolchain, pred_len) in enumerate(tasks):
            row = _run_one_safe(args, spec, toolchain, pred_len, gpus[idx % len(gpus)])
            if on_complete:
                on_complete(row, spec, toolchain, pred_len)
            rows.append(row)
        return rows

    print(f"[parallel] {len(tasks)} job(s) on GPU(s) {gpus}, max_workers={max_workers}")
    rows: list[dict[str, Any] | None] = [None] * len(tasks)

    def _work(idx: int, spec, toolchain, pred_len: int) -> None:
        gpu = gpus[idx % len(gpus)]
        row = _run_one_safe(args, spec, toolchain, pred_len, gpu)
        rows[idx] = row
        if on_complete:
            on_complete(row, spec, toolchain, pred_len)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_work, i, spec, toolchain, pred_len) for i, (spec, toolchain, pred_len) in enumerate(tasks)]
        for fut in as_completed(futures):
            fut.result()
    return [r for r in rows if r is not None]


def _run_one(args: argparse.Namespace, spec, toolchain, pred_len: int) -> dict[str, Any]:
    overrides = {
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
    }
    run = run_forecast(
        spec,
        toolchain,
        pred_len=pred_len,
        seq_len=args.seq_len,
        python_bin=args.python_bin,
        use_gpu=not args.cpu,
        gpu=args.gpu,
        quick=args.quick,
        dry_run=args.dry_run,
        force_preprocess=args.force_preprocess,
        overrides=overrides,
        timeout=args.timeout,
        prune_artifacts=not args.keep_checkpoints,
    )
    row = run.to_dict()
    if args.dry_run:
        print(row["command"])
    else:
        print(json.dumps(row, indent=2, ensure_ascii=False))
    return row


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "dataset",
        "pred_len",
        "toolchain",
        "model",
        "mse",
        "mae",
        "command",
        "error",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"Wrote {len(rows)} row(s) -> {path}")


if __name__ == "__main__":
    main()
