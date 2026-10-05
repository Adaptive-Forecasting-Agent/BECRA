#!/usr/bin/env python
"""BECRA command-line entry point: profile, explore, induce, verify, run."""
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

from becra.config import LONG_TERM_HORIZONS, OUTPUT_ROOT, TOOLCHAINS, all_dataset_names, all_toolchain_names, get_dataset, get_toolchain
from becra.induction import (
    build_induction_prompts,
    induce_lessons_with_llm,
    verify_lessons_real_paired_rollout,
)
from becra.lessons import (
    rank_lessons_by_meta_overlap,
    lessons_for_paper_planning,
    load_lessons,
    read_lessons,
    write_lessons,
)
from becra.llm import OpenAICompatibleClient
from becra.meta_features import compute_meta_features, summarize_meta
from becra.planner import (
    llm_plan_toolchain,
    paper_plan_toolchain,
    paper_planning_context,
    _pick_by_lesson_votes,
)
from becra.runner import run_forecast
from becra.ucb import ContrastAwareUCB, reward_from_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="BECRA long-term forecasting pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_profile = sub.add_parser("profile", help="Compute dataset meta-features")
    add_dataset_args(p_profile)
    p_profile.add_argument("--output", type=Path, default=OUTPUT_ROOT / "meta_features.json")

    p_explore = sub.add_parser("explore", help="Stage 1: contrast-aware UCB exploration over toolchains")
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
        help="Score boost for arms with zero updates.",
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

    p_induce = sub.add_parser("induce", help="Stage 2: induce candidate lessons from contrasting explore results (LLM)")
    add_dataset_args(p_induce)
    p_induce.add_argument("--results-csv", type=Path, default=OUTPUT_ROOT / "explore_results.csv")
    p_induce.add_argument("--prompt-dir", type=Path, default=OUTPUT_ROOT / "lesson_induction_prompts")
    p_induce.add_argument(
        "--llm-raw-dir",
        type=Path,
        default=OUTPUT_ROOT / "lesson_induction_llm_raw",
        help="Save per-group LLM prompts/responses and dedupe report",
    )
    p_induce.add_argument("--lesson-json", type=Path, default=OUTPUT_ROOT / "candidate_lessons.json")
    p_induce.add_argument("--threshold-quantile", type=float, default=0.5)
    p_induce.add_argument(
        "--max-induction-groups",
        type=int,
        default=25,
        help="Max strategy contrasts sent to the LLM.",
    )
    p_induce.add_argument("--max-groups-per-toolchain", type=int, default=4)
    p_induce.add_argument(
        "--max-lessons-per-group",
        type=int,
        default=7,
        help="Stage-lesson budget per contrast (1 strategy lesson + up to this many stage-tool lessons).",
    )
    p_induce.add_argument("--overwrite", action="store_true", help="Overwrite the lesson JSON instead of appending+deduping")

    p_verify = sub.add_parser("verify", help="Stage 3: paired controlled-intervention verification of candidate lessons")
    add_dataset_args(p_verify)
    p_verify.add_argument("--results-csv", type=Path, default=OUTPUT_ROOT / "explore_results.csv")
    p_verify.add_argument("--lesson-json", type=Path, default=OUTPUT_ROOT / "candidate_lessons.json")
    p_verify.add_argument(
        "--verify-success-metric",
        choices=["paired", "top_alpha", "top_alpha_paired"],
        default="top_alpha_paired",
        help=(
            "top_alpha_paired (default): top-alpha with paired tie-break when chains differ but "
            "both in top quantile. top_alpha: quantile only. paired: direct reward compare."
        ),
    )
    p_verify.add_argument(
        "--verify-planning-top-k",
        type=int,
        default=8,
        help="Inject only top-K strict-matched lessons per verify episode.",
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
        default=3,
        help="Run paired verify on up to this many meta-matched datasets per lesson.",
    )
    p_verify.add_argument("--rollout-cache", type=Path, default=OUTPUT_ROOT / "paired_rollout_cache.json", help="JSON cache for executed (dataset, pred_len, toolchain) -> reward")
    p_verify.add_argument("--pred-lens", nargs="+", type=int, default=LONG_TERM_HORIZONS)
    p_verify.add_argument("--seq-len", type=int, default=96)
    p_verify.add_argument("--cpu", action="store_true")
    p_verify.add_argument("--gpu", type=int, default=0)
    p_verify.add_argument("--gpus", nargs="+", type=int, default=None)
    p_verify.add_argument("--max-parallel", type=int, default=None)
    p_verify.add_argument("--quick", action="store_true")
    p_verify.add_argument("--timeout", type=int, default=None)
    p_verify.add_argument("--python-bin", type=str, default=None)
    p_verify.add_argument("--overwrite", action="store_true", help="Overwrite the verified lesson JSON instead of appending+deduping")

    p_run = sub.add_parser("run", help="Stage 4: lesson-guided LLM planning and zero-shot evaluation")
    add_dataset_args(p_run)
    add_run_args(p_run)
    p_run.add_argument("--lessons-json", nargs="+", type=Path, default=[], help="Verified lesson JSON file(s).")
    p_run.add_argument(
        "--planning-top-k",
        type=int,
        default=8,
        help="Inject up to K diverse strict-matched lessons into the planner.",
    )
    p_run.add_argument("--planning-llm-retries", type=int, default=3)
    p_run.add_argument(
        "--planning-inject-scope",
        choices=("primary", "matched"),
        default="matched",
        help=(
            "matched: inject top-k diverse lessons from all strict matches. "
            "primary: inject only lessons on the primary evidence toolchain."
        ),
    )
    p_run.add_argument(
        "--explore-rewiden-csv",
        type=Path,
        default=None,
        help="Explore CSV for strategy activation widening (default: <holdout>_explore_results.csv next to --results-csv).",
    )
    p_run.add_argument(
        "--rewiden-source-datasets",
        nargs="+",
        default=None,
        help="Source datasets used in explore (default: inferred from the explore CSV name).",
    )

    args = parser.parse_args()
    {
        "profile": cmd_profile,
        "explore": cmd_explore,
        "induce": cmd_induce,
        "verify": cmd_verify,
        "run": cmd_run,
    }[args.mode](args)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", default=all_dataset_names(), choices=all_dataset_names())


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
    parser.add_argument("--results-csv", type=Path, default=OUTPUT_ROOT / "results.csv")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep TSL checkpoint/results/test_results dirs after each run (default: prune once mse/mae are recorded).",
    )


_HOLDOUT_EXPLORE_SOURCES: dict[str, list[str]] = {
    "weather": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity"],
    "electricity": ["Weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    "etth1": ["Weather", "ETTh2", "ETTm1", "ETTm2", "Electricity"],
    "etth2": ["Weather", "ETTh1", "ETTm1", "ETTm2", "Electricity"],
    "ettm1": ["Weather", "ETTh1", "ETTh2", "ETTm2", "Electricity"],
    "ettm2": ["Weather", "ETTh1", "ETTh2", "ETTm1", "Electricity"],
}


def _planning_rewiden_context(args: argparse.Namespace) -> tuple[Path | None, list[str]]:
    explore_csv = args.explore_rewiden_csv
    if explore_csv is None:
        p = Path(args.results_csv)
        if p.name.endswith("_zeroshot.csv"):
            cand = p.parent / p.name.replace("_zeroshot.csv", "_explore_results.csv")
            if cand.is_file():
                explore_csv = cand
    src = list(args.rewiden_source_datasets or [])
    if explore_csv and not src:
        key = Path(explore_csv).name.replace("_explore_results.csv", "").lower()
        src = list(_HOLDOUT_EXPLORE_SOURCES.get(key, []))
    return (Path(explore_csv) if explore_csv else None), src


def _verify_plan_fn():
    """Planner used inside verify episodes: LLM planning, lesson-vote rule if the LLM returns nothing."""
    client = OpenAICompatibleClient()
    if not client.available:
        raise SystemExit("[verify] no LLM API key found (set BECRA_LLM_API_KEY).")

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
        print("[verify] LLM planner returned no valid strategy; falling back to lesson-vote rule.")
        return _pick_by_lesson_votes(
            meta,
            lessons,
            set(TOOLCHAINS.keys()),
            note="verify_rule",
        ).toolchain.name

    return _plan


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


def cmd_explore(args: argparse.Namespace) -> None:
    state_path = args.results_csv.with_suffix(".ucb.json")
    ucb_kw = {
        "unpulled_bonus": float(args.ucb_unpulled_bonus or 0.0),
        "exploration_c": float(args.ucb_exploration_c or 0.0),
        "stratify_forecasting": bool(args.explore_stratify_forecasting),
    }
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


def cmd_induce(args: argparse.Namespace) -> None:
    group_kw = dict(
        max_total_groups=args.max_induction_groups,
        max_groups_per_toolchain=args.max_groups_per_toolchain,
        max_lessons_per_group=args.max_lessons_per_group,
    )
    prompt_paths = build_induction_prompts(
        args.results_csv,
        args.prompt_dir,
        dataset_names=args.datasets,
        threshold_quantile=args.threshold_quantile,
        **group_kw,
    )
    print(f"[induce] wrote {len(prompt_paths)} prompt(s)")

    client = OpenAICompatibleClient()
    if not client.available:
        raise SystemExit("[induce] no LLM API key found (set BECRA_LLM_API_KEY).")
    lessons, summary = induce_lessons_with_llm(
        args.results_csv,
        dataset_names=args.datasets,
        client=client,
        threshold_quantile=args.threshold_quantile,
        llm_raw_dir=args.llm_raw_dir,
        **group_kw,
    )
    print(
        f"[induce] LLM produced {summary.get('raw_lessons', len(lessons))} raw lesson(s), "
        f"accepted {summary.get('accepted', len(lessons))} after dedupe "
        f"({summary.get('requests', 0)} group requests)"
    )
    dropped = summary.get("dropped") or []
    if dropped:
        print(f"[induce] dedupe dropped {len(dropped)} lesson(s); see {args.llm_raw_dir}/dedupe_report.json")
    if not lessons:
        print("[induce] LLM produced 0 lessons; writing an empty candidate set.")

    merged = write_lessons(args.lesson_json, lessons, append=not args.overwrite)
    print(f"Wrote {len(prompt_paths)} induction prompt(s) to {args.prompt_dir}")
    mode = "overwrite" if args.overwrite else "append+dedupe by lesson_id"
    print(f"Wrote {len(merged)} candidate lesson(s) (mode={mode}) to {args.lesson_json}")


def cmd_verify(args: argparse.Namespace) -> None:
    candidate_lessons = read_lessons(args.lesson_json)
    print(
        f"[verify] success_metric={args.verify_success_metric} "
        f"plan_top_k={args.verify_planning_top_k} (L vs L\\{{phi_k}}, real paired rollout)"
    )
    verified = verify_lessons_real_paired_rollout(
        candidate_lessons,
        dataset_names=args.datasets,
        pred_lens=args.pred_lens,
        reference_results_csv=args.results_csv,
        candidate_pool=[],
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
        plan_fn=_verify_plan_fn(),
        verify_mode="paper",
        success_metric=args.verify_success_metric,
        all_candidates=candidate_lessons,
        verify_planning_top_k=args.verify_planning_top_k,
        verify_pool_scope="all",
        skip_identical_plans=True,
    )
    merged = write_lessons(args.verified_json, verified, append=not args.overwrite)
    write_mode = "overwrite" if args.overwrite else "append+dedupe by lesson_id"
    print(f"[verify] new={len(verified)}; total after merge={len(merged)} -> {args.verified_json} ({write_mode})")


def cmd_run(args: argparse.Namespace) -> None:
    lesson_pool = load_lessons(args.lessons_json)
    top_k = int(args.planning_top_k or 0)
    inject_scope = args.planning_inject_scope
    print(f"[run] pool={len(lesson_pool)} top_k={top_k} inject_scope={inject_scope}")
    client = OpenAICompatibleClient()
    explore_csv, rewiden_src = _planning_rewiden_context(args)
    rows = []
    for name in args.datasets:
        spec = get_dataset(name)
        tasks = []
        for pred_len in args.pred_lens:
            meta = compute_meta_features(spec).to_dict()
            meta["pred_len"] = int(pred_len)
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
                print(f"[run]   lesson={lesson.lesson_id} overlap={overlap} strict={strict}")
            plan = paper_plan_toolchain(
                meta,
                lesson_pool,
                client,
                top_k=top_k,
                llm_retries=args.planning_llm_retries,
                use_llm=True,
                explore_csv=explore_csv,
                rewiden_source_datasets=rewiden_src,
                inject_scope=inject_scope,
            )
            toolchain = plan.toolchain
            print(
                f"[run] dataset={name} pred_len={pred_len} planner=paper+llm "
                f"-> {toolchain.name} ({plan.reason[:120]}...)"
            )
            tasks.append((spec, toolchain, pred_len))
        rows.extend(_run_tasks_parallel(args, tasks))
    _write_results(args.results_csv, rows)


def _resolve_gpus(args: argparse.Namespace) -> list[int]:
    if args.cpu:
        return [0]
    if getattr(args, "gpus", None):
        return list(args.gpus)
    return [args.gpu]


def _run_one_safe(args: argparse.Namespace, spec, toolchain, pred_len: int, gpu: int) -> dict[str, Any]:
    try:
        local = copy.copy(args)
        local.gpu = gpu
        return _run_one(local, spec, toolchain, pred_len)
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
    fieldnames = ["dataset", "pred_len", "toolchain", "model", "mse", "mae", "command", "error"]
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
