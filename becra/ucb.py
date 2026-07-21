from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# Prior uncertainty for never-evaluated arms. Paper Alg. 1 ranks U by sigma(a);
# an unobserved arm must outrank already-measured empirical stds so the
# high-uncertainty slice actually covers unexplored toolchains.
UNPULLED_PRIOR_STD = 2.0


@dataclass
class ArmStats:
    values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        n = len(self.values)
        if n == 0:
            return float(UNPULLED_PRIOR_STD)
        if n == 1:
            return 1.0
        return float(np.std(self.values))

    @property
    def n(self) -> int:
        return len(self.values)

    def update(self, value: float) -> None:
        self.values.append(float(value))


def forecasting_model_from_arm(arm: str) -> str:
    """Forecast model slug from toolchain name, e.g. ..._standard_timexer -> timexer."""
    if "_standard_" in arm:
        return arm.rsplit("_standard_", 1)[-1]
    return arm


def preprocessing_depth_from_arm(arm: str) -> int:
    """Count active preprocessing stages in slug imp_anom_none_decomp_standard_model.

    Lower depth = fewer active imputation/anomaly/decomposition stages. Used by
    stratified explore to try plain baselines before heavier preprocess variants
    within the same forecast-model family (generic grid policy, not dataset seeds).
    """
    parts = arm.split("_")
    if len(parts) != 6 or parts[2] != "none" or parts[4] != "standard":
        return 99
    imp, anom, _, decomp, _, _ = parts
    depth = 0
    if imp != "none":
        depth += 1
    if anom != "none":
        depth += 1
    if decomp != "none":
        depth += 1
    return depth


class ContrastAwareUCB:
    """Paper Alg. 1: S(a) = mu(a) + lambda*sigma(a), with early-round diversity noise and
    a DiverseMerge over high-uncertainty and low-score arms to populate the negative slice.

    Optional extras (no fixed toolchain seed list):
    - unpulled_bonus: score boost for arms never updated (implementation prior; paper uses sigma).
    - exploration_c: UCB1-style sqrt(log(t)/(n+1)) term for arms with few pulls.
    - stratify_forecasting: round-robin across forecast-model families when filling positives,
      preferring never/least-tried arms within each family so contrastive evidence spans models.
  """

    def __init__(
        self,
        arms: Iterable[str],
        lam: float = 0.5,
        epsilon: float = 0.02,
        seed: int = 2026,
        *,
        unpulled_bonus: float = 1.0,
        exploration_c: float = 0.0,
        stratify_forecasting: bool = False,
    ):
        self.arms = list(arms)
        self.lam = lam
        self.epsilon = epsilon
        self.unpulled_bonus = float(unpulled_bonus)
        self.exploration_c = float(exploration_c)
        self.stratify_forecasting = bool(stratify_forecasting)
        self.rng = np.random.default_rng(seed)
        self.stats = {arm: ArmStats() for arm in self.arms}
        self.round = 0

    def scores(self, noisy: bool = False) -> dict[str, float]:
        total_pulls = sum(self.stats[arm].n for arm in self.arms)
        log_t = math.log(max(total_pulls, 1) + 1.0)
        scores: dict[str, float] = {}
        for arm in self.arms:
            n = self.stats[arm].n
            s = self.stats[arm].mean + self.lam * self.stats[arm].std
            if self.unpulled_bonus > 0.0 and n == 0:
                s += self.unpulled_bonus
            if self.exploration_c > 0.0:
                s += self.exploration_c * math.sqrt(log_t / (n + 1.0))
            scores[arm] = s
        if noisy:
            for arm in scores:
                scores[arm] += float(self.rng.uniform(-self.epsilon, self.epsilon))
        return scores

    def _rank_key(self, arm: str, scores: dict[str, float]) -> tuple[float, float]:
        """Primary: score; secondary: random tie-break (not alphabetical arm name)."""
        return (scores[arm], float(self.rng.random()))

    def _uncertainty_key(self, arm: str) -> tuple[float, int, float]:
        """Paper U = TopK by sigma: prefer high sigma, then fewer pulls, then random."""
        return (self.stats[arm].std, -self.stats[arm].n, float(self.rng.random()))

    def select_batch(
        self,
        k_pos: int = 1,
        k_neg: int = 1,
        T_noise: int = 1,
        early_noise: bool | None = None,
    ) -> list[str]:
        """Advance the internal round counter and return the next batch.

        T_noise controls the number of leading rounds in which ``epsilon`` diversity noise is
        added to S(a). ``early_noise`` is kept for backward compatibility and overrides T_noise
        when supplied.
        """
        self.round += 1
        if early_noise is None:
            apply_noise = self.round <= T_noise
        else:
            apply_noise = bool(early_noise)
        scores = self.scores(noisy=apply_noise)
        ranked_high = sorted(self.arms, key=lambda arm: self._rank_key(arm, scores), reverse=True)
        ranked_low = sorted(self.arms, key=lambda arm: self._rank_key(arm, scores))
        ranked_uncertain = sorted(self.arms, key=self._uncertainty_key, reverse=True)

        if self.stratify_forecasting and k_pos > 0 and self.round <= 5:
            positives = _coverage_bootstrap_positives(
                self.arms, self.stats, scores, k_pos, self.rng, round_num=self.round
            )
        elif self.stratify_forecasting and k_pos > 0:
            positives = _stratified_positives(self.arms, scores, k_pos, self.stats, self.rng)
            positives = _ensure_unpulled_in_positives(
                positives, self.arms, self.stats, scores, k_pos, self.rng
            )
        else:
            positives = []
            for arm in ranked_high:
                if len(positives) >= k_pos:
                    break
                _append_unique(positives, arm)
        negatives = _diverse_merge(ranked_uncertain, ranked_low, k_neg, exclude=positives)
        return positives + negatives

    def update(self, arm: str, reward: float) -> None:
        self.stats[arm].update(reward)

    def to_dict(self) -> dict[str, list[float]]:
        return {arm: self.stats[arm].values for arm in self.arms}

    @classmethod
    def from_dict(
        cls,
        values: dict[str, list[float]],
        lam: float = 0.5,
        epsilon: float = 0.02,
        seed: int = 2026,
        **kwargs: float | bool,
    ) -> "ContrastAwareUCB":
        obj = cls(values.keys(), lam=lam, epsilon=epsilon, seed=seed, **kwargs)
        for arm, history in values.items():
            obj.stats[arm].values = [float(v) for v in history]
        return obj


def reward_from_metrics(mse: float, mae: float) -> float:
    return float(1.0 / (mse + mae + 1e-8))


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _family_shallowest_unpulled(
    arms: list[str],
    stats: dict[str, ArmStats],
    scores: dict[str, float],
    rng: np.random.Generator,
    *,
    exclude: set[str] | None = None,
    rank: int = 0,
) -> dict[str, str]:
    """Map each forecast family to its rank-th shallowest never-tried arm (if any)."""
    skip = exclude or set()
    by_model: dict[str, list[str]] = defaultdict(list)
    for arm in arms:
        if stats[arm].n == 0 and arm not in skip:
            by_model[forecasting_model_from_arm(arm)].append(arm)
    picks: dict[str, str] = {}
    for model, family_arms in by_model.items():
        ranked = sorted(
            family_arms,
            key=lambda arm: (
                preprocessing_depth_from_arm(arm),
                -scores[arm],
                float(rng.random()),
            ),
        )
        if rank < len(ranked):
            picks[model] = ranked[rank]
    return picks


def _family_tier_unpulled(
    arms: list[str],
    stats: dict[str, ArmStats],
    scores: dict[str, float],
    rng: np.random.Generator,
    *,
    min_depth: int,
) -> dict[str, str]:
    """Map each forecast family to its shallowest unpulled arm at or above ``min_depth``."""
    by_model: dict[str, list[str]] = defaultdict(list)
    for arm in arms:
        if stats[arm].n == 0 and preprocessing_depth_from_arm(arm) >= min_depth:
            by_model[forecasting_model_from_arm(arm)].append(arm)
    picks: dict[str, str] = {}
    for model, family_arms in by_model.items():
        picks[model] = min(
            family_arms,
            key=lambda arm: (
                preprocessing_depth_from_arm(arm),
                -scores[arm],
                float(rng.random()),
            ),
        )
    return picks


def _ensure_unpulled_in_positives(
    positives: list[str],
    arms: list[str],
    stats: dict[str, ArmStats],
    scores: dict[str, float],
    k_pos: int,
    rng: np.random.Generator,
) -> list[str]:
    """Reserve positive slots for never-tried arms while any remain.

    Generic grid-coverage policy (no named toolchain seeds): inject up to two
    family-stratified unpulled arms per batch so plain baselines and deeper
    preprocess variants both receive contrastive trials within eight rounds.
    """
    if k_pos <= 0 or not positives:
        return positives
    out = list(positives)
    budget = 2 if k_pos >= 4 else 1
    remaining_unpulled = sum(1 for arm in arms if stats[arm].n == 0)
    target = min(budget, k_pos, remaining_unpulled)
    while sum(1 for arm in out if stats[arm].n == 0) < target:
        family_picks = _family_shallowest_unpulled(arms, stats, scores, rng, exclude=set(out))
        if not family_picks:
            break
        unpulled_models_in_out = {
            forecasting_model_from_arm(arm) for arm in out if stats[arm].n == 0
        }
        pick = None
        for model in sorted(family_picks.keys()):
            if model not in unpulled_models_in_out:
                pick = family_picks[model]
                break
        if pick is None:
            pick = min(
                family_picks.values(),
                key=lambda arm: (
                    preprocessing_depth_from_arm(arm),
                    -scores[arm],
                    float(rng.random()),
                ),
            )
        replace = max(out, key=lambda arm: (stats[arm].n, preprocessing_depth_from_arm(arm)))
        out[out.index(replace)] = pick
    return out


def _coverage_bootstrap_positives(
    arms: list[str],
    stats: dict[str, ArmStats],
    scores: dict[str, float],
    k_pos: int,
    rng: np.random.Generator,
    *,
    round_num: int,
) -> list[str]:
    """Opening rounds: shallow unpulled coverage per forecast family (round-robin).

    Round 1-2: shallowest unpulled arm for each family slice (covers all 8 families).
    Round 3: second-shallowest unpulled for the upper family slice (plain+FFT baselines).
    Round 4: shallowest unpulled with depth>=2 for the lower family slice (active preprocess).
    Round 5: shallowest unpulled with depth>=3 for the upper family slice (deeper preprocess).
    """
    models = sorted({forecasting_model_from_arm(arm) for arm in arms})
    if not models:
        return []
    if round_num <= 2:
        rank = 0
        offset = max(0, round_num - 1) * k_pos
        family_picks = _family_shallowest_unpulled(arms, stats, scores, rng, rank=rank)
    elif round_num == 3:
        offset = k_pos
        family_picks = _family_shallowest_unpulled(arms, stats, scores, rng, rank=1)
    elif round_num == 4:
        offset = 0
        family_picks = _family_tier_unpulled(arms, stats, scores, rng, min_depth=2)
    else:
        offset = k_pos
        family_picks = _family_tier_unpulled(arms, stats, scores, rng, min_depth=3)
    positives: list[str] = []
    for model in models[offset : offset + k_pos]:
        if model in family_picks:
            _append_unique(positives, family_picks[model])
    cursor = 0
    while len(positives) < k_pos and models:
        model = models[cursor % len(models)]
        if model in family_picks:
            _append_unique(positives, family_picks[model])
        cursor += 1
    return positives


def _stratified_positives(
    arms: list[str],
    scores: dict[str, float],
    k_pos: int,
    stats: dict[str, ArmStats],
    rng: np.random.Generator,
) -> list[str]:
    """Pick positives by round-robin over forecast-model families.

    Within each family prefer never/least-tried arms, then score. This is a generic
    coverage policy over the combinatorial grid (no named toolchain seeds): every
    model family keeps receiving contrastive trials while unexplored variants remain.
    """
    by_model: dict[str, list[str]] = defaultdict(list)
    for arm in arms:
        by_model[forecasting_model_from_arm(arm)].append(arm)

    def model_pulls(model: str) -> int:
        return sum(stats[arm].n for arm in by_model[model])

    models = sorted(by_model.keys(), key=model_pulls)
    positives: list[str] = []
    while len(positives) < k_pos:
        added = False
        for model in models:
            if len(positives) >= k_pos:
                break
            ranked = sorted(
                by_model[model],
                key=lambda arm: (
                    stats[arm].n,
                    preprocessing_depth_from_arm(arm),
                    -scores[arm],
                    float(rng.random()),
                ),
            )
            for arm in ranked:
                if arm not in positives:
                    positives.append(arm)
                    added = True
                    break
        if not added:
            break
    if len(positives) < k_pos:
        ranked_all = sorted(arms, key=lambda arm: (scores[arm], float(rng.random())), reverse=True)
        for arm in ranked_all:
            if len(positives) >= k_pos:
                break
            _append_unique(positives, arm)
    return positives


def _diverse_merge(uncertain: list[str], low: list[str], k_neg: int, exclude: list[str]) -> list[str]:
    """Alternate-take merge of two ranked lists into a budget of size k_neg, skipping arms in `exclude`.

    Matches the paper's DiverseMerge(U, L, K_neg) intent: the negative slice should mix
    high-uncertainty (U) and low-score (L) arms instead of being dominated by one ranking.
    """
    if k_neg <= 0:
        return []
    out: list[str] = []
    sources = [iter(uncertain), iter(low)]
    exhausted = [False, False]
    cursor = 0
    while len(out) < k_neg and not all(exhausted):
        if not exhausted[cursor]:
            try:
                cand = next(sources[cursor])
            except StopIteration:
                exhausted[cursor] = True
            else:
                if cand not in exclude and cand not in out:
                    out.append(cand)
        cursor = 1 - cursor
    return out
