from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreferencePair:
    prompt: Any
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    prompt_id: str


def build_preference_pairs(
    rollouts: Iterable[dict[str, Any]],
    *,
    min_margin: float = 0.25,
    min_chosen_reward: float = 0.9,
) -> list[PreferencePair]:
    """Turn execution-scored rollouts into one best-vs-worst DPO pair per prompt."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        grouped[str(rollout["prompt_id"])].append(rollout)

    pairs: list[PreferencePair] = []
    for prompt_id, candidates in grouped.items():
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates, key=lambda item: float(item["reward"]))
        worst, best = ranked[0], ranked[-1]
        best_reward = float(best["reward"])
        worst_reward = float(worst["reward"])
        if best_reward < min_chosen_reward or best_reward - worst_reward < min_margin:
            continue
        if best["completion"].strip() == worst["completion"].strip():
            continue
        pairs.append(
            PreferencePair(
                prompt=best["prompt"],
                chosen=best["completion"],
                rejected=worst["completion"],
                chosen_reward=best_reward,
                rejected_reward=worst_reward,
                prompt_id=prompt_id,
            )
        )
    return pairs


def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[Any, Any, Any]:
    """Reference DPO objective over sequence log-probabilities.

    The preference logit is beta times the policy log-ratio improvement over the
    reference log-ratio. This function intentionally exposes the core math that
    high-level trainers hide.
    """

    import torch.nn.functional as functional

    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_log_ratio - reference_log_ratio)
    losses = -(
        (1.0 - label_smoothing) * functional.logsigmoid(logits)
        + label_smoothing * functional.logsigmoid(-logits)
    )
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses.mean(), chosen_reward, rejected_reward


def grpo_group_advantages(rewards: Any, *, epsilon: float = 1e-4) -> Any:
    """Normalize rewards within each prompt group; rewards shape is [batch, group]."""

    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return ((rewards - group_mean) / (group_std + epsilon)).detach()


def clipped_policy_loss(
    new_log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
    mask: Any,
    *,
    clip_epsilon: float = 0.2,
) -> Any:
    """PPO/GRPO clipped surrogate loss over response tokens."""

    import torch

    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    token_loss = -torch.minimum(unclipped, clipped)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1)


def ppo_clipped_value_loss(
    values: Any,
    old_values: Any,
    returns: Any,
    mask: Any,
    *,
    clip_epsilon: float = 0.2,
) -> Any:
    """Clipped critic loss used by PPO; GRPO removes this learned value model."""

    import torch

    clipped_values = old_values + torch.clamp(values - old_values, -clip_epsilon, clip_epsilon)
    loss_unclipped = (values - returns).pow(2)
    loss_clipped = (clipped_values - returns).pow(2)
    token_loss = 0.5 * torch.maximum(loss_unclipped, loss_clipped)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1)
