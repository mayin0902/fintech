#!/usr/bin/env python3
"""Inspect one PPO update kernel without pretending this project trained PPO.

A production PPO run additionally needs rollout collection, old/reference policy
log-probabilities, a trained value model, GAE/returns, weight synchronization,
checkpointing, and distributed resource management. The project deliberately uses
GRPO first because PostgreSQL supplies a verifiable outcome reward and GRPO removes
the critic.
"""

from __future__ import annotations

import torch

from text2sql_feedback.post_training import clipped_policy_loss, ppo_clipped_value_loss


def main() -> None:
    old_log_probs = torch.tensor([[-1.2, -0.8, -1.0]])
    new_log_probs = torch.tensor([[-1.1, -0.9, -0.7]], requires_grad=True)
    advantages = torch.tensor([[0.8, 0.8, 0.8]])
    mask = torch.ones_like(advantages)

    old_values = torch.tensor([[0.2, 0.3, 0.4]])
    values = torch.tensor([[0.25, 0.45, 0.55]], requires_grad=True)
    returns = torch.tensor([[0.9, 0.9, 1.0]])

    policy_loss = clipped_policy_loss(new_log_probs, old_log_probs, advantages, mask)
    value_loss = ppo_clipped_value_loss(values, old_values, returns, mask)
    loss = policy_loss + 0.1 * value_loss
    loss.backward()

    print(
        {
            "policy_loss": round(policy_loss.item(), 6),
            "value_loss": round(value_loss.item(), 6),
            "new_log_prob_grad": new_log_probs.grad.tolist(),
            "value_grad": values.grad.tolist(),
        }
    )


if __name__ == "__main__":
    main()
