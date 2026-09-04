import pytest

from text2sql_feedback.post_training import build_preference_pairs


def test_build_preference_pairs_keeps_clear_best_vs_worst() -> None:
    rollouts = [
        {"prompt_id": "p1", "prompt": "q", "completion": "bad", "reward": 0.1},
        {"prompt_id": "p1", "prompt": "q", "completion": "mid", "reward": 0.6},
        {"prompt_id": "p1", "prompt": "q", "completion": "good", "reward": 1.0},
        {"prompt_id": "p2", "prompt": "q2", "completion": "a", "reward": 0.7},
        {"prompt_id": "p2", "prompt": "q2", "completion": "b", "reward": 0.8},
    ]
    pairs = build_preference_pairs(rollouts)
    assert len(pairs) == 1
    assert pairs[0].chosen == "good"
    assert pairs[0].rejected == "bad"


def test_grpo_advantages_are_normalized_within_prompt() -> None:
    torch = pytest.importorskip("torch")
    from text2sql_feedback.post_training import grpo_group_advantages

    rewards = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    advantages = grpo_group_advantages(rewards)
    assert torch.allclose(advantages[0].mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(advantages[1], torch.zeros(3))
