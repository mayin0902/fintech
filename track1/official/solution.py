import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MoEBlockOptimized(nn.Module):
    """
    选手在该类中实现自己的显存优化版本。
    约束：保持与 MoEBlockBaseline 数学定义等价。

    允许使用 PyTorch、torch.compile、自定义 CUDA extension、Triton kernel。
    不允许改变参数含义、路由定义、输出定义或梯度数学定义。

    参数结构必须与 baseline 一致，以支持 load_state_dict 加载权重：
    - self.experts.gate_up_proj: [num_experts, 2*moe_intermediate_size, hidden_size]
    - self.experts.down_proj: [num_experts, hidden_size, moe_intermediate_size]
    - self.shared_expert.gate_proj.weight: [intermediate_size, hidden_size]
    - self.shared_expert.up_proj.weight: [intermediate_size, hidden_size]
    - self.shared_expert.down_proj.weight: [hidden_size, intermediate_size]
    - self.gate.weight: [num_experts, hidden_size]
    - self.post_norm.weight: [hidden_size]
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size)
        )

        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.moe_intermediate_size,
                self.hidden_size,
            )
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            )
        )

        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.shared_expert.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.shared_expert.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

        self.post_norm = nn.Module()
        self.post_norm.weight = nn.Parameter(torch.ones(self.hidden_size))

        self.shared_chunk_size = int(os.getenv("MOE_SHARED_CHUNK_SIZE", "131072"))
        self.expert_chunk_size = int(os.getenv("MOE_EXPERT_CHUNK_SIZE", "131072"))
        self.use_checkpoint = os.getenv("MOE_USE_CHECKPOINT", "0") != "0"

    def _checkpoint(self, fn, *args):
        if not self.training or not self.use_checkpoint:
            return fn(*args)
        if not any(torch.is_tensor(arg) and arg.requires_grad for arg in args):
            return fn(*args)
        return checkpoint(
            fn,
            *args,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def _route(self, hidden_flat):
        router_logits = F.linear(hidden_flat, self.gate.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_weights, selected_experts = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(router_logits.dtype)
        return top_k_weights, selected_experts

    def _shared_chunk(self, hidden_chunk):
        gate = F.linear(hidden_chunk, self.shared_expert.gate_proj.weight)
        up = F.linear(hidden_chunk, self.shared_expert.up_proj.weight)
        return F.linear(
            F.silu(gate) * up,
            self.shared_expert.down_proj.weight,
        )

    def _shared_forward(self, hidden_flat):
        if hidden_flat.shape[0] <= self.shared_chunk_size:
            return self._checkpoint(self._shared_chunk, hidden_flat)

        shared_output = torch.empty_like(hidden_flat)
        for start in range(0, hidden_flat.shape[0], self.shared_chunk_size):
            end = min(start + self.shared_chunk_size, hidden_flat.shape[0])
            shared_output[start:end] = self._checkpoint(
                self._shared_chunk,
                hidden_flat[start:end],
            )
        return shared_output

    def _expert_chunk(self, hidden_chunk, expert_idx):
        gate_up = F.linear(hidden_chunk, self.experts.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(
            F.silu(gate) * up,
            self.experts.down_proj[expert_idx],
        )

    def _experts_forward(self, hidden_flat, selected_experts, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_flat)

        with torch.no_grad():
            expert_hit = torch.unique(selected_experts, sorted=True)

        expert_by_topk = selected_experts.transpose(0, 1)
        for expert_idx in expert_hit.tolist():
            top_k_pos, token_idx = torch.where(expert_by_topk == expert_idx)
            if token_idx.numel() == 0:
                continue

            for start in range(0, token_idx.numel(), self.expert_chunk_size):
                end = min(start + self.expert_chunk_size, token_idx.numel())
                token_chunk = token_idx[start:end]
                topk_chunk = top_k_pos[start:end]
                current_state = hidden_flat[token_chunk]
                current_hidden_states = self._checkpoint(
                    self._expert_chunk,
                    current_state,
                    expert_idx,
                )
                current_hidden_states = (
                    current_hidden_states
                    * top_k_weights[token_chunk, topk_chunk, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_chunk,
                    current_hidden_states.to(final_hidden_states.dtype),
                )

        return final_hidden_states

    def _post_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        return self.post_norm.weight * hidden_states.to(input_dtype)

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)

        top_k_weights, selected_experts = self._route(hidden_flat)
        routed_output = self._experts_forward(
            hidden_flat,
            selected_experts,
            top_k_weights,
        )
        shared_output = self._shared_forward(hidden_flat)

        combined = routed_output + shared_output
        output = self._post_norm(combined)
        return output.reshape(bsz, seq_len, hidden_size)
