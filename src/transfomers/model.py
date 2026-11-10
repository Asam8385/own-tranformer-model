from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass
class GPTConfig:
    vocab_size: int = 261
    max_seq_len: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len is too small")

    def to_dict(self):
        return asdict(self)


# ============================================================
# CAUSAL MULTI-HEAD SELF-ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        # A single projection produces Q, K, and V together.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower triangular mask: True = visible, False = future token.
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
        mask = mask.view(1, 1, config.max_seq_len, config.max_seq_len)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C], where C = d_model.
        batch_size, seq_len, channels = x.shape

        # QKV projection: [B, T, C] -> [B, T, 3*C].
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Rearrange from [B, T, H, D] to [B, H, T, D].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention: scores = (Q @ K^T) / sqrt(D).
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # Block attention to future tokens.
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # Convert scores to attention weights, then weight the values.
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        output = attention_weights @ v

        # Join all heads: [B, H, T, D] -> [B, T, C].
        output = output.transpose(1, 2)
        output = output.contiguous().view(batch_size, seq_len, channels)

        # Learned output projection and residual-branch dropout.
        output = self.out_proj(output)
        output = self.resid_dropout(output)
        return output


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dimension = config.ff_mult * config.d_model

        self.network = nn.Sequential(
            nn.Linear(config.d_model, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK (PRE-LAYERNORM)
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


# ============================================================
# COMPLETE GPT MODEL
# ============================================================

class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        # Token embeddings and learned position embeddings.
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # A stack of decoder-only Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization and vocabulary projection (LM head).
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tie the LM head weights to the input token embedding weights.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        # Add token and position embeddings.
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        # Forward pass through all Transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Final normalization and LM head: [B, T, C] -> [B, T, vocab_size].
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Only compute loss when targets are provided.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_token_id: int = 2,
        suppress_token_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        self.eval()
        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        for _ in range(max_new_tokens):
            # Use only the most recent max_seq_len tokens.
            context = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(context)
            logits = logits[:, -1, :]

            # Prevent sampling selected special tokens.
            if suppress_token_ids:
                logits[:, suppress_token_ids] = -float("inf")

            if temperature == 0:
                # Greedy decoding.
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature scaling.
                logits = logits / max(temperature, 1e-8)

                # Top-k filtering.
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    cutoff = values[:, [-1]]
                    logits = torch.where(logits < cutoff, -float("inf"), logits)

                # Nucleus (top-p) filtering.
                if top_p and 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probabilities = F.softmax(sorted_logits, dim=-1)
                    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
                    remove = cumulative_probabilities > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False

                    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                    filtered_logits = torch.full_like(logits, -float("inf"))
                    filtered_logits.scatter_(1, sorted_indices, sorted_logits)
                    logits = filtered_logits

                # Convert logits to probabilities and sample a token.
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            # Append the new token and stop at EOS.
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if torch.all(next_token.squeeze(-1) == eos_token_id):
                break

        return input_ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
