from __future__ import annotations

import math

from dataclasses import (
    asdict,
    dataclass
)

from typing import Optional

import torch

import torch.nn as nn

import torch.nn.functional as F


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

@dataclass
class GPTConfig:

    # Tokenizer vocabulary size
    vocab_size: int = 261

    # Maximum context length
    max_seq_len: int = 512

    # Embedding dimension
    d_model: int = 256

    # Number of Transformer blocks
    n_layers: int = 6

    # Number of attention heads
    n_heads: int = 8

    # FFN size multiplier
    ff_mult: int = 4

    # Dropout
    dropout: float = 0.1

    def validate(self) -> None:

        if (
            self.d_model
            % self.n_heads
            != 0
        ):

            raise ValueError(
                "d_model must be divisible "
                "by n_heads"
            )

        if self.max_seq_len < 8:

            raise ValueError(
                "max_seq_len is too small"
            )

    def to_dict(self):

        return asdict(self)


# ==========================================================
# CAUSAL MULTI-HEAD SELF ATTENTION
# ==========================================================

class CausalSelfAttention(
    nn.Module
):

    def __init__(
        self,
        config: GPTConfig
    ) -> None:

        super().__init__()

        config.validate()

        # Number of attention heads
        self.n_heads = (
            config.n_heads
        )

        # Size of each attention head
        self.head_dim = (
            config.d_model
            //
            config.n_heads
        )

        self.d_model = (
            config.d_model
        )

        # --------------------------------------------------
        # Q, K, V projection
        # --------------------------------------------------

        # Instead of using three Linear layers:
        #
        # Wq
        # Wk
        # Wv
        #
        # we calculate all three at once.

        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=False
        )

        # Attention output projection
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.attn_dropout = nn.Dropout(
            config.dropout
        )

        self.resid_dropout = nn.Dropout(
            config.dropout
        )

        # --------------------------------------------------
        # CAUSAL MASK
        # --------------------------------------------------

        # Example:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        #
        # A token cannot look into future tokens.

        mask = torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len,
                dtype=torch.bool
            )
        )

        mask = mask.view(
            1,
            1,
            config.max_seq_len,
            config.max_seq_len
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False
        )

    # ------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # x:
        #
        # [batch, sequence, d_model]

        batch_size, seq_len, channels = (
            x.shape
        )

        # --------------------------------------------------
        # Generate Q K V
        # --------------------------------------------------

        qkv = self.qkv(x)

        # Current:
        #
        # [B,T,3*d_model]
        #
        # Change:
        #
        # [B,T,3,H,D]

        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.n_heads,
            self.head_dim
        )

        # Separate query, key and value

        q, k, v = qkv.unbind(
            dim=2
        )

        # --------------------------------------------------
        # Move attention heads forward
        # --------------------------------------------------

        # [B,T,H,D]
        #
        # ->
        #
        # [B,H,T,D]

        q = q.transpose(
            1,
            2
        )

        k = k.transpose(
            1,
            2
        )

        v = v.transpose(
            1,
            2
        )

        # --------------------------------------------------
        # SCALED DOT PRODUCT ATTENTION
        # --------------------------------------------------

        # Q @ K^T
        #
        # result:
        #
        # [B,H,T,T]

        scores = (
            q
            @
            k.transpose(
                -2,
                -1
            )
        )

        # Scale by sqrt(head dimension)

        scores = (
            scores
            /
            math.sqrt(
                self.head_dim
            )
        )

        # --------------------------------------------------
        # APPLY CAUSAL MASK
        # --------------------------------------------------

        mask = self.causal_mask[
            :,
            :,
            :seq_len,
            :seq_len
        ]

        scores = scores.masked_fill(
            ~mask,
            torch.finfo(
                scores.dtype
            ).min
        )

        # --------------------------------------------------
        # SOFTMAX
        # --------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        attention_weights = (
            self.attn_dropout(
                attention_weights
            )
        )

        # --------------------------------------------------
        # ATTENTION × VALUES
        # --------------------------------------------------

        output = (
            attention_weights
            @
            v
        )

        # [B,H,T,D]
        #
        # ->
        #
        # [B,T,H,D]

        output = output.transpose(
            1,
            2
        )

        # Combine all heads

        output = (
            output
            .contiguous()
            .view(
                batch_size,
                seq_len,
                channels
            )
        )

        # Final output projection

        output = self.out_proj(
            output
        )

        output = self.resid_dropout(
            output
        )

        return output


# ==========================================================
# FEED-FORWARD NETWORK
# ==========================================================

class FeedForward(
    nn.Module
):

    def __init__(
        self,
        config: GPTConfig
    ) -> None:

        super().__init__()

        hidden_dimension = (
            config.ff_mult
            *
            config.d_model
        )

        self.network = nn.Sequential(

            nn.Linear(
                config.d_model,
                hidden_dimension
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dimension,
                config.d_model
            ),

            nn.Dropout(
                config.dropout
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.network(x)


# ==========================================================
# TRANSFORMER BLOCK
# ==========================================================

class TransformerBlock(
    nn.Module
):

    def __init__(
        self,
        config: GPTConfig
    ) -> None:

        super().__init__()

        # Layer normalization before attention
        self.ln1 = nn.LayerNorm(
            config.d_model
        )

        self.attention = (
            CausalSelfAttention(
                config
            )
        )

        # Layer normalization before FFN
        self.ln2 = nn.LayerNorm(
            config.d_model
        )

        self.feed_forward = (
            FeedForward(
                config
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # --------------------------------------------------
        # Attention + residual connection
        # --------------------------------------------------

        x = (
            x
            +
            self.attention(
                self.ln1(x)
            )
        )

        # --------------------------------------------------
        # FFN + residual connection
        # --------------------------------------------------

        x = (
            x
            +
            self.feed_forward(
                self.ln2(x)
            )
        )

        return x


# ==========================================================
# COMPLETE GPT MODEL
# ==========================================================

class MiniGPT(
    nn.Module
):

    def __init__(
        self,
        config: GPTConfig
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        # --------------------------------------------------
        # TOKEN EMBEDDINGS
        # --------------------------------------------------

        self.token_embedding = (
            nn.Embedding(
                config.vocab_size,
                config.d_model
            )
        )

        # --------------------------------------------------
        # POSITION EMBEDDINGS
        # --------------------------------------------------

        self.position_embedding = (
            nn.Embedding(
                config.max_seq_len,
                config.d_model
            )
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        # --------------------------------------------------
        # TRANSFORMER BLOCKS
        # --------------------------------------------------

        self.blocks = nn.ModuleList(

            [
                TransformerBlock(
                    config
                )

                for _ in range(
                    config.n_layers
                )
            ]
        )

        # --------------------------------------------------
        # FINAL NORMALIZATION
        # --------------------------------------------------

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        # --------------------------------------------------
        # LANGUAGE MODEL HEAD
        # --------------------------------------------------

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Initialize weights
        self.apply(
            self._init_weights
        )

        # --------------------------------------------------
        # WEIGHT TYING
        # --------------------------------------------------

        # Input token embeddings and output
        # token classifier share parameters.

        self.lm_head.weight = (
            self.token_embedding.weight
        )

    # ======================================================
    # INITIALIZE MODEL WEIGHTS
    # ======================================================

    @staticmethod
    def _init_weights(
        module: nn.Module
    ) -> None:

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    # ======================================================
    # FORWARD PASS
    # ======================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[
            torch.Tensor
        ] = None
    ):

        batch_size, seq_len = (
            input_ids.shape
        )

        # Context length check

        if (
            seq_len
            >
            self.config.max_seq_len
        ):

            raise ValueError(

                f"Sequence length "
                f"{seq_len} exceeds "
                f"max_seq_len="
                f"{self.config.max_seq_len}"
            )

        # --------------------------------------------------
        # POSITION IDs
        # --------------------------------------------------

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        positions = positions.unsqueeze(
            0
        )

        # --------------------------------------------------
        # EMBEDDINGS
        # --------------------------------------------------

        token_embeddings = (
            self.token_embedding(
                input_ids
            )
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        # Token + position

        x = (
            token_embeddings
            +
            position_embeddings
        )

        x = self.dropout(x)

        # --------------------------------------------------
        # TRANSFORMER
        # --------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # --------------------------------------------------
        # FINAL LAYER NORM
        # --------------------------------------------------

        x = self.final_norm(x)

        # --------------------------------------------------
        # VOCABULARY LOGITS
        # --------------------------------------------------

        logits = self.lm_head(x)

        # logits shape:
        #
        # [batch, sequence, vocab_size]

        loss = None

        # --------------------------------------------------
        # CROSS ENTROPY LOSS
        # --------------------------------------------------

        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(
                    -1,
                    logits.size(-1)
                ),

                targets.reshape(-1),

                # Ignore prompt/padding
                ignore_index=-100
            )

        return (
            logits,
            loss
        )

    # ======================================================
    # TEXT GENERATION
    # ======================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_token_id: int = 2,
        suppress_token_ids:
            Optional[list[int]] = None
    ) -> torch.Tensor:

        self.eval()

        if temperature < 0:

            raise ValueError(
                "temperature must be >= 0"
            )

        # --------------------------------------------------
        # AUTOREGRESSIVE GENERATION
        # --------------------------------------------------

        for _ in range(
            max_new_tokens
        ):

            # If input gets too long,
            # only use latest context.

            context = input_ids[
                :,
                -self.config.max_seq_len:
            ]

            # Forward pass

            logits, _ = self(
                context
            )

            # Only care about prediction
            # from final position.

            logits = logits[
                :,
                -1,
                :
            ]

            # ------------------------------------------------
            # Prevent special structural tokens
            # ------------------------------------------------

            if suppress_token_ids:

                logits[
                    :,
                    suppress_token_ids
                ] = -float(
                    "inf"
                )

            # ------------------------------------------------
            # GREEDY DECODING
            # ------------------------------------------------

            if temperature == 0:

                next_token = (
                    torch.argmax(
                        logits,
                        dim=-1,
                        keepdim=True
                    )
                )

            else:

                # --------------------------------------------
                # TEMPERATURE
                # --------------------------------------------

                logits = (
                    logits
                    /
                    max(
                        temperature,
                        1e-8
                    )
                )

                # --------------------------------------------
                # TOP-K
                # --------------------------------------------

                if (
                    top_k
                    and
                    top_k > 0
                ):

                    k = min(
                        top_k,
                        logits.size(-1)
                    )

                    values, _ = (
                        torch.topk(
                            logits,
                            k
                        )
                    )

                    cutoff = values[
                        :,
                        [-1]
                    ]

                    logits = torch.where(

                        logits < cutoff,

                        -float("inf"),

                        logits
                    )

                # --------------------------------------------
                # TOP-P / NUCLEUS SAMPLING
                # --------------------------------------------

                if (
                    top_p
                    and
                    0 < top_p < 1.0
                ):

                    sorted_logits, sorted_indices = (
                        torch.sort(
                            logits,
                            descending=True
                        )
                    )

                    sorted_probabilities = (
                        F.softmax(
                            sorted_logits,
                            dim=-1
                        )
                    )

                    cumulative_probabilities = (
                        torch.cumsum(
                            sorted_probabilities,
                            dim=-1
                        )
                    )

                    remove = (
                        cumulative_probabilities
                        >
                        top_p
                    )

                    # Keep first token that
                    # crosses threshold.

                    remove[:, 1:] = (
                        remove[
                            :,
                            :-1
                        ].clone()
                    )

                    remove[:, 0] = False

                    sorted_logits = (
                        sorted_logits
                        .masked_fill(
                            remove,
                            -float("inf")
                        )
                    )

                    filtered_logits = (
                        torch.full_like(
                            logits,
                            -float("inf")
                        )
                    )

                    filtered_logits.scatter_(
                        1,
                        sorted_indices,
                        sorted_logits
                    )

                    logits = filtered_logits

                # --------------------------------------------
                # CONVERT LOGITS -> PROBABILITIES
                # --------------------------------------------

                probabilities = F.softmax(
                    logits,
                    dim=-1
                )

                # Sample token

                next_token = (
                    torch.multinomial(
                        probabilities,
                        num_samples=1
                    )
                )

            # ------------------------------------------------
            # ADD PREDICTED TOKEN
            # ------------------------------------------------

            input_ids = torch.cat(

                [
                    input_ids,
                    next_token
                ],

                dim=1
            )

            # ------------------------------------------------
            # STOP AT EOS
            # ------------------------------------------------

            if torch.all(

                next_token.squeeze(-1)
                ==
                eos_token_id

            ):

                break

        return input_ids

    # ======================================================
    # PARAMETER COUNT
    # ======================================================

    def parameter_count(
        self
    ) -> int:

        return sum(

            parameter.numel()

            for parameter
            in self.parameters()
        )