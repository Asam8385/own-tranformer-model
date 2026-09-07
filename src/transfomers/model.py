from __future__ import annotations

import math

from dataclasses import (

   asdict,
   dataclass

)

from typing import Optional

import torch 

import torch.nn as nn

import torch.nn.functional as f

@dataclass
class GPTconfig:

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

class  CasualSelfAttention(
    nn.modules
):

    def __init__(
            self, 
            config: GPTconfig
    )-> None:

        super.__init__()

        config.validate()

        # n of attention heads

        self. h_heads = config.n_heads

        self.head_dim = config.d_model // config.n_heads


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
            3* config.d_model,
            bias=False
        )

        #attention output projection 
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
            "casual_mask" ,
            mask ,
            persistent = False
        )

        

