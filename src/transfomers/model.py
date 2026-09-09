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

class  CausalSelfAttention(
    nn.modules
):

    def __init__(
            self, 
            config: GPTconfig
    )-> None:

        super.__init__()

        config.validate()

        # n of attention heads

        self.n_heads = config.n_heads

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
            "causal_mask" ,
            mask ,
            persistent = False
        )
 

    # ------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------


    def forward(
            self,
            x: torch.Tensor 
    ) -> torch.Tensor:

        # x :
        #
        # [batch , sequence , d_model]

        batch_size , seq_len, channels = (
            x.shape
        )

        # generate q k v

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
            k.trasnpose(
                -2 ,
                -1
            )
        )

        mask = self.causal_mask[
            : , : ,
            :seq_len,
            :seq_len
        ]

        scores = scores.maked_fill(
            ~mask,
            torch.finfo(
                scores.dtype
            ).min
        )

        # softmax

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        attention_weights = (
            self.attn_dropout(
                attention_weights
            )
        )

        # -------------------------------------------------
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
    nn.modules
):

    def __init__(
            self ,
            config: GPTconfig
    )-> None:

        super.__init__()

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
    )-> torch.Tensor:

        return self.network(x)



# ==========================================================
# TRANSFORMER BLOCK
# ==========================================================

class TransformerBlock(
    nn.Module
):

    def __init__(
        self,
        config: GPTconfig
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



class MiniGPT(
    nn.Module
):
    def __int__(
            self,
            config: GPTconfig
    ) -> None:
        
        super.__init__()

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

        self.position_embedding = (
            nn.Embedding(
                config.max_seq_len,
                config.d_model
            )
        )    

        self.dropout = nn.Dropout(
            config.dropout
        )

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

        # this used to convert my 256 embeddings to  261 
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Initialize the weights
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

              




