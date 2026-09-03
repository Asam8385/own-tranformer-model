from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable , List

@dataclass(frozen=True)
class SpecialTokens():
   pad: int = 0
   bos: int = 1
   eos: int = 2
   question: int = 3
   answer: int = 4


class ByteTokenizer():

    """
    Simple UTF-8 byte tokenizer built from scratch.

    Special tokens:
        0 = <PAD>
        1 = <BOS>
        2 = <EOS>
        3 = <Q>
        4 = <A>

    Byte values:
        raw byte 0   -> token 5
        raw byte 1   -> token 6
        ...
        raw byte 255 -> token 260

    Total vocabulary size = 261
    """

    BYTE_OFFSET = 5 

    def __init__(self):
        self.special = SpecialTokens() 

        self.vocab_size = 256 + self.BYTE_OFFSET

    def encode_text(self , text:str):

        raw_bytes = text.encode("utf-8")

        token_ids = [  token + self.BYTE_OFFSET for token in raw_bytes ]

    


