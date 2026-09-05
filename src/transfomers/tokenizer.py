from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable , List

@dataclass(frozen=True)
class SpecialTokens:
   pad: int = 0
   bos: int = 1
   eos: int = 2
   question: int = 3
   answer: int = 4

class ByteTokenizer:

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

        return token_ids


    def decode(self, ids: Iterable[int] , skip_special: bool) -> str:

        raw = bytearray()

        for token_id in ids:
            token_id = int(token_id)

            if token_id >= self.BYTE_OFFSET:

                byte_value = token_id - self.BYTE_OFFSET

                if 0<= byte_value <= 255:
                    raw.append(byte_value)

            #special token
            elif not skip_special:

                special_map = {
                    self.special.pad: b"<PAD>",
                    self.special.bos: b"<BOS>",
                    self.special.eos: b"<EOS>",
                    self.special.question: b"<Q>",
                    self.special.answer: b"<A>",              
                }

                marker = special_map.get(
                    token_id,
                    b"<UNKNOWN>"                  
                )

                raw.extend(marker)

    def build_prompt(
            self ,
            question: str ,

    )-> List[int]:

        """"
       [
            1,      # <BOS>
            3,      # <Q>
            77,110, # "Hi"
            4       # <A>
        ]
        """

        tokens = (
            [self.special.bos]
            + 
            [self.special.question]
            +
            self.encode_text(question.strip())
            +
            [self.special.answer]

        )

        return tokens

    def build_qa(
            self ,
            question: str ,
            answer: str
    ) -> list[int]:

        tokens = (
            [self.build_prompt(question)]
            +
            self.encode_text(answer.strip())
            +
            [self.special.eos]
        )
        


        

        














































