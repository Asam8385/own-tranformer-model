from __future__ import annotations

import csv
import json
import random

from dataclasses import dataclass
from pathlib import Path
from typing import Dict , List , Sequence , Tuple

import torch

from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


@dataclass
class QAItem:

   question: str

   answer: str


# load dataset

def load_qa_file(
      path : str | Path
) -> List[QAItem]:

   path = Path(path)

   items: list[QAItem] = []


   if path.suffix.lower() == ".jsonl":

      with path.open(
         "r",
         encoding="utf-8"
      ) as f:

         for line_no , line in enumerate(
            f,
            start=1

         ):
            line = line.strip()

            if not line:
               continue

            obj = json.loads(line)

            question = str(
               obj["question"]
            ).strip()

            answer = str(
               obj["answer"]
            ).strip()

            if question and answer:

               items.append(
                  QAItem(
                     question=question,
                     answer=answer
                  )
               )


            # ------------------------------------------------------
            # CSV
            # ------------------------------------------------------

            elif path.suffix.lower() == ".csv":

               with path.open(
                     "r",
                     encoding="utf-8-sig",
                     newline=""
               ) as f:

                     reader = csv.DictReader(f)

                     if (
                        not reader.fieldnames
                        or "question" not in reader.fieldnames
                        or "answer" not in reader.fieldnames
                     ):

                        raise ValueError(
                           "CSV must contain columns "
                           "'question' and 'answer'."
                        )

                     for row in reader:

                        question = str(
                           row.get(
                                 "question",
                                 ""
                           )
                        ).strip()

                        answer = str(
                           row.get(
                                 "answer",
                                 ""
                           )
                        ).strip()

                        if question and answer:

                           items.append(
                                 QAItem(
                                    question=question,
                                    answer=answer
                                 )
                           )

            else:

               raise ValueError(
                     "Supported formats: "
                     ".jsonl and .csv"
               )

            if not items:

               raise ValueError(
                     f"No valid Q&A records found in {path}"
               )

            return items



# split dataset 

def split_items(
      items: Sequence[QAItem],
      val_fraction: float = 0.1,
      seed: int = 1337

) -> tuple[
   List[QAItem],
   List[QAItem]
]:
   if not 0 <= val_fraction < 1:

      raise ValueError(
         "val_fraction must be between 0 and 1"
      )

   indices = list(
      range(len(items))
   )

   rng = random.Random(seed)

   rng.shuffle(indices)

   n_val = int(
      len(item) * val_fraction
   )

   if (
      len(items) >= 20 
      and val_fraction > 0
   ):

      n_val = max(1, n_val)


   val_indices = set(
      indices[:n_val]
   )

   train_items = [
      item 
      for i, item in enumerate(items)
      if i not in val_indices
   ]

   val_items = [
      item 
      for i , item in enumerate(items)
      if i in val_indices
   ]

   return (
      train_items , val_items
   )



   

class QADataset(Dataset):

    """
    Supervised causal language-model dataset.

    Example:

    <BOS><Q>
    What is 2+2?
    <A>
    4
    <EOS>

    The question is given to the model as context.

    By default, loss is calculated only for
    answer tokens and EOS.
    """

    def __init__(
          self,
          items: Sequence[QAItem],
          tokenizer: ByteTokenizer,
          max_seq_len: int,
          train_on_prompt: bool = False
    ):

       self.items = list(items)
       self.tokenizer = tokenizer
       self.max_seq_len = max_seq_len
       self.train_on_prompt = train_on_prompt

    def __len__(self) -> int:
       return len(self.items)

    



    def _encode_limited(
          self,
          item: QAItem
    ) -> tuple[list[int], int]:


       tok  = self.tokenizer

       question_ids = tok.encode_text(
          item.question.strip()
       )

       answer_ids = tok.encode_text(
          item.answer.strip()
       )

       
   
   


