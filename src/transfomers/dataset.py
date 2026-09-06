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

    



    # ------------------------------------------------------
    # Encode while respecting context size
    # ------------------------------------------------------

    def _encode_limited(
        self,
        item: QAItem
    ) -> tuple[list[int], int]:

        tok = self.tokenizer

        question_ids = tok.encode_text(
            item.question.strip()
        )

        answer_ids = tok.encode_text(
            item.answer.strip()
        )

        # We need max_seq_len + 1 because:
        #
        # full:
        # [1,2,3,4,5]
        #
        # input:
        # [1,2,3,4]
        #
        # target:
        # [2,3,4,5]

        max_total = (
            self.max_seq_len + 1
        )

        # BOS
        # Q
        # A
        # EOS
        #
        # = 4 fixed special tokens

        available = (
            max_total - 4
        )

        if available <= 0:

            raise ValueError(
                "max_seq_len is too small"
            )

        # Initially reserve approximately
        # half for question.

        question_budget = min(
            len(question_ids),
            available,
            max(
                1,
                max(
                    8,
                    available // 2
                )
            )
        )

        answer_budget = (
            available - question_budget
        )

        # If answer is short, allow question
        # to use unused capacity.

        if len(answer_ids) < answer_budget:

            extra = (
                answer_budget
                - len(answer_ids)
            )

            question_budget = min(
                len(question_ids),
                question_budget + extra
            )

            answer_budget = (
                available
                - question_budget
            )

        question_ids = (
            question_ids[
                :question_budget
            ]
        )

        answer_ids = (
            answer_ids[
                :answer_budget
            ]
        )

        full_sequence = (

           [
              tok.special.bos ,
              tok.special.question
           ]
           +
           question_ids
           +
           [
              tok.special.answer
           ]
           +
           answer_ids
           +
           [
              tok.special.eos
           ]
        )

        # position of <A>

        answer_marker_index = (
           2 + len(question_ids)
        )


        return (
           full_sequence,
           answer_marker_index
        )

    def __getitem__(
          self,
          idx, int
    ) -> Dict[str , torch.Tensor]:

       full_sequence , answer_marker_index  = (
          self._encode_limited(
             self.items[idx]
          )
       )

       # input token 
       x = torch.tensor(
          full_sequence[:-1],
          dtype=torch.long
       )

       # input token 
       y = torch.tensor(
          full_sequence[1:],
          dtype=torch.long
       ) 

        # --------------------------------------------------
        # Ignore question/prompt in training loss
        # --------------------------------------------------

       if not self.train_on_prompt:

            y[
                :answer_marker_index
            ] = -100

       return {
            "input_ids": x,
            "targets": y
        }

    def collate_qa(
          batch: List[
             Dict[str , torch.Tensor]
          ],
          pad_token_id: int = 0
    ):

       max_length = max(
          item["input_ids"].numel()
          for item in batch
       )

       batch_size = max_length(batch)

       input_ids = torch.full(
          (
             batch_size, max_length
          ),
          pad_token_id,
          dtype=torch.long
       )

       targets = torch.full(
          (
             batch_size, max_length
          ),
          -100,
          dtype=torch.long
       )

       lengths = torch.zeros(
          batch_size,
          dtype=torch.long
       )

       for i , item in enumerate(batch):

          n = item["input_ids"].numel()

          input_ids[
             i , 
             :n
          ] = item["input_ids"]

          targets[
             i,
             :n
          ] = item["targets"]

          lengths[i] = n

       return {
          "input_ids":
            input_ids,

         "targets":
            targets,

         "lengths":
            lengths
          
       }







    




     

       
       
       
       

        




       





        




   
   


