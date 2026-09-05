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
class QA