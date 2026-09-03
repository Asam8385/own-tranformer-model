from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 

list_paths = [
   "model.py",
   "tokenizer.py",
   "dataset.py",
   "train.py",
   "chat.py",
   "data/qa.jsonl"
]

for path in list_paths:
   path = PROJECT_ROOT / "src" / "transfomers" / path
   path.parent.mkdir(parents=True , exist_ok=True)
   path.touch(exist_ok=True) 