from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 

list_paths = [
   "model.py",
   "tokenizer.py",
   "dataset.py",
   "train.py",
   "chat.py",
]

for path in list_paths:
   path = PROJECT_ROOT / "src" / "transfomers" / path
   path.touch(exist_ok=True) 