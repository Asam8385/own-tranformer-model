from __future__ import annotations

from pathlib import Path

import torch


# ============================================================
# CHANGE YOUR QUESTION HERE, THEN PRESS RUN
# ============================================================

QUESTION = "what is python list?"
CHECKPOINT = "checkpoints/best.pt"
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.5      # 0 = greedy decoding (repeatable results)
TOP_K = 40
TOP_P = 0.95
INTERACTIVE = False    # True = keep asking questions until you type exit


# ============================================================
# LOAD CHECKPOINT AND GENERATE ANSWERS
# ============================================================

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: Path, device: torch.device) -> MiniGPT:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run train.py first.")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MiniGPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def decode_answer(tokenizer: ByteTokenizer, token_ids: list[int]) -> str:
    """Turn generated token ids into text using the tokenizer's own byte offset.

    Works with the tokenizer exactly as it is: special tokens (ids below BYTE_OFFSET)
    are skipped and every other id is converted back to its UTF-8 byte.
    """
    offset = tokenizer.BYTE_OFFSET
    raw = bytes(t - offset for t in token_ids if offset <= t < offset + 256)
    return raw.decode("utf-8", errors="replace")


def answer_question(model: MiniGPT, tokenizer: ByteTokenizer, question: str, device: torch.device) -> str:
    prompt_ids = tokenizer.build_prompt(question)
    if len(prompt_ids) >= model.config.max_seq_len:
        raise ValueError("Question exceeds the context length. Shorten QUESTION or increase max_seq_len.")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    suppress = [tokenizer.special.pad, tokenizer.special.bos, tokenizer.special.question,
                tokenizer.special.answer]
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        eos_token_id=tokenizer.special.eos,
        suppress_token_ids=suppress,
    )

    generated = output_ids[0, len(prompt_ids):].tolist()
    if tokenizer.special.eos in generated:
        generated = generated[:generated.index(tokenizer.special.eos)]
    return decode_answer(tokenizer, generated).strip()


def main() -> None:
    root = Path("/kaggle/working")
    path = Path(CHECKPOINT)
    if not path.is_absolute():
        path = root / path

    device = choose_device()
    model = load_model(path, device)
    tokenizer = ByteTokenizer()

    if not INTERACTIVE:
        print(f"Question: {QUESTION}")
        print(f"MiniGPT: {answer_question(model, tokenizer, QUESTION, device)}")
        return

    print("Interactive MiniGPT. Type 'exit' to quit.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print(f"MiniGPT: {answer_question(model, tokenizer, question, device)}")


if __name__ == "__main__":
    main()