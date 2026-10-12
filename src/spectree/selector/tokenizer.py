"""Complete chat token counter for locally hosted frozen language models."""
from __future__ import annotations

class HFChatCounter:
    def __init__(self, checkpoint: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)

    def __call__(self, text: str) -> int:
        inputs = self.tokenizer.apply_chat_template(
            [{"role":"user", "content": text}], tokenize=True, add_generation_prompt=True)
        return len(inputs)

    def count_record(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
