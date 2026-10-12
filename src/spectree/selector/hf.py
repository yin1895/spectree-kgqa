"""Frozen HF causal LM selector and exact complete-chat input-token counter."""
from __future__ import annotations


class FrozenHFSelector:
    def __init__(self, checkpoint: str = "Qwen/Qwen3.5-9B", device_map="auto",
                 max_new_tokens: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, device_map=device_map, torch_dtype="auto", trust_remote_code=False)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.max_new_tokens = max_new_tokens

    def _ids(self, packet_text: str):
        conversation = [{"role": "user", "content": packet_text}]
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            return_tensors="pt")

    def count_complete_chat(self, packet_text: str) -> int:
        return int(self._ids(packet_text).shape[-1])

    def select(self, packet_text: str) -> str:
        ids = self._ids(packet_text).to(next(self.model.parameters()).device)
        with self.torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        completion = self.tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
        # Return raw text. The allocation controller validates membership and
        # returns an empty prediction for invalid/multi-line outputs.
        return completion.strip()
