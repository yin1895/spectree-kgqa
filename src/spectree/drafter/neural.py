"""Optional PyTorch/HuggingFace dual-projection drafter.

Shared text-embedding encoder + LoRA adapters + two normalized projections.
Intent scores directed relation types; transition scores grounded actions.
Only loaded when the `ml` extra is installed; model weights are not bundled.
"""
from __future__ import annotations


def multi_positive_loss(logits, positive_mask, valid_mask):
    """-log(sum exp positive logits / sum exp valid action logits).

    Unlabeled states (without a positive action) are excluded, NOT treated as
    negative examples. Requires torch tensors of shape [B, num_actions].
    """
    import torch
    if logits.shape != positive_mask.shape or logits.shape != valid_mask.shape:
        raise ValueError("Masks must match logits")
    pos = positive_mask.bool() & valid_mask.bool()
    valid = valid_mask.bool()
    eligible = pos.any(-1)
    if not eligible.any():
        return logits.sum() * 0
    log_num = torch.logsumexp(logits.masked_fill(~pos, float("-inf")), dim=-1)
    log_den = torch.logsumexp(logits.masked_fill(~valid, float("-inf")), dim=-1)
    return (log_den[eligible] - log_num[eligible]).mean()


class DualScoringDrafter:
    """Lazily initialize Qwen3-Embedding-scale encoder and LoRA adapters.

    Initializes a shared text encoder, LoRA adapters and two projections.
    Tokenizer and checkpoint revisions are configurable.
    """
    def __init__(self, checkpoint: str = "Qwen/Qwen3-Embedding-0.6B",
                 *, lora_rank: int = 8, temperature: float = 0.07,
                 device: str = "auto"):
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.torch, self.temperature = torch, temperature
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
        encoder = AutoModel.from_pretrained(checkpoint, trust_remote_code=False,
                                            device_map=device)
        for p in encoder.parameters():
            p.requires_grad_(False)
        config = LoraConfig(r=lora_rank, lora_alpha=lora_rank*2, lora_dropout=0.05,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                            bias="none", task_type=TaskType.FEATURE_EXTRACTION)
        self.encoder = get_peft_model(encoder, config)
        hidden = getattr(encoder.config, "hidden_size", None)
        if not hidden:
            raise RuntimeError("Set encoder hidden_size for this checkpoint")
        self.intent = nn.Linear(hidden, hidden, bias=False).to(encoder.device)
        self.transition = nn.Linear(hidden, hidden, bias=False).to(encoder.device)
        self.checkpoint = checkpoint
        self.lora_rank = lora_rank

    def train(self):
        self.encoder.train()
        self.intent.train()
        self.transition.train()
        return self

    def eval(self):
        self.encoder.eval()
        self.intent.eval()
        self.transition.eval()
        return self

    def save(self, directory):
        from pathlib import Path
        import json
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(str(out / "lora"))
        self.tokenizer.save_pretrained(str(out / "tokenizer"))
        self.torch.save({"intent": self.intent.state_dict(),
                         "transition": self.transition.state_dict()}, out / "projections.pt")
        (out / "metadata.json").write_text(json.dumps({
            "base_checkpoint": self.checkpoint,
            "lora_rank": self.lora_rank,
            "temperature": self.temperature}, indent=2) + "\n")

    @classmethod
    def load(cls, directory, *, device="auto"):
        from pathlib import Path
        import json
        out = Path(directory)
        cfg = json.loads((out / "metadata.json").read_text())
        model = cls(cfg["base_checkpoint"], lora_rank=cfg["lora_rank"],
                    temperature=cfg["temperature"], device=device)
        model.encoder.load_adapter(str(out / "lora"), adapter_name="trained", is_trainable=False)
        model.encoder.set_adapter("trained")
        states = model.torch.load(out / "projections.pt", map_location="cpu", weights_only=True)
        model.intent.load_state_dict(states["intent"])
        model.transition.load_state_dict(states["transition"])
        model.eval()
        return model

    def parameters(self):
        return (p for m in [self.encoder, self.intent, self.transition]
                for p in m.parameters() if p.requires_grad)

    def encode(self, texts: list[str], tower: str):
        import torch.nn.functional as F
        if tower not in ("intent", "transition"):
            raise ValueError("Unknown tower")
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        dev = next(self.encoder.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        h = self.encoder(**inputs).last_hidden_state
        indices = inputs["attention_mask"].sum(dim=1).long() - 1
        pooled = h[self.torch.arange(h.size(0), device=dev), indices]
        projection = self.intent if tower == "intent" else self.transition
        return F.normalize(projection(pooled), p=2, dim=-1)

    def action_logits(self, state_text: str, action_texts: list[str]):
        u = self.encode([state_text], "intent")
        a = self.encode(action_texts, "transition")
        return (a @ u.T).squeeze(-1) / self.temperature

    def train_step(self, state_texts: list[str], action_text_groups: list[list[str]],
                   positive_masks: list[list[bool]], optimizer):
        """One question/state batch. Variable action widths are handled per state.

        Training files must specify which actions are supported from TRAIN labels;
        evaluation answer sets must never be used here.
        """
        import torch
        optimizer.zero_grad()
        loss = None
        contributing = 0
        if not (len(state_texts) == len(action_text_groups) == len(positive_masks)):
            raise ValueError("Training groups must have equal length")
        for state, actions, positives in zip(state_texts, action_text_groups, positive_masks):
            if len(actions) != len(positives):
                raise ValueError("Action/label count mismatch")
            logits = self.action_logits(state, actions).unsqueeze(0)
            pos = torch.tensor(positives, device=logits.device).unsqueeze(0)
            valid = torch.ones_like(pos, dtype=torch.bool)
            if any(positives):
                v = multi_positive_loss(logits, pos, valid)
                loss = v if loss is None else loss + v
                contributing += 1
        if loss is None or contributing == 0:
            raise ValueError("No positively supervised training states")
        loss = loss / contributing
        loss.backward(); optimizer.step()
        return float(loss.detach().cpu())
