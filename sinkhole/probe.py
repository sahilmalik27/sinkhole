from __future__ import annotations

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from sinkhole.models import RawCapture


class ModelProbe:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            output_attentions=True,
        )
        self.model.eval()
        self._hidden_captures: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _get_layers(self):
        model_inner = self.model.model
        if hasattr(model_inner, "layers"):
            return model_inner.layers
        raise ValueError(f"Unsupported architecture: {type(self.model).__name__}")

    def _register_hooks(self):
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            hook = layer.input_layernorm.register_forward_hook(
                self._make_hidden_hook(i)
            )
            self._hooks.append(hook)

    def _make_hidden_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # input to layernorm is the hidden state before normalization
            hidden = input[0] if isinstance(input, tuple) else input
            self._hidden_captures.append(hidden.detach().float().cpu())
        return hook_fn

    def run(self, prompt: str, max_new_tokens: int = 1) -> RawCapture:
        self._hidden_captures.clear()

        # Apply chat template if available (gets <|im_start|> etc.)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        token_ids = inputs["input_ids"][0].tolist()
        token_texts = [self.tokenizer.decode([tid]) for tid in token_ids]

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, max_new_tokens=max_new_tokens)

        # Extract attention weights: tuple of (n_layers,) each [batch, n_heads, seq, seq]
        attn_weights = []
        for layer_attn in outputs.attentions:
            attn_weights.append(layer_attn[0].float().cpu().numpy())

        # Hidden states from hooks: one per layer
        hidden_states = [h[0].numpy() for h in self._hidden_captures]

        return RawCapture(
            hidden_states=hidden_states,
            attn_weights=attn_weights,
            token_ids=token_ids,
            token_texts=token_texts,
        )

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
