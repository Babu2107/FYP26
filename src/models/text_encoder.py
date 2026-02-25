"""
Module 7: Text Encoder.

Encodes clinical text descriptions of each hemorrhage type using
a medical language model (BiomedBERT or BiomedCLIP text tower).
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


# Default hemorrhage type descriptions
DEFAULT_DESCRIPTIONS = [
    "Hyperdense fluid collection within the ventricular system with fluid-fluid levels and ventricular enlargement on CT.",
    "Well-defined hyperdense lesion within brain parenchyma, round or oval, surrounded by low-density perilesional edema ring on CT.",
    "Hyperdense material filling sulci and cisterns, following brain surface contour with star-shaped pattern in basal cisterns on CT.",
    "Biconvex lens-shaped hyperdense collection between skull and dura mater that does not cross suture lines on CT.",
    "Crescent-shaped hyperdense collection along brain convexity crossing suture lines with possible midline shift on CT.",
]


class TextEncoder(nn.Module):
    """
    Encodes clinical hemorrhage descriptions into fixed embeddings.

    Uses a pretrained medical language model (BiomedBERT) to encode
    text descriptions, then projects to the visual feature dimension.

    IMPORTANT: BiomedBERT is initialized eagerly in __init__ so that
    .to(device) properly moves all weights to GPU.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        embed_dim: int = 256,
        descriptions_path: Optional[str] = None,
        freeze_initial: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.freeze_initial = freeze_initial
        self.model_name = model_name
        self._descriptions_path = descriptions_path

        # Projection: BERT hidden dim (768) → embed_dim
        self.projection = nn.Linear(768, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Cache
        self._cached_embeddings = None

        # Initialize encoder eagerly (NOT lazily!)
        self._has_transformers = False
        self._tokenizer = None
        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            self._has_transformers = True

            if freeze_initial:
                for param in self.bert.parameters():
                    param.requires_grad = False

            print(f"✅ BiomedBERT loaded ({model_name})")
        except Exception as e:
            print(f"WARNING: Could not load BiomedBERT: {e}")
            print("Using random text embeddings as fallback.")
            # Register a dummy parameter so .to(device) has something
            self.bert = None

    def _load_descriptions(self) -> list:
        """Load hemorrhage descriptions."""
        if self._descriptions_path and Path(self._descriptions_path).exists():
            with open(self._descriptions_path, 'r') as f:
                data = json.load(f)
                return list(data.values())
        return DEFAULT_DESCRIPTIONS

    def _encode_texts(self, device: torch.device) -> torch.Tensor:
        """Encode all hemorrhage descriptions into embeddings."""
        descriptions = self._load_descriptions()

        if self.bert is None or not self._has_transformers:
            # Fallback: random embeddings
            return torch.randn(len(descriptions), self.embed_dim, device=device)

        # Tokenize
        tokens = self._tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # Move token tensors to the correct device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Encode with BiomedBERT (already on correct device via .to())
        with torch.no_grad() if self.freeze_initial else torch.enable_grad():
            outputs = self.bert(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (5, 768)

        # Project to visual dimension
        projected = self.projection(cls_embeddings)  # (5, 256)
        projected = self.norm(projected)

        return projected

    def forward(self, device: torch.device = None) -> torch.Tensor:
        """
        Get text embeddings for all hemorrhage types.

        Args:
            device: Device to put embeddings on.

        Returns:
            Text embeddings T ∈ ℝ^(5, embed_dim).
        """
        if device is None:
            device = self.projection.weight.device

        # Use cache if available and on same device
        if self._cached_embeddings is not None and self._cached_embeddings.device == device:
            return self._cached_embeddings

        embeddings = self._encode_texts(device)
        self._cached_embeddings = embeddings
        return embeddings

    def unfreeze(self):
        """Unfreeze the text encoder for fine-tuning."""
        if self.bert is not None:
            for param in self.bert.parameters():
                param.requires_grad = True
        self.freeze_initial = False
        self._cached_embeddings = None

    def invalidate_cache(self):
        """Clear the embedding cache."""
        self._cached_embeddings = None
