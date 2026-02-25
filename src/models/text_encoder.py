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


class TextEncoder(nn.Module):
    """
    Encodes clinical hemorrhage descriptions into fixed embeddings.

    Uses a pretrained medical language model (BiomedBERT) to encode
    text descriptions, then projects to the visual feature dimension.

    The text embeddings serve as semantic anchors — each pixel in the
    image can be compared against these embeddings to determine which
    hemorrhage type it best matches.
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

        # Will be initialized lazily (to handle different environments)
        self._encoder = None
        self._tokenizer = None
        self._projection = nn.Linear(768, embed_dim)  # BERT hidden=768 → 256
        self._norm = nn.LayerNorm(embed_dim)

        # Cache for text embeddings (computed once, reused every forward)
        self._cached_embeddings = None

    def _init_encoder(self):
        """Lazily initialize the text encoder (avoids import errors on systems without transformers)."""
        if self._encoder is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            encoder = AutoModel.from_pretrained(self.model_name)

            if self.freeze_initial:
                for param in encoder.parameters():
                    param.requires_grad = False

            # Register as submodule so .to(device) moves it correctly
            self.add_module('bert_encoder', encoder)
            self._encoder = self.bert_encoder

            # Move encoder to same device as projection layer
            target_device = self._projection.weight.device
            self._encoder = self._encoder.to(target_device)

        except ImportError:
            print("WARNING: transformers not installed. Using random text embeddings.")
            self._encoder = "dummy"

    def _load_descriptions(self) -> dict:
        """Load hemorrhage descriptions from JSON file."""
        if self._descriptions_path and Path(self._descriptions_path).exists():
            with open(self._descriptions_path, 'r') as f:
                return json.load(f)

        # Default descriptions (fallback)
        return {
            "intraventricular": "Hyperdense fluid collection within the ventricular system with fluid-fluid levels and ventricular enlargement on CT.",
            "intraparenchymal": "Well-defined hyperdense lesion within brain parenchyma, round or oval, surrounded by low-density perilesional edema ring on CT.",
            "subarachnoid": "Hyperdense material filling sulci and cisterns, following brain surface contour with star-shaped pattern in basal cisterns on CT.",
            "epidural": "Biconvex lens-shaped hyperdense collection between skull and dura mater that does not cross suture lines on CT.",
            "subdural": "Crescent-shaped hyperdense collection along brain convexity crossing suture lines with possible midline shift on CT.",
            "ambiguous": "Ill-defined hyperdense region with mixed density patterns and unclear boundaries between hemorrhage types on CT.",
        }

    def _encode_texts(self, device: torch.device) -> torch.Tensor:
        """Encode all hemorrhage descriptions into embeddings."""
        self._init_encoder()
        descriptions = self._load_descriptions()
        texts = list(descriptions.values())

        if self._encoder == "dummy":
            # Return random embeddings when transformers is not available
            return torch.randn(len(texts), self.embed_dim, device=device)

        # Tokenize
        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        # Encode
        with torch.no_grad() if self.freeze_initial else torch.enable_grad():
            outputs = self._encoder(**tokens)
            # Use [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (6, 768)

        # Project to visual dimension
        projected = self._projection(cls_embeddings)  # (6, 256)
        projected = self._norm(projected)

        return projected

    def forward(self, device: torch.device = None) -> torch.Tensor:
        """
        Get text embeddings for all hemorrhage types.

        Args:
            device: Device to put embeddings on.

        Returns:
            Text embeddings T ∈ ℝ^(6, embed_dim).
        """
        if device is None:
            device = self._projection.weight.device

        # Use cache if available and on same device
        if self._cached_embeddings is not None and self._cached_embeddings.device == device:
            return self._cached_embeddings

        embeddings = self._encode_texts(device)
        self._cached_embeddings = embeddings
        return embeddings

    def unfreeze(self):
        """Unfreeze the text encoder for fine-tuning."""
        if self._encoder is not None and self._encoder != "dummy":
            for param in self._encoder.parameters():
                param.requires_grad = True
        self.freeze_initial = False
        self._cached_embeddings = None  # Clear cache to allow gradient flow

    def invalidate_cache(self):
        """Clear the embedding cache (call after unfreezing or changing descriptions)."""
        self._cached_embeddings = None
