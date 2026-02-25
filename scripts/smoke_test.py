"""Quick smoke test to verify that the model builds and runs correctly."""
import sys
sys.path.insert(0, '.')

import torch
print("Importing SymPanICH-Net v2...")
from src.models.sympanich_net import SymPanICHNetV2

# Build model with 6 classes (no pretrained to speed up test)
model = SymPanICHNetV2(
    pretrained=False,  # Skip downloading weights for quick test
    use_context=False,
    num_classes=6,
    num_decoder_layers=3,  # Fewer layers for speed
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model built! Total params: {total_params / 1e6:.1f}M")

# Forward pass
model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 256, 256)
    x_flip = torch.flip(x, dims=[3])
    out = model(x, x_flip)

print(f"pred_logits: {out['pred_logits'].shape}")
print(f"pred_masks:  {out['pred_masks'].shape}")
print(f"hv_maps:     {out['hv_maps'].shape}")
print(f"text_emb:    {out['text_embeddings'].shape}")
print("\n✅ Smoke test PASSED! Model works correctly.")
