"""
Prediction Visualizer for SymPanICH-Net v2.

Loads a trained checkpoint, runs inference on test samples, and creates
an interactive visualization with CT slices, predictions, and AI reports.

Usage:
    python scripts/visualize.py --checkpoint checkpoints/best.ckpt --num_samples 10
    python scripts/visualize.py --checkpoint checkpoints/best.ckpt --patient_id 76
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from src.training.trainer import SymPanICHNetModule
from src.data.datamodule import ICHDataModule
from src.utils.panoptic_fusion import panoptic_fusion
from src.utils.visualization import CLASS_COLORS, CLASS_NAMES, colorize_mask, overlay_mask
from src.models.report_generator import ReportGenerator


def visualize_predictions(
    model,
    dataloader,
    output_dir: str = "results/visualizations",
    num_samples: int = 10,
    device: str = "cpu",
):
    """Run inference and create visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    model.eval()

    reporter = ReportGenerator(image_size=256)
    sample_idx = 0
    all_reports = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_idx >= num_samples:
                break

            images = batch['image'].to(device)
            images_flipped = batch['image_flipped'].to(device)
            gt_masks = batch['mask']
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx']

            # Forward pass
            outputs = model(images, images_flipped)

            # Process each sample in the batch
            B = images.shape[0]
            for b in range(B):
                if sample_idx >= num_samples:
                    break

                pid = patient_ids[b] if isinstance(patient_ids[b], int) else patient_ids[b].item()
                sid = slice_indices[b] if isinstance(slice_indices[b], int) else slice_indices[b].item()

                # Get the brain-window image for display (first 3 channels)
                img_np = images[b, :3].cpu().numpy().transpose(1, 2, 0)
                gt_mask_np = gt_masks[b].cpu().numpy()

                # Run panoptic fusion
                fusion_result = panoptic_fusion(
                    outputs['pred_logits'][b],
                    outputs['pred_masks'][b],
                    image_size=256,
                )
                pred_mask_np = fusion_result['semantic_map']

                # Generate AI report
                segments = fusion_result['segments']
                if segments:
                    classes = np.array([s['class_id'] for s in segments])
                    masks = np.stack([s['mask'] for s in segments])
                    scores = np.array([s['score'] for s in segments])
                    report = reporter.generate(classes, masks, scores,
                                               slice_idx=sid, patient_id=f"Patient_{pid}")
                else:
                    report = reporter.generate(
                        np.array([]), np.zeros((0, 256, 256)), np.array([]),
                        slice_idx=sid, patient_id=f"Patient_{pid}"
                    )

                all_reports.append(report)

                # --- CREATE VISUALIZATION ---
                fig = plt.figure(figsize=(24, 10))
                gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

                # Panel 1: Original CT slice (brain window)
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(img_np[:, :, 0], cmap='gray')
                ax1.set_title(f'Brain Window\nPatient {pid}, Slice {sid}', fontsize=12)
                ax1.axis('off')

                # Panel 2: Multi-window composite
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(img_np)
                ax2.set_title('Multi-Window\n(Brain/Subdural/Bone)', fontsize=12)
                ax2.axis('off')

                # Panel 3: Ground Truth mask overlay
                ax3 = fig.add_subplot(gs[0, 2])
                gt_overlay = overlay_mask(img_np, gt_mask_np, alpha=0.5)
                ax3.imshow(gt_overlay)
                has_hem = "ICH ✓" if gt_mask_np.max() > 0 else "Normal ✗"
                ax3.set_title(f'Ground Truth ({has_hem})', fontsize=12)
                ax3.axis('off')

                # Panel 4: Prediction mask overlay
                ax4 = fig.add_subplot(gs[0, 3])
                pred_overlay = overlay_mask(img_np, pred_mask_np, alpha=0.5)
                ax4.imshow(pred_overlay)
                num_detected = len(segments)
                ax4.set_title(f'Prediction ({num_detected} detections)', fontsize=12)
                ax4.axis('off')

                # Panel 5: GT mask colores
                ax5 = fig.add_subplot(gs[1, 0])
                gt_colored = colorize_mask(gt_mask_np)
                ax5.imshow(gt_colored)
                ax5.set_title('GT Mask (Colored)', fontsize=12)
                ax5.axis('off')

                # Panel 6: Pred mask colored
                ax6 = fig.add_subplot(gs[1, 1])
                pred_colored = colorize_mask(pred_mask_np)
                ax6.imshow(pred_colored)
                ax6.set_title('Pred Mask (Colored)', fontsize=12)
                ax6.axis('off')

                # Panel 7-8: AI Report
                ax_report = fig.add_subplot(gs[1, 2:])
                ax_report.axis('off')
                # Display report as text
                report_lines = report.split('\n')[:15]  # First 15 lines
                report_text = '\n'.join(report_lines)
                ax_report.text(0.02, 0.98, report_text,
                               transform=ax_report.transAxes,
                               fontsize=9, fontfamily='monospace',
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9),
                               color='#00ff88')
                ax_report.set_title('AI Clinical Report', fontsize=12)

                # Legend
                patches = [
                    mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255, label=CLASS_NAMES[i])
                    for i in range(1, 6)
                ]
                fig.legend(handles=patches, loc='lower center', ncol=5,
                          fontsize=11, framealpha=0.9)

                fig.suptitle(f'SymPanICH-Net v2 — Patient {pid}, Slice {sid}',
                           fontsize=16, fontweight='bold')

                # Save
                save_path = os.path.join(output_dir, f'pred_patient{pid}_slice{sid}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                print(f'  ✅ Saved: {save_path}')

                sample_idx += 1

    # Save all reports to a text file
    reports_path = os.path.join(output_dir, 'all_reports.txt')
    with open(reports_path, 'w') as f:
        for report in all_reports:
            f.write(report + '\n\n' + '=' * 60 + '\n\n')
    print(f'\n📄 All reports saved to: {reports_path}')

    # Create summary HTML
    create_html_gallery(output_dir, num_samples)


def create_html_gallery(output_dir: str, num_images: int):
    """Create an HTML gallery of all prediction visualizations."""
    html_path = os.path.join(output_dir, 'gallery.html')
    images = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SymPanICH-Net v2 — Prediction Gallery</title>
    <style>
        body {{ background: #0f0f1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
        h1 {{ text-align: center; color: #00ff88; font-size: 2rem; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .gallery {{ display: flex; flex-direction: column; align-items: center; gap: 30px; }}
        .card {{ background: #1a1a2e; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,255,136,0.1); max-width: 1400px; width: 100%; }}
        .card img {{ width: 100%; display: block; }}
        .nav {{ position: fixed; top: 20px; right: 20px; display: flex; gap: 10px; }}
        .nav a {{ background: #00ff88; color: #0f0f1a; padding: 8px 16px; border-radius: 8px; text-decoration: none; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🧠 SymPanICH-Net v2 — Predictions</h1>
    <p class="subtitle">{len(images)} prediction visualizations</p>
    <div class="gallery">
"""
    for img in images:
        html += f'        <div class="card"><img src="{img}" alt="{img}"></div>\n'

    html += """    </div>
</body>
</html>"""

    with open(html_path, 'w') as f:
        f.write(html)
    print(f'🌐 Gallery saved to: {html_path}')


def main():
    parser = argparse.ArgumentParser(description="Visualize SymPanICH-Net v2 Predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="results/visualizations", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = SymPanICHNetModule.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Setup data
    datamodule = ICHDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 for visualization to avoid multiprocessing issues
        context_slices=2,
    )
    datamodule.setup("test")

    print(f"Test set: {len(datamodule.test_dataset)} slices")
    print(f"Generating {args.num_samples} visualizations...\n")

    # Run visualization
    visualize_predictions(
        model=model,
        dataloader=datamodule.test_dataloader(),
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=device,
    )

    print(f"\n🎉 Done! Open results/visualizations/gallery.html to view predictions.")


if __name__ == "__main__":
    main()
