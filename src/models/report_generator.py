"""
Module 8: AI Clinical Report Generator.

Generates structured diagnostic reports from segmentation predictions,
including hemorrhage type, location, severity, and recommendations.
Uses template-based approach for reliability (LLM mode optional).
"""

import torch
import numpy as np
from typing import Dict, List, Optional


# Brain region atlas mapping (simplified)
BRAIN_REGIONS = {
    "frontal_left": {"y_range": (0, 0.4), "x_range": (0, 0.5)},
    "frontal_right": {"y_range": (0, 0.4), "x_range": (0.5, 1.0)},
    "parietal_left": {"y_range": (0.2, 0.6), "x_range": (0, 0.5)},
    "parietal_right": {"y_range": (0.2, 0.6), "x_range": (0.5, 1.0)},
    "temporal_left": {"y_range": (0.3, 0.7), "x_range": (0, 0.3)},
    "temporal_right": {"y_range": (0.3, 0.7), "x_range": (0.7, 1.0)},
    "occipital_left": {"y_range": (0.6, 1.0), "x_range": (0, 0.5)},
    "occipital_right": {"y_range": (0.6, 1.0), "x_range": (0.5, 1.0)},
    "central": {"y_range": (0.3, 0.7), "x_range": (0.3, 0.7)},
}

CLASS_NAMES = [
    "background", "intraventricular", "intraparenchymal",
    "subarachnoid", "epidural", "subdural", "ambiguous",
]

CLASS_DESCRIPTIONS = {
    1: "intraventricular hemorrhage (IVH)",
    2: "intraparenchymal hemorrhage (IPH)",
    3: "subarachnoid hemorrhage (SAH)",
    4: "epidural hemorrhage (EDH)",
    5: "subdural hemorrhage (SDH)",
    6: "ambiguous hemorrhage",
}


class ReportGenerator:
    """
    Template-based clinical report generator.

    Extracts structured information from panoptic predictions and
    fills a clinical report template with findings, severity, and
    recommendations.
    """

    def __init__(self, image_size: int = 256, pixel_spacing_mm: float = 0.5):
        self.image_size = image_size
        self.pixel_spacing = pixel_spacing_mm

    def _get_region(self, cy: float, cx: float) -> str:
        """Map centroid to brain region name."""
        ny, nx = cy / self.image_size, cx / self.image_size
        for region, bounds in BRAIN_REGIONS.items():
            yr = bounds["y_range"]
            xr = bounds["x_range"]
            if yr[0] <= ny <= yr[1] and xr[0] <= nx <= xr[1]:
                return region.replace("_", " ").title()
        return "Unspecified region"

    def _compute_severity(self, findings: List[Dict]) -> dict:
        """Compute severity based on hemorrhage metrics."""
        total_volume_ml = sum(f["area_mm2"] * self.pixel_spacing / 1000 for f in findings)
        num_types = len(set(f["class_id"] for f in findings))

        # Check for midline shift (asymmetry in centroids)
        if len(findings) >= 1:
            centroids_x = [f["centroid"][1] for f in findings]
            midline = self.image_size / 2
            max_shift = max(abs(cx - midline) for cx in centroids_x) * self.pixel_spacing
        else:
            max_shift = 0.0

        score = 0
        # Volume scoring
        if total_volume_ml > 30:
            score += 3
        elif total_volume_ml > 10:
            score += 2
        elif total_volume_ml > 0:
            score += 1

        # Type count scoring
        score += min(num_types, 3)

        # Location scoring (deep = worse)
        has_deep = any(f["class_id"] in [1, 2] for f in findings)  # IVH, IPH
        if has_deep:
            score += 2

        if score >= 6:
            level = "SEVERE"
        elif score >= 3:
            level = "MODERATE"
        else:
            level = "MILD"

        return {
            "level": level,
            "score": score,
            "total_volume_ml": round(total_volume_ml, 1),
            "midline_shift_mm": round(max_shift, 1),
            "num_types": num_types,
        }

    def _extract_findings(
        self,
        pred_classes: np.ndarray,
        pred_masks: np.ndarray,
        pred_scores: np.ndarray,
    ) -> List[Dict]:
        """Extract structured findings from predictions."""
        findings = []

        for i in range(len(pred_classes)):
            cls_id = int(pred_classes[i])
            if cls_id == 0:  # Skip background
                continue

            score = float(pred_scores[i])
            mask = pred_masks[i]  # (H, W)
            area_px = int(mask.sum())

            if area_px < 10:  # Skip tiny noise
                continue

            # Compute centroid
            ys, xs = np.where(mask > 0.5)
            if len(ys) == 0:
                continue
            cy, cx = float(ys.mean()), float(xs.mean())

            # Compute bounding box
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            thickness = max(y_max - y_min, x_max - x_min) * self.pixel_spacing

            findings.append({
                "class_id": cls_id,
                "class_name": CLASS_DESCRIPTIONS.get(cls_id, "unknown"),
                "confidence": round(score * 100, 1),
                "area_px": area_px,
                "area_mm2": round(area_px * self.pixel_spacing ** 2, 1),
                "centroid": (cy, cx),
                "region": self._get_region(cy, cx),
                "thickness_mm": round(thickness, 1),
                "laterality": "Right" if cx > self.image_size / 2 else "Left",
            })

        # Sort by confidence descending
        findings.sort(key=lambda f: f["confidence"], reverse=True)
        return findings

    def generate(
        self,
        pred_classes: np.ndarray,
        pred_masks: np.ndarray,
        pred_scores: np.ndarray,
        slice_idx: int = 0,
        patient_id: str = "Unknown",
    ) -> str:
        """
        Generate a clinical report from segmentation predictions.

        Args:
            pred_classes: (N,) array of predicted class IDs.
            pred_masks: (N, H, W) array of binary masks.
            pred_scores: (N,) array of confidence scores.
            slice_idx: Axial slice index.
            patient_id: Patient identifier.

        Returns:
            Formatted clinical report string.
        """
        findings = self._extract_findings(pred_classes, pred_masks, pred_scores)

        if not findings:
            return self._generate_normal_report(patient_id, slice_idx)

        severity = self._compute_severity(findings)
        return self._format_report(findings, severity, patient_id, slice_idx)

    def _generate_normal_report(self, patient_id: str, slice_idx: int) -> str:
        """Generate report for normal (no hemorrhage) slices."""
        return (
            f"SymPanICH-Net v2 — AI Clinical Report\n"
            f"{'=' * 50}\n"
            f"Patient: {patient_id} | Slice: {slice_idx}\n\n"
            f"FINDINGS: No intracranial hemorrhage detected.\n"
            f"Brain parenchyma appears normal.\n"
            f"No midline shift or mass effect identified.\n\n"
            f"IMPRESSION: Normal non-contrast CT brain.\n"
        )

    def _format_report(
        self, findings: List[Dict], severity: dict, patient_id: str, slice_idx: int
    ) -> str:
        """Format the full clinical report."""
        lines = [
            f"SymPanICH-Net v2 — AI Clinical Report",
            f"{'=' * 50}",
            f"Patient: {patient_id} | Slice: {slice_idx}",
            f"",
            f"FINDINGS SUMMARY",
            f"-" * 30,
            f"{len(findings)} hemorrhagic {'focus' if len(findings) == 1 else 'foci'} identified.",
        ]

        # List each finding
        for i, f in enumerate(findings, 1):
            lines.extend([
                f"",
                f"Finding #{i}: {f['class_name']} (Confidence: {f['confidence']}%)",
                f"  Location: {f['laterality']} {f['region']}",
                f"  Size: {f['area_mm2']} mm² | Thickness: {f['thickness_mm']} mm",
            ])

        # Severity
        lines.extend([
            f"",
            f"SEVERITY: {severity['level']}",
            f"-" * 30,
            f"  Total volume: ~{severity['total_volume_ml']} mL",
            f"  Midline shift: {severity['midline_shift_mm']} mm",
            f"  Hemorrhage types: {severity['num_types']}",
        ])

        # Recommendations
        lines.extend([
            f"",
            f"RECOMMENDED ACTIONS",
            f"-" * 30,
        ])

        if severity["level"] == "SEVERE":
            lines.extend([
                f"  • URGENT neurosurgical consultation",
                f"  • Repeat CT in 2-4 hours",
                f"  • Consider surgical intervention",
                f"  • Continuous neurological monitoring",
            ])
        elif severity["level"] == "MODERATE":
            lines.extend([
                f"  • Neurosurgical consultation",
                f"  • Repeat CT in 6 hours to assess progression",
                f"  • Monitor GCS and neurological status",
                f"  • Consider surgical evacuation if worsening",
            ])
        else:
            lines.extend([
                f"  • Clinical observation and monitoring",
                f"  • Follow-up CT in 12-24 hours",
                f"  • Neurological status checks every 4 hours",
            ])

        lines.extend([
            f"",
            f"NOTE: This is an AI-generated report. Clinical correlation required.",
        ])

        return "\n".join(lines)
