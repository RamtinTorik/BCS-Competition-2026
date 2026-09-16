"""ICH inference module for the IAAA brain CT competition.

Loads four V4 SegResNet fold-best checkpoints and predicts:
V_EDH, V_SDH, V_IPH, V_SAH, V_IVH (mL).

Preprocessing matches V4: WL=40, WW=80, 2.5D k=2 (5 channels),
slice ordering by ImagePositionPatient[2], boundary replication, and
V4's physical volume calculation using real local z spacing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pydicom
import torch
from monai.networks.nets import SegResNet

CLASS_NAMES = ["BG", "IVH", "IPH", "SDH", "EDH", "SAH"]
ICH_CLASSES = CLASS_NAMES[1:]
NUM_CLASSES = 6
WINDOW_LEVEL = 40.0
WINDOW_WIDTH = 80.0
NEIGHBOR_K = 2
INIT_FILTERS = 16
BLOCKS_DOWN = [1, 2, 2, 4]
BLOCKS_UP = [1, 1, 1]
DROPOUT_PROB = 0.2


def build_model(device: torch.device) -> torch.nn.Module:
    """Build exactly the V4 SegResNet architecture."""
    model = SegResNet(
        spatial_dims=2,
        in_channels=2 * NEIGHBOR_K + 1,
        out_channels=NUM_CLASSES,
        init_filters=INIT_FILTERS,
        blocks_down=BLOCKS_DOWN,
        blocks_up=BLOCKS_UP,
        dropout_prob=DROPOUT_PROB,
    )
    return model.to(device)


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    """Load a V4 checkpoint (.pt or PyTorch .zip archive)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


def load_sorted_series(case_dir: str | Path) -> List[dict]:
    """Read DICOM headers and sort slices by ImagePositionPatient[2], as in V4."""
    case_dir = Path(case_dir)
    paths = sorted(case_dir.glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No .dcm files found in: {case_dir}")

    slices = []
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if not hasattr(ds, "ImagePositionPatient"):
            raise ValueError(f"Missing ImagePositionPatient: {path}")
        slices.append({
            "path": str(path),
            "z": float(ds.ImagePositionPatient[2]),
            "sop_uid": str(ds.SOPInstanceUID),
            "rows": int(ds.Rows),
            "cols": int(ds.Columns),
            "pixel_spacing": [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])],
            "slice_thickness": float(getattr(ds, "SliceThickness", 0.0)),
        })
    slices.sort(key=lambda s: s["z"])
    return slices


def hu_from_dicom(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def load_slice_hu(slice_info: dict) -> np.ndarray:
    return hu_from_dicom(pydicom.dcmread(slice_info["path"]))


def window_ct(hu_array: np.ndarray, level: float = WINDOW_LEVEL, width: float = WINDOW_WIDTH) -> np.ndarray:
    """V4 brain windowing, normalized to [0, 1]."""
    low = level - width / 2.0
    high = level + width / 2.0
    clipped = np.clip(hu_array, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def build_25d_stack(slices: Sequence[dict], center_pos: int) -> np.ndarray:
    """Build the five-channel V4 2.5D stack with boundary replication."""
    n = len(slices)
    channels = []
    for offset in range(-NEIGHBOR_K, NEIGHBOR_K + 1):
        pos = max(0, min(n - 1, center_pos + offset))
        image = window_ct(load_slice_hu(slices[pos]))
        if image.shape != (512, 512):
            raise ValueError(
                f"Expected 512x512 slices, got {image.shape}: {slices[pos]['path']}"
            )
        channels.append(image)
    return np.stack(channels, axis=0).astype(np.float32)


def local_z_thicknesses(slices: Sequence[dict]) -> List[float]:
    """Exact local-z rule used by V4 compute_case_volumes_ml."""
    n = len(slices)
    zs = [float(s["z"]) for s in slices]
    if n == 1:
        return [1.0]

    thickness = []
    for i in range(n):
        if i == 0:
            t = abs(zs[1] - zs[0])
        elif i == n - 1:
            t = abs(zs[-1] - zs[-2])
        else:
            t = (abs(zs[i + 1] - zs[i]) + abs(zs[i] - zs[i - 1])) / 2.0
        thickness.append(float(t))
    return thickness


def calculate_volumes_ml(predicted_masks: Sequence[np.ndarray], slices: Sequence[dict]) -> Dict[str, float]:
    """Convert per-slice class masks into study-level ICH volumes, matching V4."""
    if len(predicted_masks) != len(slices):
        raise ValueError("Number of masks must equal number of slices.")
    if not slices:
        return {f"V_{name}": 0.0 for name in ICH_CLASSES}

    # V4 uses the first slice's pixel spacing for the case-level calculation.
    ps0, ps1 = slices[0]["pixel_spacing"]
    thicknesses = local_z_thicknesses(slices)
    volumes = {f"V_{name}": 0.0 for name in ICH_CLASSES}

    for mask, thickness in zip(predicted_masks, thicknesses):
        if mask.shape != (512, 512):
            raise ValueError(f"Expected prediction mask (512,512), got {mask.shape}")
        voxel_ml = ps0 * ps1 * thickness / 1000.0
        for class_idx, name in enumerate(ICH_CLASSES, start=1):
            volumes[f"V_{name}"] += int((mask == class_idx).sum()) * voxel_ml
    return volumes


class ICHModel:
    """Four-fold V4 ensemble for ICH segmentation and volume prediction."""

    def __init__(self, checkpoint_paths: Sequence[str | Path], device: Optional[str] = None):
        if len(checkpoint_paths) != 4:
            raise ValueError("Provide exactly four checkpoints: fold0..fold3 best.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: List[torch.nn.Module] = []

        for i, path in enumerate(checkpoint_paths):
            print(f"Loading ICH fold {i}: {path}")
            model = build_model(self.device)
            load_checkpoint_weights(model, path, self.device)
            model.eval()
            self.models.append(model)
        print(f"ICH ensemble ready on {self.device}")

    @torch.inference_mode()
    def predict_case(self, case_dir: str | Path, return_masks: bool = False):
        """Predict one CT study and return the five required ICH volumes."""
        slices = load_sorted_series(case_dir)
        predicted_masks = []

        for pos in range(len(slices)):
            stack = build_25d_stack(slices, pos)
            x = torch.from_numpy(stack).unsqueeze(0).to(self.device)
            probs_sum = None

            for model in self.models:
                probs = torch.softmax(model(x), dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

            pred_mask = torch.argmax(probs_sum / len(self.models), dim=1)[0]
            predicted_masks.append(pred_mask.cpu().numpy().astype(np.uint8))

        volumes = calculate_volumes_ml(predicted_masks, slices)
        return (volumes, predicted_masks) if return_masks else volumes

    def predict_many(self, case_dirs: Sequence[str | Path]) -> Dict[str, Dict[str, float]]:
        return {Path(p).name: self.predict_case(p) for p in case_dirs}

