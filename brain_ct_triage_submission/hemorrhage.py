# -*- coding: utf-8 -*-
"""
hemorrhage.py — بخش خونریزی (V_EDH, V_SDH, V_IPH, V_SAH, V_IVH) برای سابمیت
=============================================================================

قرارداد تیم (تعریف‌شده در submission.py):
    load_models(weights_dir)          → یک بار در شروع؛ لود فقط local
    predict_volumes(dicom_files)      → {"V_EDH": .., "V_SDH": .., "V_IPH": ..,
                                         "V_SAH": .., "V_IVH": ..}  (mL)

این ماژول کد و وزن‌های تیم خونریزی (mani) را **دست‌نخورده** استفاده می‌کند:
  - پیش‌پردازش، 2.5D (k=2)، محاسبه‌ی حجم فیزیکی و معماری V4 SegResNet همه از
    `ich_model.py` (فایل اصلی او، بدون تغییر) import می‌شود.
  - چهار چک‌پوینت fold0..fold3 از `models/ich/` لود می‌شوند (خودِ فایل‌های او).
  - اگر MONAI در محیط داوری موجود نباشد، به‌صورت خودکار از پیاده‌سازی
    pure-PyTorch معادل (`segresnet_standalone.py` — با تساوی عددی تأییدشده)
    استفاده می‌شود؛ وزن‌ها همان وزن‌ها هستند.

تنها تفاوت با حلقه‌ی خام او: اینفرنس **batch شده** است (نتیجه‌ی عددی همان
softmax-mean → argmax است؛ فقط سریع‌تر تا محدودیت زمانی مسابقه راحت رعایت شود).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

import _dicom_plugins  # noqa: F401  — ثبت افزونه‌های JPEG Lossless قبل از هر dcmread

HERE = Path(__file__).resolve().parent

# ثابت‌های V4 تیم خونریزی (فایل ich_model.py — بدون تغییر)
ICH_VOLUME_KEYS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"]
NUM_CLASSES = 6
IN_CHANNELS = 5          # 2.5D با k=2
INIT_FILTERS = 16
BLOCKS_DOWN = [1, 2, 2, 4]
BLOCKS_UP = [1, 1, 1]
DROPOUT_PROB = 0.2

_ENSEMBLE: List[torch.nn.Module] = []
_DEVICE = torch.device("cpu")


def _build_one(device: torch.device) -> torch.nn.Module:
    """ساخت مدل V4؛ اولویت با SegResNet خودِ monai (فایل اصلی تیم خونریزی)."""
    try:
        from ich_model import build_model as mani_build
        return mani_build(device)
    except Exception:
        from segresnet_standalone import SegResNetV2
        return SegResNetV2(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES,
                           init_filters=INIT_FILTERS, blocks_down=BLOCKS_DOWN,
                           blocks_up=BLOCKS_UP, dropout_prob=DROPOUT_PROB).to(device)


def load_models(weights_dir) -> None:
    """لود چهار چک‌پوینت fold0..fold3 (فایل‌های zip همان چیزی که تیم خونریزی داد)."""
    global _ENSEMBLE, _DEVICE
    weights_dir = Path(weights_dir)
    paths = sorted(weights_dir.glob("fold*_best.zip"))
    if not paths:
        raise FileNotFoundError(f"هیچ چک‌پوینتی در {weights_dir} پیدا نشد (fold*_best.zip).")

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ENSEMBLE = []
    for p in paths:
        model = _build_one(_DEVICE)
        ck = torch.load(p, map_location=_DEVICE, weights_only=False)
        sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
        model.load_state_dict(sd, strict=True)
        model.eval()
        _ENSEMBLE.append(model)
    print(f"[hemorrhage] {len(_ENSEMBLE)} fold لود شد از {weights_dir} (device={_DEVICE})")


def _cleanup():
    global _ENSEMBLE
    _ENSEMBLE = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_case_slices(case_dir: Path):
    """خواندن و مرتب‌سازی اسلایس‌ها — دقیقاً از فایل اصلی تیم خونریزی."""
    try:
        from ich_model import load_sorted_series
        return load_sorted_series(case_dir)
    except Exception:
        return _load_sorted_series_pure(case_dir)


def _load_sorted_series_pure(case_dir: Path):
    """کپی معادل تابع تیم خونریزی برای حالتی که monai در دسترس نیست."""
    import pydicom
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


def _build_25d_stack(slices: Sequence[dict], center_pos: int) -> np.ndarray:
    try:
        from ich_model import build_25d_stack as f
        return f(slices, center_pos)
    except Exception:
        return _build_25d_stack_pure(slices, center_pos)


def _build_25d_stack_pure(slices: Sequence[dict], center_pos: int) -> np.ndarray:
    """کپی معادل V4: پنجره‌ی WL=40/WW=80، ۵ کانال با boundary replication."""
    import pydicom
    n = len(slices)
    channels = []
    for offset in range(-2, 3):
        pos = max(0, min(n - 1, center_pos + offset))
        ds = pydicom.dcmread(slices[pos]["path"])
        arr = ds.pixel_array.astype(np.float32)
        hu = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        lo, hi = 40.0 - 80.0 / 2.0, 40.0 + 80.0 / 2.0
        img = ((np.clip(hu, lo, hi) - lo) / (hi - lo)).astype(np.float32)
        if img.shape != (512, 512):
            raise ValueError(f"Expected 512x512 slices, got {img.shape}: {slices[pos]['path']}")
        channels.append(img)
    return np.stack(channels, axis=0).astype(np.float32)


def _local_z_thicknesses(slices: Sequence[dict]) -> List[float]:
    try:
        from ich_model import local_z_thicknesses as f
        return f(slices)
    except Exception:
        return _local_z_thicknesses_pure(slices)


def _local_z_thicknesses_pure(slices: Sequence[dict]) -> List[float]:
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


def _calculate_volumes_ml(masks, slices) -> Dict[str, float]:
    try:
        from ich_model import calculate_volumes_ml as f
        return f(masks, slices)
    except Exception:
        return _calculate_volumes_ml_pure(masks, slices)


def _calculate_volumes_ml_pure(masks, slices) -> Dict[str, float]:
    """کپی معادل محاسبه‌ی حجم فیزیکی V4 (voxel_ml = ps0*ps1*t/1000)."""
    class_names = ["IVH", "IPH", "SDH", "EDH", "SAH"]
    if len(masks) != len(slices):
        raise ValueError("Number of masks must equal number of slices.")
    if not slices:
        return {f"V_{n}": 0.0 for n in class_names}
    ps0, ps1 = slices[0]["pixel_spacing"]
    thicknesses = _local_z_thicknesses(slices)
    volumes = {f"V_{n}": 0.0 for n in class_names}
    for mask, thickness in zip(masks, thicknesses):
        voxel_ml = ps0 * ps1 * thickness / 1000.0
        for class_idx, name in enumerate(class_names, start=1):
            volumes[f"V_{name}"] += int((mask == class_idx).sum()) * voxel_ml
    return volumes


@torch.inference_mode()
def _predict_masks_batched(slices: Sequence[dict], batch_size: int = 8):
    """اینفرنس ensembل ۴-فولدی به‌صورت batch شده؛ خروجی = ماسک argmax هر اسلایس."""
    n = len(slices)
    masks = []
    for s0 in range(0, n, batch_size):
        stacks = np.stack([_build_25d_stack(slices, p)
                           for p in range(s0, min(s0 + batch_size, n))], axis=0)
        x = torch.from_numpy(stacks).to(_DEVICE)
        probs = None
        for model in _ENSEMBLE:
            p = torch.softmax(model(x), dim=1)
            probs = p if probs is None else probs + p
        pred = torch.argmax(probs / len(_ENSEMBLE), dim=1)
        masks.extend(m.cpu().numpy().astype(np.uint8) for m in pred)
    return masks


def predict_volumes(dicom_files, batch_size: int = 8) -> Dict[str, float]:
    """ورودی: مسیر فایل‌های DICOM یک سری. خروجی: پنج حجم ICH بر حسب mL."""
    if not _ENSEMBLE:
        load_models(HERE / "models" / "ich")

    parents = sorted({Path(str(f)).resolve().parent for f in dicom_files})
    totals = {k: 0.0 for k in ICH_VOLUME_KEYS}
    for case_dir in parents:
        slices = _load_case_slices(case_dir)
        if not slices:
            continue
        masks = _predict_masks_batched(slices, batch_size=batch_size)
        vols = _calculate_volumes_ml(masks, slices)
        for k in ICH_VOLUME_KEYS:
            totals[k] += float(vols.get(k, 0.0))
    return totals


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
        files = sorted(d.glob("*.dcm"))
        load_models(HERE / "models" / "ich")
        print(json.dumps(predict_volumes(files), indent=2))
