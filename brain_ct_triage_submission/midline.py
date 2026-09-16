# -*- coding: utf-8 -*-
"""
midline.py — بخش شیفت خط وسط (MLS_mm) برای سابمیت
===================================================

قرارداد تیم (تعریف‌شده در submission.py):
    load_models(weights_dir)      → یک بار در شروع؛ لود فقط local
    predict_mls_mm(dicom_files)   → float  (mm)

معماری و وزن‌های تیم میدلاین (javad) **بدون تغییر** استفاده می‌شوند:
  - U-Net با انکودر ResNet34، ورودی ۱ کانال، خروجی ۳ کانال heatmap
    (AnteriorFalxAttachment / PosteriorFalxAttachment / OutermostPointOfTheFalx)
  - ۵ چک‌پوینت fold (`cv_fold1..5_best.zip`) از `models/mls/` به‌صورت ensemble
  - decode مطابق نوت‌بوک ترین او: sigmoid → argmax هر heatmap → مختصات،
    فاصله‌ی نقطه‌ی Outermost از خط Anterior-Posterior × فاصله‌ی پیکسل = MLS
  - تجمیع سطح سری: بیشینه روی همه‌ی اسلایس‌ها (همان case-level نوت‌بوک)

معماری از `smpunet_standalone.py` (معادل smp.Unet با تساوی عددی تأییدشده)
لود می‌شود تا هیچ وابستگی اضافه‌ای در محیط داوری لازم نباشد.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pydicom
import torch

import _dicom_plugins  # noqa: F401  — ثبت افزونه‌های JPEG Lossless قبل از هر dcmread

HERE = Path(__file__).resolve().parent

KEYPOINT_NAMES = ["AnteriorFalxAttachment", "PosteriorFalxAttachment", "OutermostPointOfTheFalx"]
IN_SIZE = 256                    # ورودی مدل در ترین
WINDOW_LEVEL = 40.0              # پنجره‌ی مغز — مطابق پیش‌پردازش تست‌شده با وزن‌ها
WINDOW_WIDTH = 80.0
N_FOLDS = 5
MLS_CLAMP_MM = 25.0              # سقف فیزیولوژیک MLS — جلوی خروجی‌های non-sense decode را می‌گیرد

_MODELS: List[torch.nn.Module] = []
_DEVICE = torch.device("cpu")


def load_models(weights_dir) -> None:
    global _MODELS, _DEVICE
    weights_dir = Path(weights_dir)
    paths = sorted(weights_dir.glob("cv_fold*_best.zip"))
    if not paths:
        raise FileNotFoundError(f"هیچ چک‌پوینتی در {weights_dir} پیدا نشد (cv_fold*_best.zip).")

    from smpunet_standalone import MidlineHeatmapModel

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODELS = []
    for p in paths:
        model = MidlineHeatmapModel().to(_DEVICE)
        sd = torch.load(p, map_location=_DEVICE, weights_only=True)
        model.load_state_dict(sd, strict=True)
        model.eval()
        _MODELS.append(model)
    print(f"[midline] {len(_MODELS)} fold لود شد از {weights_dir} (device={_DEVICE})")


def _cleanup():
    global _MODELS
    _MODELS = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slice_hu(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    return arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))


def _window(hu_arr: np.ndarray) -> np.ndarray:
    lo, hi = WINDOW_LEVEL - WINDOW_WIDTH / 2.0, WINDOW_LEVEL + WINDOW_WIDTH / 2.0
    return ((np.clip(hu_arr, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def _mls_from_points(pts, spacing) -> float:
    """pts: [(x,y)×3] در مختصات پیکسلِ ابعاد اصلی؛ MLS = فاصله‌ی عمود O از خط A-P (mm)."""
    (ax, ay), (px, py), (ox, oy) = pts
    dx, dy = float(px) - float(ax), float(py) - float(ay)
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0
    cross = dx * (float(oy) - float(ay)) - dy * (float(ox) - float(ax))
    px_mm = (float(spacing[0]) + float(spacing[1])) / 2.0
    return abs(cross) / norm * px_mm


@torch.inference_mode()
def _predict_series_mls(dicom_files: Sequence, batch_size: int = 16) -> float:
    """بیشینه‌ی MLSdecoded روی همه‌ی اسلایس‌های سری (case-level نوت‌بوک)."""
    files = sorted(str(f) for f in dicom_files)
    best = 0.0
    for s0 in range(0, len(files), batch_size):
        chunk = files[s0:s0 + batch_size]
        tensors, metas = [], []
        for fp in chunk:
            try:
                ds = pydicom.dcmread(fp)
                img = _window(_slice_hu(ds))
                t = torch.from_numpy(img[None, None])
                t = torch.nn.functional.interpolate(t, size=(IN_SIZE, IN_SIZE),
                                                    mode="bilinear", align_corners=False)
                tensors.append(t)
                metas.append(([float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])],
                              float(img.shape[0]) / IN_SIZE))
            except Exception as e:  # noqa: BLE001 — برش خراب کل سری را نمی‌اندازد
                print(f"[midline][warn] برش رد شد {Path(fp).name}: {e}")
        if not tensors:
            continue

        x = torch.cat(tensors, dim=0).to(_DEVICE)
        hm = None
        for model in _MODELS:
            p = torch.sigmoid(model(x))
            hm = p if hm is None else hm + p
        hm = (hm / len(_MODELS)).cpu()

        for b in range(hm.shape[0]):
            (spacing, scale) = metas[b]
            pts = []
            for c in range(3):
                idx = hm[b, c].argmax().item()
                y, xx = divmod(idx, IN_SIZE)
                pts.append((xx * scale, y * scale))
            best = max(best, _mls_from_points(pts, spacing))
    return float(min(max(best, 0.0), MLS_CLAMP_MM))


def predict_mls_mm(dicom_files, batch_size: int = 16) -> float:
    if not _MODELS:
        load_models(HERE / "models" / "mls")
    return _predict_series_mls(dicom_files, batch_size=batch_size)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
        files = sorted(d.glob("*.dcm"))
        load_models(HERE / "models" / "mls")
        print(f"MLS_mm = {predict_mls_mm(files):.3f}")
