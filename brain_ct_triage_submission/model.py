# -*- coding: utf-8 -*-
"""
model.py — بخش «شکستگی جمجمه» (fracture_prob) برای فایل سابمیت مسابقه
======================================================================

خودکفا و آفلاین: فقط `torch`/`torchvision`/`pydicom`/`numpy` و هیچ وزنی از
اینترنت دانلود نمی‌شود.

معماری (نتیجهٔ ترین ۵-فولدی پروژه):
    1) Detector: `Faster R-CNN ResNet50-FPN` با anchor های کوچک
       `(16, 32, 64, 128, 256)` — `num_classes=2` (bg + fracture).
    2) Classifier: `ResNet18` روی crop ثابت `64×64` هر box پیشنهادی.
    3) امتیاز نهایی هر box: `detector_score × classifier_score`.

پیش‌پردازش هر برش DICOM (بدون تغییر نسبت آموزش):
    `HU = pixel × RescaleSlope + RescaleIntercept` → clip به
    `[-3024, 1726]` → سه پنجرهٔ `brain (40/120)`، `bone (300/1500)`،
    `subdural (50/350)` به‌عنوان ۳ کانال در بازهٔ `[0,1]`.

وزن‌ها (fp16، برای کاهش حجم — حین لود به fp32 برمی‌گردند):
    `models/fracture/fracture_detector_best.pth`
    `models/fracture/fracture_classifier_best.pth`
    (جزئیات در `models/fracture/weights_manifest.json`)

علاوه بر بخش شکستگی، در انتهای همین فایل کلاس `Model` (API استاندارد مسابقه)
تعریف شده که هر سه بخش (شکستگی + خونریزی + میدلاین) را تجمیع می‌کند و
خروجی `predict(study_dir)` آن دیکشنری ۷ کلیدی کمیت‌های واسط است.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
import torch

import _dicom_plugins  # noqa: F401  — ثبت افزونه‌های JPEG Lossless قبل از هر dcmread
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ---------------------------------------------------------------------------
# ثابت‌ها — دقیقاً مطابق ترین
# ---------------------------------------------------------------------------

NUM_CLASSES = 2                     # 0=background، 1=fracture
CLIP_RANGE: Tuple[float, float] = (-3024.0, 1726.0)
WINDOW_CONFIGS: Dict[str, Tuple[float, float]] = {
    "brain": (40.0, 120.0),
    "bone": (300.0, 1500.0),
    "subdural": (50.0, 350.0),
}
WINDOW_ORDER: Tuple[str, str, str] = ("brain", "bone", "subdural")

# معماری برنده: anchor کوچک (اجباری برای سازگاری با وزن‌های ترین‌شده)
ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)

CROP_SIZE_DEFAULT = 64


# ---------------------------------------------------------------------------
# پیش‌پردازش DICOM (ورودی → تنسور ۳-کاناله)
# ---------------------------------------------------------------------------

def dicom_to_hu(dicom_path) -> np.ndarray:
    ds = pydicom.dcmread(str(dicom_path))
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = pixels * slope + intercept
    return np.clip(hu, CLIP_RANGE[0], CLIP_RANGE[1])


def apply_window(hu_array: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(hu_array, low, high)
    return ((windowed - low) / (high - low)).astype(np.float32)


def load_dicom_as_tensor(dicom_path) -> torch.Tensor:
    """مسیر DICOM → تنسور `(3, H, W)` در بازهٔ `[0,1]` (ترتیب کانال‌ها مثل ترین)."""
    hu = dicom_to_hu(dicom_path)
    channels = [apply_window(hu, *WINDOW_CONFIGS[name]) for name in WINDOW_ORDER]
    return torch.from_numpy(np.stack(channels, axis=0)).float()


# ---------------------------------------------------------------------------
# تعریف معماری (weights=None همیشه — هیچ دانلود اینترنتی در کار نیست)
# ---------------------------------------------------------------------------

def build_detector(num_classes: int = NUM_CLASSES,
                   small_anchors: bool = True) -> torch.nn.Module:
    """`Faster R-CNN ResNet50-FPN` با anchor کوچک — مطابق وزن‌های ترین‌شده.

    نکته: `weights_backbone=None` الزامی است — در غیر این صورت torchvision
    به‌صورت پیش‌فرض وزن ImageNet بک‌بون را دانلود می‌کند و در محیط داوری
    (بدون اینترنت) اجرا با URLError می‌سوزد.
    """
    kwargs = {}
    if small_anchors:
        kwargs["rpn_anchor_generator"] = AnchorGenerator(ANCHOR_SIZES, ASPECT_RATIOS)
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                    trainable_backbone_layers=0, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_classifier() -> torch.nn.Module:
    """`ResNet18` دوارده — خروجی ۲ کلاس (0=no-fracture، 1=fracture)."""
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def _load_state_float(ckpt: dict) -> dict:
    """state_dict ذخیره‌شده به fp16 → fp32 (سازگار با معماری)."""
    return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in ckpt["model"].items()}


# ---------------------------------------------------------------------------
# Predictor سری-محور
# ---------------------------------------------------------------------------

class FracturePredictor:
    """
    بارگذاری یک بار در شروع؛ سپس برای هر سری DICOM فقط
    `predict_series(dicom_files)` صدا زده می‌شود.

    `fracture_prob` سری = بیشینهٔ `detector_score × classifier_score`
    روی همهٔ برش‌ها و همهٔ boxهای پیشنهادی آن سری (همان فرمول
    تجمیع پروژه؛ در سطح بیمار/سری معتبرترین تجمیع بود).

    اینفرنس batch شده است (دیتکتور batch های ۸تایی + کلاسیفایر یک
    forward برای همهٔ cropهای هر batch) تا محدودیت زمانی ۱۵-۳۰ دقیقه
    با خیال راحت رعایت شود.
    """

    DETECTOR_NAME = "fracture_detector_best.pth"
    CLASSIFIER_NAME = "fracture_classifier_best.pth"

    def __init__(self, weights_dir: Optional[str] = None,
                 device: str = "auto",
                 batch_size: int = 8,
                 detector_ckpt: Optional[str] = None,
                 classifier_ckpt: Optional[str] = None):
        base = Path(weights_dir) if weights_dir else Path(__file__).resolve().parent / "models" / "fracture"
        det_path = Path(detector_ckpt) if detector_ckpt else base / self.DETECTOR_NAME
        cls_path = Path(classifier_ckpt) if classifier_ckpt else base / self.CLASSIFIER_NAME
        for p in (det_path, cls_path):
            if not p.exists():
                raise FileNotFoundError(f"وزن پیدا نشد: {p}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        det_ckpt = torch.load(det_path, map_location="cpu", weights_only=True)
        self.detector = build_detector(
            small_anchors=bool(det_ckpt.get("use_small_anchors", True)))
        self.detector.load_state_dict(_load_state_float(det_ckpt))
        self.detector.to(self.device).eval()

        cls_ckpt = torch.load(cls_path, map_location="cpu", weights_only=True)
        self.classifier = build_classifier()
        self.classifier.load_state_dict(_load_state_float(cls_ckpt))
        self.classifier.to(self.device).eval()
        self.crop_size = int(cls_ckpt.get("crop_size", CROP_SIZE_DEFAULT))

        n_det = sum(p.numel() for p in self.detector.parameters())
        n_cls = sum(p.numel() for p in self.classifier.parameters())
        print(f"[fracture] مدل‌ها لود شدند (device={self.device}, "
              f"crop_size={self.crop_size}) | detector params={n_det/1e6:.1f}M, "
              f"classifier params={n_cls/1e6:.1f}M")

    @torch.inference_mode()
    def predict_series(self, dicom_files) -> float:
        """
        ورودی: لیست مسیرهای DICOM یک سری (هر ترتیبی).
        خروجی: `fracture_prob` سری — float در `[0,1]`.
        برش‌های خراب فقط اخطار می‌دهند و رد می‌شوند (کل سری نمی‌پزد).
        """
        tensors: List[torch.Tensor] = []
        for f in dicom_files:
            try:
                tensors.append(load_dicom_as_tensor(f))
            except Exception as e:  # noqa: BLE001
                print(f"[fracture][warn] برش رد شد {Path(str(f)).name}: {e}")

        best = 0.0
        for s in range(0, len(tensors), self.batch_size):
            chunk = tensors[s: s + self.batch_size]
            outputs = self.detector([t.to(self.device) for t in chunk])

            crops, det_scores = [], []
            for img, out in zip(chunk, outputs):
                h, w = int(img.shape[-2]), int(img.shape[-1])
                for box, score in zip(out["boxes"].tolist(), out["scores"].tolist()):
                    x1, y1, x2, y2 = (int(round(v)) for v in box)
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    crops.append(TF.resize(img[:, y1:y2, x1:x2],
                                           [self.crop_size, self.crop_size],
                                           antialias=True))
                    det_scores.append(float(score))

            if crops:
                probs = F.softmax(self.classifier(torch.stack(crops).to(self.device)),
                                  dim=1)[:, 1].tolist()
                best = max(best, max(d * p for d, p in zip(det_scores, probs)))

        return float(min(best, 1.0))

    def cleanup(self):
        """آزادسازی حافظهٔ GPU (الزام مسابقه: مدیریت `empty_cache`)."""
        del self.detector, self.classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model API استاندارد مسابقه — تجمیع هر سه بخش تیم
# ---------------------------------------------------------------------------

TRIAGE_REQUIRED_KEYS = ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH",
                        "fracture_prob", "MLS_mm")


class Model:
    """
    API استاندارد مسابقه:

        from model import Model
        model = Model()                     # یک بار — همهٔ وزن‌ها لود می‌شود
        inter = model.predict(study_dir)    # study_dir = "dicom/{dicom_series.id}"

    خروجی `predict` دیکشنری با دقیقاً ۷ کلید زیر است (ورودی تابع رسمی
    `triage_from_intermediates`):

        V_EDH, V_SDH, V_IPH, V_SAH, V_IVH  (mL)
        fracture_prob                      (0..1)
        MLS_mm                             (mm)
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = "auto",
                 batch_size: int = 8, use_mls: bool = False):
        base = Path(model_dir) if model_dir else Path(__file__).resolve().parent / "models"
        self._fracture = FracturePredictor(weights_dir=str(base / "fracture"),
                                           device=device, batch_size=batch_size)

        import hemorrhage as _hem
        import midline as _mid
        self._hem, self._mid = _hem, _mid
        _hem.load_models(base / "ich")
        # پیش‌فرض use_mls=False: وزن‌های فعلی بخش میدلاین در CV خودِ تیم میدلاین
        # case-level MAE ≈ 95mm داشته‌اند و فعال‌بودنشان تریاژ را خراب می‌کند
        # (روی train: macro-F1 با MLS=0 برابر 0.79 و با این مدل 0.31 است).
        # کد و وزن‌ها کامل داخل بسته هستند و با use_mls=True فعال می‌شوند.
        self._use_mls = use_mls
        if use_mls:
            _mid.load_models(base / "mls")

    def list_dicom_files(self, study_dir) -> List[Path]:
        study_dir = Path(str(study_dir))
        files = sorted(p for p in study_dir.glob("*.dcm") if p.is_file())
        if not files:  # ساختار تودرتو هم پشتیبانی می‌شود
            files = sorted(p for p in study_dir.rglob("*.dcm") if p.is_file())
        if not files:
            raise FileNotFoundError(f"هیچ فایل .dcm در {study_dir} پیدا نشد.")
        return files

    def predict(self, study_dir) -> Dict[str, float]:
        files = self.list_dicom_files(study_dir)
        out = {k: 0.0 for k in TRIAGE_REQUIRED_KEYS}

        try:
            out["fracture_prob"] = float(self._fracture.predict_series(files))
        except Exception as e:  # noqa: BLE001 — خطای یک بخش فقط ستون خودش را صفر می‌کند
            print(f"[model][warn] fracture failed: {e}")

        try:
            vols = self._hem.predict_volumes(files)
            for k in ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"):
                out[k] = float(vols.get(k, 0.0))
        except Exception as e:  # noqa: BLE001
            print(f"[model][warn] hemorrhage failed: {e}")

        if self._use_mls:
            try:
                out["MLS_mm"] = float(self._mid.predict_mls_mm(files))
            except Exception as e:  # noqa: BLE001
                print(f"[model][warn] midline failed: {e}")

        return out

    def cleanup(self):
        self._fracture.cleanup()
        self._hem._cleanup()
        self._mid._cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
