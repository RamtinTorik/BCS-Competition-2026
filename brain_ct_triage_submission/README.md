# بستهٔ سابمیت — Brain CT Triage Challenge

این بسته دقیقاً مطابق قالب رسمی مسابقه است: **`model.py`** (API استاندارد) +
**`models/`** (همهٔ وزن‌ها) + **`submission.py`** (CLI رسمی تولید CSV).

## اجرای رسمی

```bash
python submission.py --data-dir /path/to/data-dir --predictions-file-path /path/to/submission.csv
```

`data-dir` ریشهٔ دادهٔ تست است (قالب `dicom/{dicom_series.id}/*.dcm` یا هر ساختاری که
فایل‌های `.dcm` داخلش باشند — سری‌بندی از `SeriesInstanceUID` هدر DICOM انجام می‌شود).

خروجی CSV دقیقاً با ۸ ستون رسمی:
`series_id, V_EDH, V_SDH, V_IPH, V_SAH, V_IVH, fracture_prob, MLS_mm`

## API استاندارد (پایپ‌لاین داوری)

```python
from model import Model

model = Model()                       # یک بار؛ همهٔ وزن‌ها لود می‌شود
inter = model.predict("dicom/1011")   # study_dir = "dicom/{dicom_series.id}"
# → {"V_EDH": .., "V_SDH": .., "V_IPH": .., "V_SAH": .., "V_IVH": ..,
#     "fracture_prob": .., "MLS_mm": ..}   (ورودی رسمی triage_from_intermediates)
```

## ساختار

```
├── submission.py                     # CLI رسمی (مرج سه بخش → CSV)
├── model.py                          # API استاندارد + بخش شکستگی (FracturePredictor)
├── hemorrhage.py                     # بخش خونریزی (V_EDH..V_IVH) — ۴-fold ensemble
├── midline.py                        # بخش شیفت میدلاین (MLS_mm) — ۵-fold ensemble
├── ich_model.py                      # کد اصلی تیم خونریزی (بدون تغییر)
├── segresnet_standalone.py           # معماری SegResNet بدون نیاز به monai (تساوی عددی تأییدشده)
├── smpunet_standalone.py             # معماری U-Net/ResNet34 بدون نیاز به smp (تساوی عددی تأییدشده)
├── triage.py                         # کپی رسمی triage_from_intermediates (برای debug)
├── models/
│   ├── fracture/                     # تیم شکستگی (ایمان) — Faster R-CNN + ResNet18
│   │   ├── fracture_detector_best.pth
│   │   ├── fracture_classifier_best.pth
│   │   └── weights_manifest.json
│   ├── ich/                          # تیم خونریزی (mani) — ۴ fold SegResNet
│   │   └── fold0..3_best.zip
│   └── mls/                          # تیم میدلاین (javad) — ۵ fold U-Net
│       └── cv_fold1..5_best.zip
└── pyproject.toml                    # فقط پکیج‌های پایه
```

## بخش‌ها

| بخش | خروجی | مدل | وزن‌ها |
|---|---|---|---|
| شکستگی | `fracture_prob` (0..1) | Faster R-CNN ResNet50-FPN + ResNet18 classifier | `models/fracture/*.pth` (fp16) |
| خونریزی | `V_EDH..V_IVH` (mL) | ۴-fold SegResNet (V4، 2.5D، WL40/WW80) | `models/ich/fold*_best.zip` |
| میدلاین | `MLS_mm` (mm) | ۵-fold U-Net ResNet34 heatmap (۳ keypoint falx) — **پیش‌فرض خاموش** (`--with-mls` برای فعال‌سازی) | `models/mls/cv_fold*_best.zip` |

## چرا بخش میدلاین پیش‌فرض خاموش است؟

وزن‌های فعلی تیم میدلاین در CV خودِ آن تیم **case-level MAE ≈ 95mm** داشته‌اند
(مدل با lr=1e-5 و ۱۵ epoch عملاً همگرا نشده). تأثیر روی تریاژ (شبیه‌سازی روی
۳۳۸ سری train با حجم‌ها و شکستگی GT و تابع رسمی تریاژ):

| استراتژی MLS | macro-F1 | QWK |
|---|---|---|
| MLS = 0 (پیش‌فرض فعلی بسته) | **0.791** | **0.886** |
| MLS از مدل فعلی (clamp به 25mm) | 0.311 | 0.556 |

تنها ~۴٪ سری‌ها صرفاً به MLS وابسته‌اند؛ به همین دلیل تا اصلاح مدل، خروجی MLS=0
امتیاز بهتری می‌دهد. کد و وزن‌های این بخش کامل داخل بسته هستند و با
`--with-mls` (در CLI) یا `Model(use_mls=True)` قابل فعال‌شدن‌اند.

## الزامات مسابقه

- **آفلاین**: هیچ دانلودی در زمان اجرا انجام نمی‌شود — `weights=None` و
  `weights_backbone=None` همه‌جا (بدون این، torchvision وزن ImageNet بک‌بون را
  دانلود می‌کند و در محیط بدون اینترنت داوری خطای URLError می‌دهد)؛ وزن‌ها فقط
  از `models/` لود می‌شوند. این رفتار با پروکسی قطع‌شده و کش خالی torch تست شده است.
- **بدون وابستگی اضافه**: فقط `torch / torchvision / pydicom / numpy / pandas`.
  معماری‌های SegResNet و SMP-U-Net به‌صورت pure-PyTorch داخل بسته هستند (تساوی
  عددی خروجی با monai 1.6 و segmentation-models-pytorch 0.5 به‌صورت محلی تأیید شد:
  max|diff| = 0.0)؛ اگر آن کتابخانه‌ها در ایمیج داوری باشند همان‌ها استفاده می‌شوند.
- **زمان**: هر سه بخش batch-شده‌اند؛ روی GPU تک‌کارت 24GB هدف زیر ۱۵ دقیقه.
- **مقاوم‌سازی**: خطای هر بخش فقط ستون‌های همان بخش را 0.0 می‌کند؛ برش خراب فقط
  همان برش را رد می‌کند؛ در پایان `torch.cuda.empty_cache()` صدا زده می‌شود.
- **تریاژ**: پیش‌بینی کلاس تریاژ در کد انجام نمی‌شود — فقط ۷ کمیت واسط؛ کلاس تریاژ
  با تابع رسمی مسابقه (`triage.py` — کپی بدون تغییر) از همین کمیت‌ها محاسبه می‌شود.
