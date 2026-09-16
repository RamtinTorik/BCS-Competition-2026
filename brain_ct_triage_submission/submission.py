# -*- coding: utf-8 -*-
"""
submission.py — اسکریپت اصلی سابمیت (مرج سه بخش تیم)
======================================================

اجرای رسمی:
    python submission.py --data-dir /path/to/data-dir --predictions-file-path /path/to/submission.csv

خروجی: فایل CSV با دقیقاً این ۸ ستون (به همین ترتیب):
    series_id, V_EDH, V_SDH, V_IPH, V_SAH, V_IVH, fracture_prob, MLS_mm

نحوهٔ کار:
    1) همهٔ فایل‌های `.dcm` زیر `--data-dir` را به‌صورت بازگشتی پیدا
       می‌کند و بر اساس `SeriesInstanceUID` (از هدر خود DICOM) سری‌بندی
       می‌کند. `series_id` خروجی = نام پوشهٔ مشترک برش‌های سری اگر
       یکتا باشد (همان قالب پوشه‌بندی دادهٔ مسابقه)، وگرنه خود UID.
    2) برای هر سری، سه بخش صدا زده می‌شوند:
         - model.py        (شکستگی — وزن fp16 داخل models/fracture/)
         - hemorrhage.py   (خونریزی — وزن‌های تیم خونریزی داخل models/ich/)
         - midline.py      (میدلاین  — وزن‌های تیم میدلاین داخل models/mls/)
       خطای هر بخش فقط همان ستون‌ها را 0.0 می‌کند، نه کل اجرا را.
    3) مدیریت حافظه: در پایان `torch.cuda.empty_cache()` صدا زده می‌شود.

الزام‌ها: آفلاین (هیچ دانلودی)، فقط پکیج‌های پایه، اجرای زیر ۱۵ دقیقه
روی GPU 24GB (همهٔ بخش‌ها batch شده‌اند).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pydicom
import torch

import _dicom_plugins  # noqa: F401  — ثبت افزونه‌های JPEG Lossless قبل از هر dcmread

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

COLUMNS = ["series_id", "V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH",
           "fracture_prob", "MLS_mm"]
HEM_KEYS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"]


# ---------------------------------------------------------------------------
# کشف و سری‌بندی DICOM ها
# ---------------------------------------------------------------------------

def discover_series(data_dir: Path) -> List[Tuple[str, List[Path]]]:
    """گروه‌بندی فایل‌ها بر اساس SeriesInstanceUID هدر DICOM."""
    files = sorted(p for p in data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".dcm")
    if not files:
        raise SystemExit(f"هیچ فایل .dcm در {data_dir} پیدا نشد.")

    groups = {}
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] هدر خوانده نشد، رد شد: {p.name} ({e})")
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or p.stem)
        groups.setdefault(uid, []).append(p)

    if not groups:
        raise SystemExit("هیچ فایل DICOM سالمی خوانده نشد.")

    # series_id خروجی: نام پوشهٔ مشترک (قالب دادهٔ مسابقه) اگر بین سری‌ها
    # یکتا بود؛ وگرنه خود SeriesInstanceUID (که همیشه یکتاست).
    folder_counts = {}
    for uid, paths in groups.items():
        parents = {q.parent.resolve() for q in paths}
        if len(parents) == 1:
            folder = next(iter(parents)).name
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

    series_list = []
    for uid, paths in groups.items():
        parents = {q.parent.resolve() for q in paths}
        folder = next(iter(parents)).name if len(parents) == 1 else None
        sid = folder if folder and folder_counts.get(folder, 0) == 1 else uid
        series_list.append((sid, sorted(paths)))
    series_list.sort(key=lambda x: str(x[0]))
    return series_list


def _load_part(module_name: str, weights_dir: Path):
    """لود یک بخش؛ اگر نبود/خطا خورد، None برمی‌گردد (مقادیر پیش‌فرض 0)."""
    try:
        mod = importlib.import_module(module_name)
        load_fn = getattr(mod, "load_models", None)
        if load_fn is not None:
            load_fn(weights_dir)
        return mod
    except Exception as e:  # noqa: BLE001
        print(f"[warn] ماژول {module_name} قابل استفاده نیست ({e})؛ "
              f"ستون‌های این بخش 0.0 می‌شوند.")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="سابمیت تیم — خروجی CSV با ۸ ستون کمیت واسط")
    parser.add_argument("--data-dir", required=True,
                        help="ریشهٔ دادهٔ تست (جست‌وجوی بازگشتی .dcm)")
    parser.add_argument("--predictions-file-path", required=True,
                        help="مسیر فایل CSV خروجی")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="اندازهٔ batch اینفرنس (پیش‌فرض 8)")
    parser.add_argument("--device", default="auto",
                        help="auto/cuda/cpu (پیش‌فرض: auto)")
    parser.add_argument("--with-mls", action="store_true",
                        help="فعال‌کردن بخش میدلاین. پیش‌فرض خاموش است (MLS_mm=0) "
                             "چون وزن‌های فعلی این بخش در CV خود تیم میدلاین "
                             "case-level MAE ≈ 95mm داشته و تریاژ را خراب می‌کند "
                             "(روی train: macro-F1 با MLS=0 برابر 0.79 و با این "
                             "مدل 0.31 است). کد و وزن‌ها داخل بسته کامل هستند و "
                             "با همین فلگ قابل فعال‌شدن.")
    args = parser.parse_args()

    t0 = time.time()
    series_list = discover_series(Path(args.data_dir))
    n_slices = sum(len(p) for _, p in series_list)
    print(f"{len(series_list)} سری / {n_slices} برش DICOM پیدا شد.")

    # --- بخش شکستگی (پیاده‌سازی کامل) ---
    from model import FracturePredictor
    fracture = FracturePredictor(weights_dir=str(HERE / "models" / "fracture"),
                                 device=args.device, batch_size=args.batch_size)

    # --- بخش خونریزی و میدلاین (کد و وزن‌های تیم‌های مربوطه) ---
    hemi = _load_part("hemorrhage", HERE / "models" / "ich")
    mid = _load_part("midline", HERE / "models" / "mls") if args.with_mls else None
    if mid is None:
        print("[info] بخش میدلاین خاموش است (MLS_mm=0). فعال‌سازی: --with-mls")

    rows = []
    for i, (sid, paths) in enumerate(series_list, 1):
        row = {"series_id": sid, **{k: 0.0 for k in HEM_KEYS},
               "fracture_prob": 0.0, "MLS_mm": 0.0}

        try:
            row["fracture_prob"] = float(fracture.predict_series(paths))
        except Exception as e:  # noqa: BLE001
            print(f"[warn][{sid}] fracture failed: {e}")

        if hemi is not None:
            try:
                vols = hemi.predict_volumes(paths)
                for k in HEM_KEYS:
                    row[k] = float(vols.get(k, 0.0))
            except Exception as e:  # noqa: BLE001
                print(f"[warn][{sid}] hemorrhage failed: {e}")

        if mid is not None:
            try:
                row["MLS_mm"] = float(mid.predict_mls_mm(paths))
            except Exception as e:  # noqa: BLE001
                print(f"[warn][{sid}] midline failed: {e}")

        rows.append(row)
        print(f"[{i}/{len(series_list)}] series_id={sid} "
              f"fracture_prob={row['fracture_prob']:.4f} "
              f"MLS_mm={row['MLS_mm']:.2f} "
              f"V_total={sum(row[k] for k in HEM_KEYS):.2f} "
              f"({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        by="series_id", key=lambda s: s.astype(str)).reset_index(drop=True)

    out_path = Path(args.predictions_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # مدیریت حافظه طبق چک‌لیست مسابقه
    fracture.cleanup()
    if hemi is not None and hasattr(hemi, "_cleanup"):
        try:
            hemi._cleanup()
        except Exception:  # noqa: BLE001
            pass
    if mid is not None and hasattr(mid, "_cleanup"):
        try:
            mid._cleanup()
        except Exception:  # noqa: BLE001
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nCSV نوشته شد: {out_path} | {len(df)} ردیف | {elapsed:.0f} ثانیه")
    print(f"ستون‌ها: {list(df.columns)}")


if __name__ == "__main__":
    main()
