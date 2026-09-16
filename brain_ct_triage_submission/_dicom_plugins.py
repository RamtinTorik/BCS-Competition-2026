# -*- coding: utf-8 -*-
"""
_dicom_plugins.py — ثبت افزونه‌های رمزگشای DICOM (JPEG Lossless و ...)
=======================================================================

بخشی از برش‌های دادهٔ مسابقه با Transfer Syntax ‏`JPEG Lossless (1.2.840.10008.1.2.4.70)`
ذخیره شده‌اند. pydicom برای رمزگشایی آن‌ها به افزونه‌ی `gdcm` یا `pylibjpeg`
نیاز دارد؛ import همین ماژول در ابتدای submission.py / model.py باعث می‌شود
افزونه‌های موجود در محیط، یک بار و به‌صورت سراسری ثبت شوند و همهٔ کدها
(شامل model.py و ich_model.py تیم‌ها که مستقیم `ds.pixel_array` می‌خوانند)
بدون هیچ تغییری بتوانند برش‌های فشرده را بخوانند.

اگر هیچ افزونه‌ای در محیط نباشد، این ماژول بی‌صدا رد می‌شود (رفتار pydicom
پیش‌فرض خواهد بود) و برش ناخوانا فقط همان برش را رد می‌کند، نه کل اجرا را.
"""
from __future__ import annotations

import importlib

_CANDIDATES = (
    "pylibjpeg.libjpeg",   # افزونه‌ی libjpeg برای pylibjpeg
    "pylibjpeg",           # خود pylibjpeg (هندلر pydicom 2.x)
    "gdcm",                # python-gdcm
)

_loaded = []


def load_dicom_plugins() -> list:
    """افزونه‌های موجود را import و ثبت می‌کند؛ لیست نام‌های لودشده برمی‌گردد."""
    global _loaded
    if _loaded:
        return _loaded
    for name in _CANDIDATES:
        try:
            importlib.import_module(name)
            _loaded.append(name)
        except Exception:  # noqa: BLE001 — افزونه‌ی نبود problem نیست
            pass
    # اطمینان از ثبت هندلرهای داخلی pydicom
    try:
        import pydicom.pixels.processors  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    return _loaded


load_dicom_plugins()
