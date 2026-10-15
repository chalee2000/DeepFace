"""ดาวน์โหลดโมเดล InsightFace buffalo_l (เฉพาะไฟล์ที่ใช้) ไปไว้ที่ models/

ใช้: python scripts/download_models.py
หมายเหตุ: โมเดล InsightFace ให้ใช้เพื่อการวิจัย/ไม่ใช่เชิงพาณิชย์ — โปรดตรวจสอบสิทธิ์การใช้งานก่อนนำไปใช้จริง
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
NEEDED = ("det_10g.onnx", "w600k_r50.onnx")
DEST = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    DEST.mkdir(exist_ok=True)
    if all((DEST / n).exists() for n in NEEDED):
        print("มีโมเดลครบแล้ว:", ", ".join(NEEDED))
        return 0
    print("กำลังดาวน์โหลด buffalo_l (~280 MB) …")
    with urllib.request.urlopen(URL, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in NEEDED:
            (DEST / n).write_bytes(z.read(n))
            print("บันทึก", DEST / n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
