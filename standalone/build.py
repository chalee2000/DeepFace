"""ประกอบเว็บแอปแบบรันในเบราว์เซอร์ทั้งหมดลง standalone/dist/

dist/page.html             หน้าเว็บ (ฝัง ONNX Runtime Web + โค้ดแอป + รายการไฟล์โมเดล)
dist/models/*.b64.N.txt    โมเดลเข้ารหัส base64 แบ่งส่วนละ ≤ 15 MB (บริการโฮสต์ไม่รับไฟล์ .onnx)
                           - det_500m      ตรวจจับใบหน้า + จุด 5 จุด (SCRFD, InsightFace buffalo_s)
                           - 2d106det      จุดโครงหน้า 106 จุด (InsightFace)
                           - w600k_r50_q8  จดจำใบหน้า ArcFace ResNet50 (WebFace600K) บีบอัดเป็น int8
dist/ort/*.wasm            ตัวรัน ONNX (WebAssembly)
dist/index.html            หน้าเดียวกันห่อด้วย <html> สำหรับเปิดทดสอบในเครื่อง (python -m http.server -d dist)

ใช้: python standalone/build.py   (ต้องมี onnx และ onnxruntime สำหรับขั้นบีบอัดโมเดล)
"""
import base64
import io
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
DIST = HERE / "dist"
RELEASE = "https://github.com/deepinsight/insightface/releases/download/v0.7/"
ORT_TGZ = "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.17.3.tgz"
ORT_FILES = ("ort.wasm.min.js", "ort-wasm-simd.wasm", "ort-wasm.wasm")
PART_CHARS = 15_000_000 // 4 * 4  # ขนาดต่อส่วน (ตัวอักษร base64) — ไฟล์ข้อความรับได้ไม่เกิน 16 MB
# ชื่อในหน้าเว็บ -> ไฟล์ใน .cache
MODELS = {"det": "det_500m.onnx", "lmk": "2d106det.onnx", "rec": "w600k_r50_q8.onnx"}


def fetch(url):
    print("ดาวน์โหลด", url)
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def ensure_assets():
    CACHE.mkdir(exist_ok=True)
    if not all((CACHE / n).exists() for n in ("det_500m.onnx", "2d106det.onnx")):
        with zipfile.ZipFile(io.BytesIO(fetch(RELEASE + "buffalo_s.zip"))) as z:
            for n in ("det_500m.onnx", "2d106det.onnx"):
                (CACHE / n).write_bytes(z.read(n))
    if not (CACHE / "w600k_r50_q8.onnx").exists():
        if not (CACHE / "w600k_r50.onnx").exists():
            with zipfile.ZipFile(io.BytesIO(fetch(RELEASE + "buffalo_l.zip"))) as z:
                (CACHE / "w600k_r50.onnx").write_bytes(z.read("w600k_r50.onnx"))
        # บีบอัดน้ำหนักเป็น 8 บิต: 174 MB -> ~44 MB, เวกเตอร์ใบหน้าตรงกับตัวเต็ม cos ≥ 0.99
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(CACHE / "w600k_r50.onnx"), str(CACHE / "w600k_r50_q8.onnx"), weight_type=QuantType.QUInt8)
    if not all((CACHE / f).exists() for f in ORT_FILES):
        with tarfile.open(fileobj=io.BytesIO(fetch(ORT_TGZ)), mode="r:gz") as t:
            for f in ORT_FILES:
                (CACHE / f).write_bytes(t.extractfile(f"package/dist/{f}").read())


def main():
    ensure_assets()
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "models").mkdir(parents=True)
    (DIST / "ort").mkdir()
    manifest = {}
    for key, fname in MODELS.items():
        b64 = base64.b64encode((CACHE / fname).read_bytes()).decode()
        stem = fname[:-5]
        parts = []
        for i, off in enumerate(range(0, len(b64), PART_CHARS)):
            chunk = b64[off:off + PART_CHARS]
            path = f"models/{stem}.b64.{i}.txt"
            (DIST / path).write_text(chunk)
            parts.append({"path": path, "size": len(chunk)})
        manifest[key] = parts
    for f in ORT_FILES[1:]:
        shutil.copy(CACHE / f, DIST / "ort" / f)

    ort_js = (CACHE / "ort.wasm.min.js").read_text(encoding="utf-8")
    ort_js = ort_js.replace("</script", "<\\/script").replace("//# sourceMappingURL=ort.wasm.min.js.map", "")
    app_js = (HERE / "app.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    app_js = app_js.replace("/*__MANIFEST__*/null", json.dumps(manifest))
    page = (HERE / "page.src.html").read_text(encoding="utf-8")
    page = page.replace("/*__ORT__*/", ort_js).replace("/*__APP__*/", app_js)
    (DIST / "page.html").write_text(page, encoding="utf-8")

    # สำหรับเปิดทดสอบในเครื่อง: ห่อด้วยโครงเอกสารแบบเดียวกับที่ Artifact เติมให้ตอนเผยแพร่
    wrapped = ('<!doctype html><html lang="th"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>'
               + page + "</body></html>")
    (DIST / "index.html").write_text(wrapped, encoding="utf-8")
    total = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(json.dumps(manifest, indent=1))
    print(f"สร้างเสร็จที่ {DIST} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
