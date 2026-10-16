"""ประกอบเว็บแอปแบบรันในเบราว์เซอร์ทั้งหมดลง standalone/dist/

dist/page.html          หน้าเว็บ (ฝัง ONNX Runtime Web + โค้ดแอป)
dist/models/*.b64.N.txt โมเดลตรวจจับ/จดจำใบหน้า (InsightFace buffalo_s) เข้ารหัส base64
dist/ort/*.wasm         ตัวรัน ONNX (WebAssembly)
dist/index.html         หน้าเดียวกันห่อด้วย <html> สำหรับเปิดทดสอบในเครื่อง (python -m http.server -d dist)

ใช้: python standalone/build.py
"""
import base64
import io
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
DIST = HERE / "dist"
BUFFALO_S = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
ORT_TGZ = "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.17.3.tgz"
MODELS = ("det_500m.onnx", "w600k_mbf.onnx")
ORT_FILES = ("ort.wasm.min.js", "ort-wasm-simd.wasm", "ort-wasm.wasm")


def fetch(url):
    print("ดาวน์โหลด", url)
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def ensure_assets():
    CACHE.mkdir(exist_ok=True)
    if not all((CACHE / m).exists() for m in MODELS):
        with zipfile.ZipFile(io.BytesIO(fetch(BUFFALO_S))) as z:
            for m in MODELS:
                (CACHE / m).write_bytes(z.read(m))
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
    # .onnx ไม่ใช่ชนิดไฟล์ที่ Artifact ให้บริการ จึงเก็บเป็น base64 ใน .txt (ส่วนละ ≤ ~9 MB)
    for m in MODELS:
        b64 = base64.b64encode((CACHE / m).read_bytes()).decode()
        step = 9_500_000 // 4 * 4
        for i, off in enumerate(range(0, len(b64), step)):
            (DIST / "models" / f"{m[:-5]}.b64.{i}.txt").write_text(b64[off:off + step])
    for f in ORT_FILES[1:]:
        shutil.copy(CACHE / f, DIST / "ort" / f)

    ort_js = (CACHE / "ort.wasm.min.js").read_text(encoding="utf-8")
    ort_js = ort_js.replace("</script", "<\\/script").replace("//# sourceMappingURL=ort.wasm.min.js.map", "")
    app_js = (HERE / "app.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    page = (HERE / "page.src.html").read_text(encoding="utf-8")
    page = page.replace("/*__ORT__*/", ort_js).replace("/*__APP__*/", app_js)
    (DIST / "page.html").write_text(page, encoding="utf-8")

    # สำหรับเปิดทดสอบในเครื่อง: ห่อด้วยโครงเอกสารแบบเดียวกับที่ Artifact เติมให้ตอนเผยแพร่
    wrapped = ('<!doctype html><html lang="th"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>'
               + page + "</body></html>")
    (DIST / "index.html").write_text(wrapped, encoding="utf-8")
    total = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"สร้างเสร็จที่ {DIST} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
