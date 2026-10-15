"""ทดสอบแกนหลักด้วยภาพสังเคราะห์ + (ถ้ามี) ภาพจริงใน tests/samples/

ภาพจริงไม่ได้ commit ไว้ — วางไฟล์ a1.jpg, a2.jpg (คนเดียวกัน) และ b.jpg (คนละคน) ที่ tests/samples/ เพื่อรันชุดทดสอบเต็ม
"""
from pathlib import Path

import numpy as np
import pytest

from app import engine as eng

SAMPLES = Path(__file__).parent / "samples"


def test_band_boundaries():
    assert eng.band_for(77)["level"] == "high"
    assert eng.band_for(55)["level"] == "high"
    assert eng.band_for(54.9)["level"] == "mid-high"
    assert eng.band_for(30)["level"] == "mid"
    assert eng.band_for(0)["level"] == "very-low"
    assert eng.band_for(-12)["level"] == "very-low"


def test_similarity_transform_recovers_scaling():
    src = eng.ARCFACE_DST * 2 + 10
    m = eng.FaceEngine._similarity(src.astype(np.float64), eng.ARCFACE_DST.astype(np.float64))
    mapped = src @ m[:, :2].T + m[:, 2]
    assert np.allclose(mapped, eng.ARCFACE_DST, atol=1e-3)


def test_decode_rejects_garbage():
    with pytest.raises(eng.ImageError):
        eng.decode_image(b"not an image")


@pytest.mark.skipif(not (SAMPLES / "a1.jpg").exists(), reason="ไม่มีภาพตัวอย่าง")
def test_same_vs_different_person():
    e = eng.get_engine()

    def emb(name):
        img = eng.decode_image((SAMPLES / name).read_bytes())
        faces = e.detect(img)
        assert faces, f"ไม่พบใบหน้าใน {name}"
        return e.embed(img, faces[0])

    same = eng.cosine(emb("a1.jpg"), emb("a2.jpg"))
    diff = eng.cosine(emb("a1.jpg"), emb("b.jpg"))
    assert same > diff + 0.3
