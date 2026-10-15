"""ตรรกะระดับแอป: แคชภาพชั่วคราวในหน่วยความจำ, อัปโหลด/ตรวจจับ, เปรียบเทียบ"""
from __future__ import annotations

import base64
import hashlib
import threading
import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from . import config, engine as eng


class NotFound(KeyError):
    """ภาพหมดอายุหรือไม่มีในแคช"""


@dataclass
class Entry:
    id: str
    filename: str
    sha256: str
    size_bytes: int
    img: np.ndarray
    faces: list[eng.Face]
    created: float


class Store:
    """เก็บภาพในหน่วยความจำชั่วคราวเท่านั้น (ไม่เขียนลงดิสก์) หมดอายุอัตโนมัติ"""

    def __init__(self):
        self._items: dict[str, Entry] = {}
        self._lock = threading.Lock()

    def _purge(self):
        now = time.time()
        for k in [k for k, v in self._items.items() if now - v.created > config.CACHE_TTL_SECONDS]:
            del self._items[k]
        while len(self._items) > config.CACHE_MAX_ITEMS:
            oldest = min(self._items.values(), key=lambda e: e.created)
            del self._items[oldest.id]

    def put(self, entry: Entry):
        with self._lock:
            self._items[entry.id] = entry
            self._purge()

    def get(self, id_: str) -> Entry:
        with self._lock:
            self._purge()
            if id_ not in self._items:
                raise NotFound(id_)
            return self._items[id_]

    def delete(self, id_: str):
        with self._lock:
            self._items.pop(id_, None)


store = Store()


def _data_url(png: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(png).decode()


def register_upload(data: bytes, filename: str) -> Entry:
    img = eng.decode_image(data)
    faces = eng.get_engine().detect(img)
    entry = Entry(
        id=uuid.uuid4().hex,
        filename=filename or "image",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        img=img,
        faces=faces,
        created=time.time(),
    )
    store.put(entry)
    return entry


def describe(entry: Entry) -> dict:
    h, w = entry.img.shape[:2]
    faces = []
    for i, f in enumerate(entry.faces):
        x1, y1, x2, y2 = f.bbox
        faces.append(
            {
                "index": i,
                "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                "landmarks": (f.kps / np.array([w, h])).round(5).tolist(),
                "quality": f.quality,
            }
        )
    return {
        "id": entry.id,
        "filename": entry.filename,
        "sha256": entry.sha256,
        "size_bytes": entry.size_bytes,
        "width": w,
        "height": h,
        "faces": faces,
    }


def preview_jpeg(entry: Entry) -> bytes:
    return eng.encode_jpeg(entry.img, config.PREVIEW_SIDE, 85)


def _face(entry: Entry, index: int) -> eng.Face:
    if not 0 <= index < len(entry.faces):
        raise IndexError("ไม่พบใบหน้าลำดับที่เลือก")
    return entry.faces[index]


@dataclass
class Comparison:
    cosine: float
    score: float
    band: dict
    reliability: dict
    a_entry: Entry
    b_entry: Entry
    a_face: eng.Face
    b_face: eng.Face
    a_aligned: np.ndarray
    b_aligned: np.ndarray
    same_image: bool


def reliability(qa: dict, qb: dict) -> dict:
    order = {"good": 0, "fair": 1, "poor": 2}
    worst = max(qa["grade"], qb["grade"], key=lambda g: order[g])
    text = {
        "good": "คุณภาพภาพโดยรวมเหมาะสมต่อการเปรียบเทียบ",
        "fair": "คุณภาพภาพพอใช้ ผลอาจคลาดเคลื่อนได้ ควรใช้ด้วยความระมัดระวัง",
        "poor": "คุณภาพภาพไม่ดี ผลการเปรียบเทียบเชื่อถือได้ต่ำ ควรหาภาพที่ดีกว่า",
    }[worst]
    notes = [f"ภาพ A: {w}" for w in qa["warnings"]] + [f"ภาพ B: {w}" for w in qb["warnings"]]
    return {"grade": worst, "text": text, "notes": notes}


def compare(a_id: str, a_idx: int, b_id: str, b_idx: int) -> Comparison:
    a, b = store.get(a_id), store.get(b_id)
    fa, fb = _face(a, a_idx), _face(b, b_idx)
    e = eng.get_engine()
    ea, ebd = e.embed(a.img, fa), e.embed(b.img, fb)
    c = eng.cosine(ea, ebd)
    score = round(c * 100, 1)
    return Comparison(
        cosine=c,
        score=score,
        band=eng.band_for(score),
        reliability=reliability(fa.quality, fb.quality),
        a_entry=a,
        b_entry=b,
        a_face=fa,
        b_face=fb,
        a_aligned=e.align(a.img, fa.kps, eng.ALIGN_SIZE),
        b_aligned=e.align(b.img, fb.kps, eng.ALIGN_SIZE),
        same_image=(a.sha256 == b.sha256 and a_idx == b_idx),
    )


def comparison_json(cmp: Comparison) -> dict:
    def side(aligned, face):
        regions = {
            k: {"label": eng.REGIONS[k][0], "image": _data_url(eng.encode_png(eng.crop_region(aligned, k)))}
            for k in eng.REGIONS
        }
        return {
            "aligned": _data_url(eng.encode_png(aligned)),
            "quality": face.quality,
            "regions": regions,
        }

    notes = list(cmp.reliability["notes"])
    if cmp.same_image:
        notes.insert(0, "เลือกภาพและใบหน้าเดียวกันทั้งสองฝั่ง คะแนนจึงสูงสุดโดยธรรมชาติ ไม่มีความหมายเชิงการพิสูจน์")
    return {
        "score": cmp.score,
        "cosine": round(cmp.cosine, 4),
        "band": cmp.band,
        "reliability": {**cmp.reliability, "notes": notes},
        "a": side(cmp.a_aligned, cmp.a_face),
        "b": side(cmp.b_aligned, cmp.b_face),
        "bands": [{"min": b[0], "label": b[1], "level": b[2]} for b in config.BANDS],
        "model": config.MODEL_NAME,
        "disclaimer": config.DISCLAIMER_TH,
    }
