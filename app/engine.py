"""ตรวจจับ จัดตำแหน่ง และเปรียบเทียบใบหน้าด้วย ONNX Runtime (ไม่ต้องใช้ TensorFlow/PyTorch)"""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config

try:  # รองรับภาพ HEIC/HEIF จาก iPhone
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - ไม่มีไลบรารีก็ยังใช้ JPEG/PNG ได้
    pass

# จุดอ้างอิง 5 จุด (ตา, จมูก, มุมปาก) ของ ArcFace ที่ขนาด 112x112
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
ALIGN_SIZE = 256  # ขนาดภาพใบหน้าที่จัดตำแหน่งแล้วสำหรับแสดงผล/รายงาน


class ImageError(ValueError):
    """ไฟล์ภาพเปิดไม่ได้หรือไม่ถูกต้อง"""


@dataclass
class Face:
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 (พิกเซลของภาพที่ประมวลผล)
    kps: np.ndarray                          # (5,2)
    score: float
    quality: dict = field(default_factory=dict)


def decode_image(data: bytes) -> np.ndarray:
    """ถอดรหัสภาพ → BGR uint8 พร้อมหมุนตาม EXIF และย่อถ้าใหญ่เกินไป"""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
    except Exception as exc:
        raise ImageError("เปิดไฟล์ภาพไม่ได้ (รองรับ JPEG, PNG, WebP, HEIC)") from exc
    w, h = im.size
    if min(w, h) < 32:
        raise ImageError("ภาพเล็กเกินไป")
    scale = config.MAX_IMAGE_SIDE / max(w, h)
    if scale < 1:
        im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def encode_jpeg(img: np.ndarray, max_side: int | None = None, quality: int = 88) -> bytes:
    if max_side:
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageError("เข้ารหัสภาพไม่สำเร็จ")
    return buf.tobytes()


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ImageError("เข้ารหัสภาพไม่สำเร็จ")
    return buf.tobytes()


# ---------------------------------------------------------------- ตรวจจับ (SCRFD)
def _distance2bbox(points, distance):
    return np.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        axis=-1,
    )


def _distance2kps(points, distance):
    out = []
    for i in range(0, distance.shape[1], 2):
        out.append(points[:, 0] + distance[:, i])
        out.append(points[:, 1] + distance[:, i + 1])
    return np.stack(out, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class FaceEngine:
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, model_dir=config.MODEL_DIR):
        import onnxruntime as ort

        det_path = model_dir / config.DETECTOR_FILE
        rec_path = model_dir / config.RECOGNIZER_FILE
        for p in (det_path, rec_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"ไม่พบโมเดล {p} — รัน `python scripts/download_models.py` ก่อน"
                )
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self._det = ort.InferenceSession(str(det_path), opts, providers=providers)
        self._rec = ort.InferenceSession(str(rec_path), opts, providers=providers)
        self._det_in = self._det.get_inputs()[0].name
        self._rec_in = self._rec.get_inputs()[0].name
        self._lock = threading.Lock()  # session เดียว ป้องกันเรียกซ้อนกัน
        self._anchor_cache: dict = {}

    # -- detection
    def _centers(self, h, w, stride):
        key = (h, w, stride)
        if key not in self._anchor_cache:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            c = np.stack([c] * self.NUM_ANCHORS, axis=1).reshape(-1, 2)
            self._anchor_cache[key] = c
        return self._anchor_cache[key]

    def _detect_once(self, img: np.ndarray, size: int) -> list[Face]:
        h, w = img.shape[:2]
        ratio = h / w
        if ratio > 1:
            new_h, new_w = size, int(size / ratio)
        else:
            new_w, new_h = size, int(size * ratio)
        scale = new_h / h
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5,) * 3, swapRB=True)
        with self._lock:
            outs = self._det.run(None, {self._det_in: blob})

        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(self.STRIDES):
            scores = outs[idx].reshape(-1)
            bbox = outs[idx + 3].reshape(-1, 4) * stride
            kps = outs[idx + 6].reshape(-1, 10) * stride
            centers = self._centers(size // stride, size // stride, stride)
            keep = np.where(scores >= config.DET_THRESHOLD)[0]
            if keep.size == 0:
                continue
            scores_l.append(scores[keep])
            boxes_l.append(_distance2bbox(centers, bbox)[keep])
            kps_l.append(_distance2kps(centers, kps)[keep].reshape(-1, 5, 2))
        if not scores_l:
            return []
        scores = np.concatenate(scores_l)
        boxes = np.concatenate(boxes_l) / scale
        kps = np.concatenate(kps_l) / scale
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, 0.4)
        faces = [Face(tuple(map(float, dets[i, :4])), kps[i].astype(np.float32), float(dets[i, 4])) for i in keep]
        return faces

    def detect(self, img: np.ndarray) -> list[Face]:
        """ตรวจจับใบหน้าทั้งหมด เรียงจากใหญ่ไปเล็ก; ถ้าไม่พบลองความละเอียดสูงขึ้น"""
        faces: list[Face] = []
        for size in (640, 1024):
            faces = self._detect_once(img, size)
            if faces:
                break
        if not faces:
            # ภาพที่ครอปใบหน้าชิดมาก (เช่น รูปติดบัตร): เติมขอบรอบภาพแล้วลองอีกครั้ง
            pad = int(max(img.shape[:2]) * 0.5)
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            faces = self._detect_once(padded, 640)
            for f in faces:
                x1, y1, x2, y2 = f.bbox
                f.bbox = (x1 - pad, y1 - pad, x2 - pad, y2 - pad)
                f.kps = f.kps - pad
        h, w = img.shape[:2]
        faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        for f in faces:
            f.quality = self.assess(img, f, (h, w))
        return faces

    # -- alignment
    @staticmethod
    def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Umeyama similarity transform (2x3)"""
        n = src.shape[0]
        sm, dm = src.mean(0), dst.mean(0)
        sd, dd = src - sm, dst - dm
        cov = dd.T @ sd / n
        d = np.ones(2)
        if np.linalg.det(cov) < 0:
            d[1] = -1
        u, s, vt = np.linalg.svd(cov)
        r = u @ np.diag(d) @ vt
        var = sd.var(0).sum()
        scale = (s * d).sum() / var
        t = dm - scale * r @ sm
        return np.hstack([scale * r, t[:, None]]).astype(np.float32)

    def align(self, img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
        dst = ARCFACE_DST * (size / 112.0)
        m = self._similarity(kps.astype(np.float64), dst.astype(np.float64))
        return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_CUBIC, borderValue=0)

    # -- embedding
    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        crop = self.align(img, face.kps, 112)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        with self._lock:
            emb = self._rec.run(None, {self._rec_in: blob})[0][0]
        return emb / np.linalg.norm(emb)

    # -- quality
    @staticmethod
    def pose(kps: np.ndarray) -> dict:
        """ประมาณมุมหน้าอย่างหยาบจากจุด 5 จุด (ไม่ใช่ค่าที่วัดจริง)"""
        le, re, nose, lm, rm = kps
        eye_vec = re - le
        eye_dist = float(np.linalg.norm(eye_vec)) + 1e-6
        roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
        mid_eye = (le + re) / 2
        # ตำแหน่งจมูกตามแนวตาซ้าย-ขวา เทียบกึ่งกลางตา: 0 = หน้าตรง
        axis = eye_vec / eye_dist
        yaw_ratio = float(np.dot(nose - mid_eye, axis) / eye_dist)
        yaw = float(np.degrees(np.arctan(yaw_ratio / 0.55)))
        mid_mouth = (lm + rm) / 2
        up = np.array([-axis[1], axis[0]])
        eye_nose = float(np.dot(nose - mid_eye, up))
        nose_mouth = float(np.dot(mid_mouth - nose, up))
        pitch_ratio = eye_nose / (eye_nose + nose_mouth + 1e-6)  # หน้าตรง ≈ 0.55–0.65
        return {"yaw": yaw, "roll": roll, "pitch_ratio": pitch_ratio}

    def assess(self, img: np.ndarray, face: Face, shape: tuple[int, int]) -> dict:
        x1, y1, x2, y2 = face.bbox
        fw, fh = x2 - x1, y2 - y1
        crop = self.align(img, face.kps, 112)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(gray.mean())
        contrast = float(gray.std())
        pose = self.pose(face.kps)
        h, w = shape
        touches_edge = x1 < 2 or y1 < 2 or x2 > w - 2 or y2 > h - 2

        warnings: list[str] = []
        penalty = 0
        if min(fw, fh) < config.MIN_FACE_PX:
            warnings.append(f"ใบหน้าเล็กมาก (~{int(min(fw, fh))} px) ความละเอียดไม่พอ"); penalty += 2
        elif min(fw, fh) < 80:
            warnings.append(f"ใบหน้าค่อนข้างเล็ก (~{int(min(fw, fh))} px)"); penalty += 1
        if face.score < 0.65:
            warnings.append("ความมั่นใจในการตรวจจับใบหน้าต่ำ"); penalty += 1
        if abs(pose["yaw"]) > 35:
            warnings.append(f"ใบหน้าหันข้างมาก (ประมาณ {abs(pose['yaw']):.0f}°)"); penalty += 2
        elif abs(pose["yaw"]) > 20:
            warnings.append(f"ใบหน้าเอียงหันเล็กน้อย (ประมาณ {abs(pose['yaw']):.0f}°)"); penalty += 1
        if abs(pose["roll"]) > 25:
            warnings.append(f"ศีรษะเอียงมาก (~{abs(pose['roll']):.0f}°)"); penalty += 1
        if pose["pitch_ratio"] < 0.33 or pose["pitch_ratio"] > 0.80:
            warnings.append("ใบหน้าก้มหรือเงยมาก"); penalty += 1
        if sharp < 40:
            warnings.append("ภาพเบลอ/ไม่คมชัด"); penalty += 2
        elif sharp < 90:
            warnings.append("ภาพค่อนข้างไม่คมชัด"); penalty += 1
        if bright < 55:
            warnings.append("ภาพมืดเกินไป"); penalty += 1
        elif bright > 200:
            warnings.append("ภาพสว่างจ้าเกินไป"); penalty += 1
        if contrast < 25:
            warnings.append("คอนทราสต์ต่ำ"); penalty += 1
        if touches_edge:
            warnings.append("ใบหน้าชิด/ถูกตัดที่ขอบภาพ"); penalty += 1
        grade = "good" if penalty == 0 else "fair" if penalty <= 2 else "poor"
        return {
            "det_score": round(face.score, 3),
            "face_px": [int(fw), int(fh)],
            "sharpness": round(sharp, 1),
            "brightness": round(bright, 1),
            "contrast": round(contrast, 1),
            "yaw": round(pose["yaw"], 1),
            "roll": round(pose["roll"], 1),
            "pitch_ratio": round(pose["pitch_ratio"], 2),
            "warnings": warnings,
            "grade": grade,
        }


def band_for(score100: float) -> dict:
    for lower, label, level, text in config.BANDS:
        if score100 >= lower:
            return {"label": label, "level": level, "text": text}
    return {"label": config.BANDS[-1][1], "level": config.BANDS[-1][2], "text": config.BANDS[-1][3]}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ภูมิภาคของใบหน้า (สัดส่วนบนภาพใบหน้าที่จัดตำแหน่งแล้ว) x1,y1,x2,y2
REGIONS = {
    "eyes": ("ดวงตา/คิ้ว", (0.12, 0.28, 0.88, 0.52)),
    "nose": ("จมูก", (0.28, 0.40, 0.72, 0.72)),
    "mouth": ("ปาก/คาง", (0.20, 0.64, 0.80, 0.96)),
}


def crop_region(aligned: np.ndarray, key: str) -> np.ndarray:
    h, w = aligned.shape[:2]
    x1, y1, x2, y2 = REGIONS[key][1]
    return aligned[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> FaceEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = FaceEngine()
        return _engine
