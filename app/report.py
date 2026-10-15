"""สร้างรายงาน PDF ภาษาไทย (ReportLab + ฟอนต์ Sarabun)"""
from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

from . import config, engine as eng
from .service import Comparison

try:
    from pythainlp.tokenize import word_tokenize
except Exception:  # pragma: no cover - ไม่มี pythainlp จะตัดบรรทัดไทยแบบหยาบ
    word_tokenize = None

pdfmetrics.registerFont(TTFont("Sarabun", str(config.FONT_DIR / "Sarabun-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Sarabun-Bold", str(config.FONT_DIR / "Sarabun-Bold.ttf")))
pdfmetrics.registerFontFamily("Sarabun", normal="Sarabun", bold="Sarabun-Bold", italic="Sarabun", boldItalic="Sarabun-Bold")

TH_TZ = timezone(timedelta(hours=7))
TH_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
             "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
# ใช้โทนสีเดียวไล่ระดับ (ไม่ใช้แดง/เขียน) เพื่อไม่ให้ตีความว่า "ผ่าน/ไม่ผ่าน"
LEVEL_COLORS = {
    "high": colors.HexColor("#1e1b6e"),
    "mid-high": colors.HexColor("#2f3fa8"),
    "mid": colors.HexColor("#3f66c4"),
    "low": colors.HexColor("#6b8cc7"),
    "very-low": colors.HexColor("#64748b"),
}
INK = colors.HexColor("#101828")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#d0d5dd")
ACCENT = colors.HexColor("#1d4ed8")

CONTENT_W = A4[0] - 36 * mm
_THAI = re.compile("[฀-๿]")


def wrap_lines(text: str, font: str, size: float, width: float) -> str:
    """ReportLab ตัดบรรทัดภาษาไทยได้เฉพาะที่ช่องว่าง (ไม่เช่นนั้นจะตัดกลางคำ/กลางสระ)
    จึงตัดบรรทัดเองตามขอบคำ (pythainlp) โดยวัดความกว้างจริง แล้วส่งเป็นบรรทัดสำเร็จรูป"""
    if not text or word_tokenize is None or not _THAI.search(text):
        return text
    out_lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for tok in word_tokenize(para, engine="newmm", keep_whitespace=True):
            if cur and pdfmetrics.stringWidth(cur + tok, font, size) > width:
                out_lines.append(cur.rstrip())
                tok = tok.lstrip()
                cur = ""
            cur += tok
        out_lines.append(cur.rstrip())
    return "\n".join(out_lines)


def P(text: str, style: ParagraphStyle, width: float | None = None) -> Paragraph:
    """width = ความกว้างที่ใช้ได้จริง (pt) ของย่อหน้านี้ ถ้าไม่ระบุใช้ความกว้างเนื้อหาเต็มหน้า"""
    w = (width if width is not None else CONTENT_W) - style.leftIndent - style.rightIndent
    return Paragraph(escape(wrap_lines(text, style.fontName, style.fontSize, w)).replace("\n", "<br/>"), style)


S = {
    "title": ParagraphStyle("title", fontName="Sarabun-Bold", fontSize=22, leading=28, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Sarabun", fontSize=11, leading=15, textColor=MUTED),
    "h": ParagraphStyle("h", fontName="Sarabun-Bold", fontSize=13.5, leading=18, textColor=ACCENT, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Sarabun", fontSize=10.5, leading=15, textColor=INK),
    "bodyb": ParagraphStyle("bodyb", fontName="Sarabun-Bold", fontSize=10.5, leading=15, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="Sarabun", fontSize=10.5, leading=15, textColor=INK, leftIndent=10, firstLineIndent=-10),
    "small": ParagraphStyle("small", fontName="Sarabun", fontSize=8.5, leading=11.5, textColor=MUTED),
    "cap": ParagraphStyle("cap", fontName="Sarabun", fontSize=9, leading=12, textColor=MUTED, alignment=TA_CENTER),
    "cell": ParagraphStyle("cell", fontName="Sarabun", fontSize=9.5, leading=13, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sarabun-Bold", fontSize=9.5, leading=13, textColor=INK),
    "warn": ParagraphStyle("warn", fontName="Sarabun", fontSize=10, leading=14.5, textColor=colors.HexColor("#7a271a")),
}


class Gauge(Flowable):
    """แถบคะแนน 0–100 พร้อมช่วงระดับและตัวชี้"""

    def __init__(self, score: float, width: float, height: float = 16 * mm):
        super().__init__()
        self.score, self.width, self.height = score, width, height

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        bar_y, bar_h = 6 * mm, 4.5 * mm
        edges = [0.0, 20.0, 30.0, 40.0, 55.0, 100.0]
        levels = ["very-low", "low", "mid", "mid-high", "high"]
        for lo, hi, lv in zip(edges, edges[1:], levels):
            c.setFillColor(LEVEL_COLORS[lv])
            c.rect(self.width * lo / 100, bar_y, self.width * (hi - lo) / 100, bar_h, stroke=0, fill=1)
        c.setFillColor(MUTED)
        c.setFont("Sarabun", 7.5)
        for v in edges:
            x = self.width * v / 100
            c.setStrokeColor(colors.white)
            c.line(x, bar_y, x, bar_y + bar_h)
            c.drawCentredString(x, bar_y - 3.2 * mm, f"{v:g}")
        x = self.width * max(0.0, min(100.0, self.score)) / 100
        c.setFillColor(INK)
        p = c.beginPath()
        p.moveTo(x, bar_y + bar_h + 0.4 * mm)
        p.lineTo(x - 2 * mm, bar_y + bar_h + 3.6 * mm)
        p.lineTo(x + 2 * mm, bar_y + bar_h + 3.6 * mm)
        p.close()
        c.drawPath(p, fill=1, stroke=0)


def _fit_image(img_bgr: np.ndarray, max_w: float, max_h: float, jpeg=False) -> RLImage:
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h)
    data = eng.encode_jpeg(img_bgr, quality=88) if jpeg else eng.encode_png(img_bgr)
    return RLImage(io.BytesIO(data), width=w * scale, height=h * scale)


def _annotate(entry, chosen: int) -> np.ndarray:
    img = entry.img.copy()
    thick = max(2, round(max(img.shape[:2]) / 400))
    for i, f in enumerate(entry.faces):
        x1, y1, x2, y2 = map(int, f.bbox)
        col = (60, 160, 30) if i == chosen else (170, 170, 170)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        if i == chosen:
            for px, py in f.kps:
                cv2.circle(img, (int(px), int(py)), thick + 1, (0, 140, 255), -1)
    h, w = img.shape[:2]
    s = 1100 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def thai_datetime(dt: datetime) -> str:
    return f"{dt.day} {TH_MONTHS[dt.month - 1]} {dt.year + 543} เวลา {dt:%H:%M} น."


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sarabun", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "คะแนนความคล้ายไม่ใช่การยืนยันตัวบุคคล — ต้องมีการตรวจสอบโดยผู้เชี่ยวชาญและใช้หลักฐานอื่นประกอบ")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"หน้า {doc.page}")
    canvas.restoreState()


def _kv_table(rows, col_widths):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _signed(v: float) -> str:
    r = round(v)
    return f"{r:+d}°" if r else "0°"


GRADE_TH = {"good": "ดี", "fair": "พอใช้", "poor": "ต่ำ"}
PAD = 12  # ระยะขอบซ้าย+ขวาในกล่อง/ตาราง (pt)


def build_report(cmp: Comparison, case: dict) -> bytes:
    now = datetime.now(TH_TZ)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
        title="รายงานการเปรียบเทียบใบหน้า", author=case.get("examiner") or "ระบบเปรียบเทียบใบหน้า",
    )
    W = CONTENT_W
    st: list = []

    st += [P("รายงานการเปรียบเทียบใบหน้า", S["title"]),
           P("Face Comparison Report — รายงานผลวิเคราะห์เชิงเทคนิคเบื้องต้น", S["sub"]), Spacer(1, 4 * mm)]

    def cv(key, default="-"):
        return (case.get(key) or "").strip() or default

    kcol = 45 * mm
    vcol = W - kcol - PAD
    st.append(_kv_table([
        [P("เลขที่เรื่อง / อ้างอิง", S["cellb"], kcol), P(cv("case_no"), S["cell"], vcol)],
        [P("วัตถุประสงค์", S["cellb"], kcol), P(cv("purpose"), S["cell"], vcol)],
        [P("ผู้จัดทำ / ผู้ตรวจ", S["cellb"], kcol), P(cv("examiner"), S["cell"], vcol)],
        [P("วันที่และเวลาที่ออกรายงาน", S["cellb"], kcol), P(thai_datetime(now), S["cell"], vcol)],
    ], [kcol, W - kcol]))

    # --- 1. ผลการเปรียบเทียบ
    st.append(P("1. ผลการเปรียบเทียบ", S["h"]))
    lvl_col = LEVEL_COLORS[cmp.band["level"]]
    score_style = ParagraphStyle("score", fontName="Sarabun-Bold", fontSize=40, leading=44, textColor=lvl_col)
    lab_style = ParagraphStyle("lab", fontName="Sarabun-Bold", fontSize=16, leading=20, textColor=lvl_col)
    lw, rw = W * 0.34, W * 0.66
    score_tbl = Table([[
        [Paragraph(f"{cmp.score:.1f}", score_style), P("คะแนนความคล้าย (0–100)", S["small"], lw - 16)],
        [P(f"ระดับ: {cmp.band['label']}", lab_style, rw - 16), P(cmp.band["text"], S["body"], rw - 16),
         P(f"ค่า cosine similarity = {cmp.cosine:.4f}", S["small"], rw - 16)],
    ]], colWidths=[lw, rw])
    score_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                                   ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    st += [score_tbl, Spacer(1, 3 * mm), Gauge(cmp.score, W),
           P("ช่วงสี: ต่ำมาก <20 · ต่ำ 20–30 · ก้ำกึ่ง 30–40 · สูง 40–55 · สูงมาก ≥55 "
             "(เกณฑ์เชิงแนวทางทั่วไป ไม่ได้สอบเทียบกับข้อมูลของงานนี้ สีใช้แสดงระดับเท่านั้น ไม่ได้บอกว่าผ่านหรือไม่ผ่าน)", S["small"]),
           Spacer(1, 2 * mm)]

    rel = cmp.reliability
    st.append(P(f"คุณภาพภาพโดยรวม: {GRADE_TH[rel['grade']]}", S["bodyb"]))
    st.append(P(rel["text"], S["body"]))
    notes = list(rel["notes"])
    if cmp.same_image:
        notes.insert(0, "เลือกภาพและใบหน้าเดียวกันทั้งสองฝั่ง คะแนนไม่มีความหมายเชิงการพิสูจน์")
    st += [P("• " + n, S["bullet"]) for n in notes]

    warn = Table([[P(config.DISCLAIMER_TH, S["warn"], W - 2 * PAD)]], colWidths=[W])
    warn.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3f2")),
                              ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#f04438")),
                              ("LEFTPADDING", (0, 0), (-1, -1), PAD / 2), ("RIGHTPADDING", (0, 0), (-1, -1), PAD / 2),
                              ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    st += [Spacer(1, 3 * mm), P("ข้อควรระวังสำคัญ", S["h"]), warn]

    # --- 2. ภาพต้นฉบับ
    colw = W / 2
    inner = colw - 6 * mm
    a, b = cmp.a_entry, cmp.b_entry
    ai = a.faces.index(cmp.a_face)
    bi = b.faces.index(cmp.b_face)

    def src_block(label, entry, idx):
        return [
            _fit_image(_annotate(entry, idx), inner, 80 * mm, jpeg=True),
            Spacer(1, 1.5 * mm),
            P(f"{label} · {entry.filename}", S["cell"], inner),
            P(f"ใบหน้าที่เลือก: ลำดับ {idx + 1} จากที่ตรวจพบ {len(entry.faces)} ใบหน้า", S["small"], inner),
            P(f"ขนาดไฟล์ {entry.size_bytes / 1024:.0f} KB · {entry.img.shape[1]}×{entry.img.shape[0]} px", S["small"], inner),
            P(f"SHA-256: {entry.sha256}", S["small"], inner),
        ]

    t = Table([[src_block("ภาพ A", a, ai), src_block("ภาพ B", b, bi)]], colWidths=[colw, colw])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    st.append(KeepTogether([P("2. ภาพที่นำมาเปรียบเทียบ", S["h"]), t,
                            P("กรอบสีเขียว = ใบหน้าที่ใช้เปรียบเทียบ · กรอบสีเทา = ใบหน้าอื่นในภาพ · "
                              "จุดสีส้ม = จุดอ้างอิง 5 จุด (ตา จมูก มุมปาก)", S["small"])]))

    # --- 3. ใบหน้าที่จัดตำแหน่งแล้ว
    fw = colw - 14 * mm

    def aligned_cell(label, img):
        return [_fit_image(img, fw, fw), P(label, S["cap"], fw)]

    face_tbl = Table([[aligned_cell("ภาพ A", cmp.a_aligned), aligned_cell("ภาพ B", cmp.b_aligned)]], colWidths=[colw, colw])
    face_tbl.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    rows = []
    for key in eng.REGIONS:
        ra, rb = eng.crop_region(cmp.a_aligned, key), eng.crop_region(cmp.b_aligned, key)
        label = eng.REGIONS[key][0]
        rows.append([[_fit_image(ra, fw * 0.8, 30 * mm), P(f"A · {label}", S["cap"], fw)],
                     [_fit_image(rb, fw * 0.8, 30 * mm), P(f"B · {label}", S["cap"], fw)]])
    rt = Table(rows, colWidths=[colw, colw])
    rt.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    st += [KeepTogether([P("3. เปรียบเทียบใบหน้าเคียงข้างกัน (จัดตำแหน่งอัตโนมัติ)", S["h"]), face_tbl]),
           KeepTogether([P("ส่วนประกอบของใบหน้า (ใช้ประกอบการพิจารณาด้วยสายตา โมเดลไม่ได้ให้คะแนนแยกรายส่วน)", S["small"]), rt])]

    # --- 4. คุณภาพภาพ
    c0, c1 = W * 0.46, W * 0.27

    def qrow(name, fn):
        return [P(name, S["cell"], c0 - 8), P(fn(cmp.a_face.quality), S["cell"], c1 - 8), P(fn(cmp.b_face.quality), S["cell"], c1 - 8)]

    qt = Table(
        [[P("รายการ", S["cellb"], c0 - 8), P("ภาพ A", S["cellb"], c1 - 8), P("ภาพ B", S["cellb"], c1 - 8)],
         qrow("ความมั่นใจการตรวจจับ", lambda q: f"{q['det_score']:.2f}"),
         qrow("ขนาดใบหน้า (px)", lambda q: f"{q['face_px'][0]}×{q['face_px'][1]}"),
         qrow("ความคมชัด (Laplacian var.)", lambda q: f"{q['sharpness']:.0f}"),
         qrow("ความสว่างเฉลี่ย (0–255)", lambda q: f"{q['brightness']:.0f}"),
         qrow("คอนทราสต์", lambda q: f"{q['contrast']:.0f}"),
         qrow("มุมหันหน้า (ประมาณ)", lambda q: _signed(q["yaw"])),
         qrow("มุมเอียงศีรษะ (ประมาณ)", lambda q: _signed(q["roll"])),
         qrow("เกรดคุณภาพ", lambda q: GRADE_TH[q["grade"]])],
        colWidths=[c0, c1, c1],
    )
    qt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                            ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    st.append(KeepTogether([P("4. คุณภาพภาพและท่าทางใบหน้า", S["h"]), qt]))

    # --- 5. วิธีการ
    method = [
        f"โมเดล: {config.MODEL_NAME}",
        "ขั้นตอน: (1) ตรวจจับใบหน้าและจุดอ้างอิง 5 จุด (2) จัดตำแหน่งใบหน้าให้ตรงแม่แบบ 112×112 พิกเซล "
        "(3) สกัดเวกเตอร์ลักษณะเฉพาะ 512 มิติ (4) คำนวณ cosine similarity ระหว่างเวกเตอร์ของสองภาพ",
        "คะแนนความคล้าย = cosine similarity × 100 (ค่ายิ่งสูงยิ่งคล้าย; คนละคนมักได้ใกล้ 0 หรือติดลบเล็กน้อย) "
        "ไม่ใช่เปอร์เซ็นต์ความน่าจะเป็น",
        "ภาพถูกหมุนตามข้อมูล EXIF และอาจถูกย่อเพื่อประมวลผล; ค่า SHA-256 คำนวณจากไฟล์ที่อัปโหลดต้นฉบับ",
        "ระบบไม่บันทึกภาพลงดิสก์ ภาพอยู่ในหน่วยความจำชั่วคราวและหมดอายุอัตโนมัติ",
    ]
    st.append(KeepTogether([P("5. วิธีการ", S["h"])] + [P("• " + m, S["bullet"]) for m in method]))

    if (case.get("notes") or "").strip():
        st.append(KeepTogether([P("6. หมายเหตุของผู้จัดทำ", S["h"]), P(case["notes"].strip(), S["body"])]))

    sig = Table([[P("ลงชื่อ ........................................ ผู้ตรวจสอบ", S["cell"], W / 2),
                  P("ลงชื่อ ........................................ ผู้รับรอง", S["cell"], W / 2)],
                 [P("(........................................)", S["cap"], W / 2), P("(........................................)", S["cap"], W / 2)]],
                colWidths=[W / 2] * 2)
    st += [Spacer(1, 10 * mm), KeepTogether([sig])]

    doc.build(st, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
