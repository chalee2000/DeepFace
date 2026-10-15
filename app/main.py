"""FastAPI: เสิร์ฟหน้าเว็บ + API ตรวจจับ/เปรียบเทียบ/ออกรายงาน"""
from __future__ import annotations

from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, engine as eng, report, service

app = FastAPI(title="Face Compare", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.on_event("startup")
def _warm():
    eng.get_engine()  # โหลดโมเดลล่วงหน้า (ล้มทันทีถ้าไม่มีไฟล์โมเดล)


@app.get("/api/health")
def health():
    return {"ok": True, "model": config.MODEL_NAME}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "ไฟล์ใหญ่เกิน 25 MB")
    if not data:
        raise HTTPException(400, "ไฟล์ว่างเปล่า")
    try:
        entry = service.register_upload(data, file.filename or "image")
    except eng.ImageError as exc:
        raise HTTPException(400, str(exc))
    return service.describe(entry)


@app.get("/api/image/{image_id}")
def image(image_id: str):
    try:
        entry = service.store.get(image_id)
    except service.NotFound:
        raise HTTPException(410, "ภาพหมดอายุ กรุณาอัปโหลดใหม่")
    return Response(service.preview_jpeg(entry), media_type="image/jpeg")


@app.delete("/api/image/{image_id}")
def delete_image(image_id: str):
    service.store.delete(image_id)
    return {"ok": True}


class Pick(BaseModel):
    id: str
    face: int = Field(0, ge=0)


class CompareReq(BaseModel):
    a: Pick
    b: Pick


class CaseInfo(BaseModel):
    case_no: str = Field("", max_length=200)
    purpose: str = Field("", max_length=500)
    examiner: str = Field("", max_length=200)
    notes: str = Field("", max_length=3000)


class ReportReq(CompareReq):
    case: CaseInfo = CaseInfo()


def _compare(req: CompareReq) -> service.Comparison:
    try:
        return service.compare(req.a.id, req.a.face, req.b.id, req.b.face)
    except service.NotFound:
        raise HTTPException(410, "ภาพหมดอายุ กรุณาอัปโหลดใหม่")
    except IndexError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/compare")
def compare(req: CompareReq):
    return JSONResponse(service.comparison_json(_compare(req)))


@app.post("/api/report")
def make_report(req: ReportReq):
    cmp = _compare(req)
    pdf = report.build_report(cmp, req.case.model_dump())
    name = quote("face-comparison-report.pdf")
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"face-comparison-report.pdf\"; filename*=UTF-8''{name}"},
    )


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/fonts", StaticFiles(directory=config.FONT_DIR), name="fonts")
app.mount("/", StaticFiles(directory=config.STATIC_DIR), name="static")
