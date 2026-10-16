# เว็บแอปแบบรันในเบราว์เซอร์ทั้งหมด (standalone/) เสิร์ฟด้วย nginx — ไม่ต้องใช้ RAM ฝั่งเซิร์ฟเวอร์
# (เวอร์ชันเซิร์ฟเวอร์ Python เดิมอยู่ที่ Dockerfile.server)
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir onnx onnxruntime
COPY standalone standalone
# ดาวน์โหลดโมเดล InsightFace + onnxruntime-web แล้วประกอบหน้าเว็บลง standalone/dist
RUN python standalone/build.py

FROM nginx:1.27-alpine
ENV PORT=10000
COPY --from=build /src/standalone/dist /usr/share/nginx/html
COPY standalone/nginx.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 10000
