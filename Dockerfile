FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY scripts scripts
RUN python scripts/download_models.py
EXPOSE 8000
# Render/Fly/Heroku กำหนดพอร์ตผ่านตัวแปร PORT (ค่าเริ่มต้นในเครื่อง = 8000)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
