"use strict";
/* เปรียบเทียบใบหน้า — ทำงานในเบราว์เซอร์ทั้งหมด (ONNX Runtime Web)
   ตรวจจับ SCRFD → จุดโครงหน้า 106 จุด → จัดตำแหน่ง → ArcFace ResNet50 (int8) + ภาพกลับด้าน → cosine similarity
   ภาพไม่ถูกส่งออกจากเครื่อง ไม่มีเซิร์ฟเวอร์ */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const SLOTS = ["A", "B"];
  const MANIFEST = /*__MANIFEST__*/null; // build.py แทนที่ด้วยรายการไฟล์โมเดล {det, lmk, rec: [{path, size}]}

  const CFG = {
    modelName: "SCRFD-500M (ตรวจจับ) + 2D-106 (จุดโครงหน้า) + ArcFace ResNet50 WebFace600K int8 (จดจำ) — InsightFace",
    maxSide: 2400,
    previewSide: 1600,
    detThreshold: 0.5,
    minFacePx: 40,
    alignSize: 256,
  };
  // ระดับการแปลผลจากค่า cosine similarity × 100 (เกณฑ์เชิงแนวทาง ไม่ได้สอบเทียบกับข้อมูลของคุณ)
  const BANDS = [
    { min: 55, label: "สูงมาก", level: "high", text: "ลักษณะใบหน้าคล้ายกันมากตามโมเดล" },
    { min: 40, label: "สูง", level: "mid-high", text: "ลักษณะใบหน้าคล้ายกันพอสมควร" },
    { min: 30, label: "ก้ำกึ่ง", level: "mid", text: "ไม่สามารถสรุปได้ ควรใช้ภาพที่ดีกว่าและหลักฐานอื่นประกอบ" },
    { min: 20, label: "ต่ำ", level: "low", text: "ลักษณะใบหน้าไม่ค่อยคล้ายกันตามโมเดล" },
    { min: -1e9, label: "ต่ำมาก", level: "very-low", text: "ลักษณะใบหน้าแตกต่างกันมากตามโมเดล" },
  ];
  const DISCLAIMER =
    "คะแนนนี้คือค่าความคล้ายของลักษณะใบหน้าที่โมเดลคำนวณได้ ไม่ใช่เปอร์เซ็นต์ความน่าจะเป็นว่าเป็นบุคคลเดียวกัน " +
    "ผลได้รับผลกระทบจากแสง มุมหน้า ความคมชัด การแสดงสีหน้า อายุที่ต่างกัน แว่น/หน้ากาก และคุณภาพภาพ " +
    "โมเดลอาจให้ผลผิดพลาดทั้งแบบบอกว่าคล้ายทั้งที่เป็นคนละคน และบอกว่าไม่คล้ายทั้งที่เป็นคนเดียวกัน " +
    "ห้ามใช้ผลนี้เพียงลำพังเพื่อยืนยันตัวบุคคล โดยเฉพาะงานสืบสวนสอบสวน " +
    "ต้องให้ผู้เชี่ยวชาญด้านการพิสูจน์บุคคลตรวจสอบ และใช้พยานหลักฐานอื่นประกอบเสมอ";
  const GRADE_TH = { good: "ดี", fair: "พอใช้", poor: "ต่ำ" };
  const REGIONS = {
    eyes: ["ดวงตา/คิ้ว", [0.12, 0.28, 0.88, 0.52]],
    nose: ["จมูก", [0.28, 0.40, 0.72, 0.72]],
    mouth: ["ปาก/คาง", [0.20, 0.64, 0.80, 0.96]],
  };
  const ARCFACE_DST = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
  // เส้นเชื่อมจุดโครงหน้า 106 จุด (ลำดับตามโมเดล InsightFace 2d106det) [จุด, ปิดรูป?]
  const MESH = [
    [[1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17], false],
    [[43, 48, 49, 51, 50, 46, 47, 45, 44], true],
    [[97, 98, 99, 100, 101, 105, 104, 103, 102], true],
    [[35, 41, 40, 42, 39, 37, 33, 36], true],
    [[89, 95, 94, 96, 93, 91, 87, 90], true],
    [[72, 73, 74, 86, 80], false],
    [[76, 77, 78, 79, 80, 85, 84, 83, 82], false],
    [[52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55], true],
    [[65, 66, 62, 70, 69, 57, 60, 54], true],
  ];
  const EYE_L = [35, 41, 40, 42, 39, 37, 33, 36], EYE_R = [89, 95, 94, 96, 93, 91, 87, 90];
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
  // สัดส่วนใบหน้า หารด้วยระยะระหว่างรูม่านตา (จุด 38–88) เพื่อไม่ขึ้นกับขนาดภาพ
  const MEASURES = [
    ["ระยะหัวตาด้านใน", (P) => dist(P[39], P[89])],
    ["ระยะหางตาซ้าย–ขวา", (P) => dist(P[35], P[93])],
    ["ความกว้างตา (เฉลี่ย)", (P) => (dist(P[35], P[39]) + dist(P[89], P[93])) / 2],
    ["ระยะคิ้ว–ตา (เฉลี่ย)", (P) => (dist(mid(P[49], P[51]), P[38]) + dist(mid(P[103], P[104]), P[88])) / 2],
    ["ความยาวจมูก (ดั้ง–ฐานจมูก)", (P) => dist(P[72], P[80])],
    ["ความกว้างปีกจมูก", (P) => dist(P[77], P[83])],
    ["ระยะฐานจมูก–ริมฝีปากบน", (P) => dist(P[80], P[71]), true],
    ["ความกว้างปาก", (P) => dist(P[52], P[61]), true],
    ["ระยะฐานจมูก–ปลายคาง", (P) => dist(P[80], P[0]), true],
    ["ความกว้างใบหน้า (ระดับโหนกแก้ม)", (P) => dist(P[12], P[28])],
    ["ความกว้างกราม", (P) => dist(P[4], P[20])],
  ];

  const state = { A: null, B: null, busy: false, comparing: false, last: null };

  // ============================================================ ภาพ
  const makeCanvas = (w, h) => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
  const ctx2d = (c) => c.getContext("2d", { willReadFrequently: true });
  const toBlob = (c, type = "image/jpeg", q = 0.88) => new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("แปลงภาพไม่สำเร็จ"))), type, q));

  async function sha256(file) {
    if (!(window.crypto && crypto.subtle)) return "(เบราว์เซอร์นี้ไม่รองรับการคำนวณ SHA-256)";
    const buf = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  async function decodeFile(file) {
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      try { await img.decode(); } catch {
        throw new Error("เปิดไฟล์ภาพไม่ได้ (รองรับ JPEG, PNG, WebP; ไฟล์ HEIC เปิดได้เฉพาะใน Safari — ถ้าไม่ได้ให้บันทึกเป็น JPEG ก่อน)");
      }
      let w = img.naturalWidth, h = img.naturalHeight;
      if (Math.min(w, h) < 32) throw new Error("ภาพเล็กเกินไป");
      const s = Math.min(1, CFG.maxSide / Math.max(w, h));
      w = Math.round(w * s); h = Math.round(h * s);
      const canvas = makeCanvas(w, h);
      const g = ctx2d(canvas);
      g.imageSmoothingQuality = "high";
      g.drawImage(img, 0, 0, w, h); // เบราว์เซอร์หมุนตาม EXIF ให้แล้ว
      return canvas;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // ============================================================ โหลดไฟล์โมเดล
  // ที่อยู่ไฟล์ที่เผยแพร่คู่กับหน้า อาจต้องอ้างอิงต่างกันตามที่หน้าถูกเสิร์ฟ จึงลองหลายรูปแบบและจำแบบที่ใช้ได้
  const B64 = new Uint8Array(256);
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".split("").forEach((c, i) => (B64[c.charCodeAt(0)] = i));
  function b64decode(src) {
    let n = src.length;
    while (n > 0 && src[n - 1] <= 32) n--;
    let pad = 0;
    if (src[n - 1] === 61) pad++;
    if (src[n - 2] === 61) pad++;
    const out = new Uint8Array((n / 4) * 3 - pad);
    let o = 0;
    for (let i = 0; i < n; i += 4) {
      const v = (B64[src[i]] << 18) | (B64[src[i + 1]] << 12) | (B64[src[i + 2]] << 6) | B64[src[i + 3]];
      if (o < out.length) out[o++] = (v >> 16) & 255;
      if (o < out.length) out[o++] = (v >> 8) & 255;
      if (o < out.length) out[o++] = v & 255;
    }
    return out;
  }

  const Assets = {
    mode: null,
    log: [],
    candidates(path) {
      const loc = window.location;
      const clean = loc.origin + loc.pathname;
      const last = clean.split("/").pop();
      const dir = clean.endsWith("/") ? clean : /\.[a-z0-9]+$/i.test(last) ? clean.slice(0, clean.lastIndexOf("/") + 1) : clean + "/";
      return [path, dir + path, dir + path + loc.search, loc.origin + "/" + path];
    },
    urlFor(path) { return this.candidates(path)[this.mode == null ? 0 : this.mode]; },
    async fetchBytes(path, onChunk) {
      const order = this.mode != null ? [this.mode] : [0, 1, 2, 3];
      for (const m of order) {
        const url = this.candidates(path)[m];
        let r;
        try { r = await fetch(url); } catch (e) { this.log.push(`ERR ${url} (${e && e.message})`); continue; }
        this.log.push(`${r.status} ${url}`);
        if (!r.ok) continue;
        this.mode = m;
        if (r.body && r.body.getReader) {
          const reader = r.body.getReader();
          const chunks = [];
          let total = 0;
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value); total += value.length;
            if (onChunk) onChunk(value.length);
          }
          const out = new Uint8Array(total);
          let off = 0;
          for (const c of chunks) { out.set(c, off); off += c.length; }
          return out;
        }
        const buf = new Uint8Array(await r.arrayBuffer());
        if (onChunk) onChunk(buf.length);
        return buf;
      }
      this.log.push(`page: ${window.location.href.slice(0, 160)} | base: ${document.baseURI.slice(0, 160)}`);
      const err = new Error(`โหลดโมเดลไม่สำเร็จ (${path})`);
      err.detail = this.log.join("\n");
      throw err;
    },
    // โหลดทุกส่วนของโมเดลหนึ่งตัว ถอด base64 ทีละส่วนแล้วต่อกัน
    async model(parts, onChunk) {
      const bins = await Promise.all(parts.map(async (p) => b64decode(await this.fetchBytes(p.path, onChunk))));
      const out = new Uint8Array(bins.reduce((s, b) => s + b.length, 0));
      let off = 0;
      for (const b of bins) { out.set(b, off); off += b.length; }
      return out;
    },
  };

  // ============================================================ โมเดล
  const Engine = {
    det: null, lmk: null, rec: null, small: null, big: null, chain: Promise.resolve(),
    progress: { loaded: 0, total: 0 },
    opts: { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 },
    async session(bytes) {
      try {
        return await ort.InferenceSession.create(bytes, this.opts);
      } catch (e) {
        const err = new Error("เริ่มตัวรันโมเดลไม่สำเร็จ (WebAssembly)");
        err.detail = `${e && e.message}\n${Assets.log.join("\n")}`;
        throw err;
      }
    },
    // โมเดลเล็ก (ตรวจจับ + จุดโครงหน้า) ใช้ตอนเลือกภาพ
    loadSmall() {
      if (!this.small) {
        this.small = (async () => {
          if (typeof ort === "undefined") throw new Error("โหลดไลบรารี ONNX Runtime ไม่สำเร็จ");
          if (!MANIFEST) throw new Error("ไม่พบรายการไฟล์โมเดล");
          ort.env.logLevel = "error";
          ort.env.wasm.numThreads = 1;
          ort.env.wasm.proxy = false;
          const det = await Assets.model(MANIFEST.det); // ไฟล์แรกใช้หาวิธีอ้างอิงที่อยู่ที่ใช้ได้
          ort.env.wasm.wasmPaths = { "ort-wasm-simd.wasm": Assets.urlFor("ort/ort-wasm-simd.wasm"), "ort-wasm.wasm": Assets.urlFor("ort/ort-wasm.wasm") };
          const lmk = await Assets.model(MANIFEST.lmk);
          this.det = await this.session(det);
          this.lmk = await this.session(lmk);
        })().catch((e) => { this.small = null; throw e; });
      }
      return this.small;
    },
    // โมเดลจดจำใบหน้า (~44 MB) โหลดเบื้องหลังหลังเลือกภาพแรก
    loadBig() {
      if (!this.big) {
        this.big = (async () => {
          await this.loadSmall();
          this.progress = { loaded: 0, total: MANIFEST.rec.reduce((s, p) => s + p.size, 0) };
          modelStatus("loading");
          const rec = await Assets.model(MANIFEST.rec, (n) => { this.progress.loaded += n; modelStatus("loading"); });
          modelStatus("init");
          this.rec = await this.session(rec);
          modelStatus("ready");
        })().catch((e) => { this.big = null; modelStatus("error"); throw e; });
      }
      return this.big;
    },
    run(session, feeds) {
      const p = this.chain.then(() => session.run(feeds));
      this.chain = p.catch(() => {});
      return p;
    },
  };

  function modelStatus(phase) {
    const pill = $("#model-pill"), text = $("#model-text"), bar = $("#model-bar"), fill = $("#model-bar-fill");
    pill.dataset.state = phase;
    if (phase === "loading") {
      const { loaded, total } = Engine.progress;
      const pct = total ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
      text.textContent = `กำลังโหลดโมเดลจดจำใบหน้า ${pct}% (${(loaded / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB)`;
      bar.hidden = false;
      fill.style.width = pct + "%";
    } else if (phase === "init") {
      text.textContent = "กำลังเตรียมโมเดลจดจำใบหน้า…";
      fill.style.width = "100%";
    } else if (phase === "ready") {
      text.textContent = "โมเดลพร้อมใช้งาน";
      bar.hidden = true;
    } else if (phase === "error") {
      text.textContent = "โหลดโมเดลไม่สำเร็จ ลองใหม่อีกครั้ง";
      bar.hidden = true;
    } else {
      text.textContent = "โมเดลโหลดเมื่อเลือกภาพแรก (~60 MB)";
    }
  }

  // ============================================================ ตรวจจับใบหน้า (SCRFD)
  const STRIDES = [8, 16, 32];
  const NUM_ANCHORS = 2;

  function nms(dets, thresh) {
    const order = dets.map((_, i) => i).sort((a, b) => dets[b].score - dets[a].score);
    const keep = [];
    while (order.length) {
      const i = order.shift();
      keep.push(i);
      const a = dets[i];
      const areaA = (a.box[2] - a.box[0] + 1) * (a.box[3] - a.box[1] + 1);
      for (let k = order.length - 1; k >= 0; k--) {
        const b = dets[order[k]];
        const xx1 = Math.max(a.box[0], b.box[0]), yy1 = Math.max(a.box[1], b.box[1]);
        const xx2 = Math.min(a.box[2], b.box[2]), yy2 = Math.min(a.box[3], b.box[3]);
        const inter = Math.max(0, xx2 - xx1 + 1) * Math.max(0, yy2 - yy1 + 1);
        const areaB = (b.box[2] - b.box[0] + 1) * (b.box[3] - b.box[1] + 1);
        if (inter / (areaA + areaB - inter) > thresh) order.splice(k, 1);
      }
    }
    return keep.map((i) => dets[i]);
  }

  function chwTensor(px, n, mean, std) {
    const t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      t[i] = (px[i * 4] - mean) / std;
      t[n + i] = (px[i * 4 + 1] - mean) / std;
      t[2 * n + i] = (px[i * 4 + 2] - mean) / std;
    }
    return t;
  }

  async function detectOnce(canvas, size) {
    const w = canvas.width, h = canvas.height;
    const ratio = h / w;
    let nw, nh;
    if (ratio > 1) { nh = size; nw = Math.floor(size / ratio); } else { nw = size; nh = Math.floor(size * ratio); }
    const scale = nh / h;
    const tmp = makeCanvas(size, size);
    const g = ctx2d(tmp);
    g.fillStyle = "#000"; g.fillRect(0, 0, size, size);
    g.imageSmoothingQuality = "high";
    g.drawImage(canvas, 0, 0, nw, nh);
    const input = chwTensor(g.getImageData(0, 0, size, size).data, size * size, 127.5, 128);
    const outs = await Engine.run(Engine.det, { [Engine.det.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, size, size]) });
    const names = Engine.det.outputNames;
    const dets = [];
    STRIDES.forEach((stride, idx) => {
      const scores = outs[names[idx]].data, bbox = outs[names[idx + 3]].data, kps = outs[names[idx + 6]].data;
      const fw = size / stride;
      const total = fw * fw * NUM_ANCHORS;
      for (let i = 0; i < total; i++) {
        if (scores[i] < CFG.detThreshold) continue;
        const cell = Math.floor(i / NUM_ANCHORS);
        const cx = (cell % fw) * stride, cy = Math.floor(cell / fw) * stride;
        const b = [cx - bbox[i * 4] * stride, cy - bbox[i * 4 + 1] * stride, cx + bbox[i * 4 + 2] * stride, cy + bbox[i * 4 + 3] * stride];
        const k = [];
        for (let j = 0; j < 5; j++) k.push([(cx + kps[i * 10 + j * 2] * stride) / scale, (cy + kps[i * 10 + j * 2 + 1] * stride) / scale]);
        dets.push({ box: b.map((v) => v / scale), kps: k, score: scores[i] });
      }
    });
    return nms(dets, 0.4);
  }

  function padReplicate(canvas, pad) {
    const w = canvas.width, h = canvas.height;
    const c = makeCanvas(w + 2 * pad, h + 2 * pad);
    const g = ctx2d(c);
    g.drawImage(canvas, pad, pad);
    g.drawImage(canvas, 0, 0, 1, h, 0, pad, pad, h);
    g.drawImage(canvas, w - 1, 0, 1, h, w + pad, pad, pad, h);
    g.drawImage(canvas, 0, 0, w, 1, pad, 0, w, pad);
    g.drawImage(canvas, 0, h - 1, w, 1, pad, h + pad, w, pad);
    g.drawImage(canvas, 0, 0, 1, 1, 0, 0, pad, pad);
    g.drawImage(canvas, w - 1, 0, 1, 1, w + pad, 0, pad, pad);
    g.drawImage(canvas, 0, h - 1, 1, 1, 0, h + pad, pad, pad);
    g.drawImage(canvas, w - 1, h - 1, 1, 1, w + pad, h + pad, pad, pad);
    return c;
  }

  // จุดโครงหน้า 106 จุด: ครอปรอบกรอบใบหน้า 1.5 เท่า ขนาด 192×192 (ค่าพิกเซลดิบ 0–255)
  async function landmarks106(canvas, bbox) {
    const [x1, y1, x2, y2] = bbox;
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const s = 192 / (Math.max(x2 - x1, y2 - y1) * 1.5);
    const c = makeCanvas(192, 192);
    const g = ctx2d(c);
    g.fillStyle = "#000"; g.fillRect(0, 0, 192, 192);
    g.imageSmoothingQuality = "high";
    g.setTransform(s, 0, 0, s, 96 - s * cx, 96 - s * cy);
    g.drawImage(canvas, 0, 0);
    g.setTransform(1, 0, 0, 1, 0, 0);
    const input = chwTensor(g.getImageData(0, 0, 192, 192).data, 192 * 192, 0, 1);
    const outs = await Engine.run(Engine.lmk, { [Engine.lmk.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, 192, 192]) });
    const p = outs[Engine.lmk.outputNames[0]].data;
    const pts = [];
    for (let i = 0; i < 106; i++) pts.push([((p[2 * i] + 1) * 96 - (96 - s * cx)) / s, ((p[2 * i + 1] + 1) * 96 - (96 - s * cy)) / s]);
    return pts;
  }

  async function detect(canvas) {
    let dets = [];
    for (const size of [640, 1024]) {
      dets = await detectOnce(canvas, size);
      if (dets.length) break;
    }
    if (!dets.length) {
      // ภาพครอปใบหน้าชิดมาก (เช่น รูปติดบัตร): เติมขอบแล้วลองอีกครั้ง
      const pad = Math.floor(Math.max(canvas.width, canvas.height) * 0.5);
      dets = await detectOnce(padReplicate(canvas, pad), 640);
      dets.forEach((d) => {
        d.box = [d.box[0] - pad, d.box[1] - pad, d.box[2] - pad, d.box[3] - pad];
        d.kps = d.kps.map(([x, y]) => [x - pad, y - pad]);
      });
    }
    dets.sort((a, b) => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]));
    const faces = [];
    for (const d of dets) {
      const face = { bbox: d.box, kps: d.kps, score: d.score };
      face.lmk = await landmarks106(canvas, d.box);
      face.quality = assess(canvas, face);
      faces.push(face);
    }
    return faces;
  }

  // ============================================================ จัดตำแหน่ง / ฝังเวกเตอร์
  function similarity(src, dst) {
    const n = src.length;
    const mean = (p) => [p.reduce((s, q) => s + q[0], 0) / n, p.reduce((s, q) => s + q[1], 0) / n];
    const sm = mean(src), dm = mean(dst);
    let num1 = 0, num2 = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const sx = src[i][0] - sm[0], sy = src[i][1] - sm[1], dx = dst[i][0] - dm[0], dy = dst[i][1] - dm[1];
      num1 += sx * dx + sy * dy; num2 += sx * dy - sy * dx; den += sx * sx + sy * sy;
    }
    const A = num1 / den, B = num2 / den;
    return { A, B, tx: dm[0] - (A * sm[0] - B * sm[1]), ty: dm[1] - (B * sm[0] + A * sm[1]) };
  }
  const alignMatrix = (kps, size) => similarity(kps, ARCFACE_DST.map(([x, y]) => [x * (size / 112), y * (size / 112)]));
  const applyM = (m, pts) => pts.map(([x, y]) => [m.A * x - m.B * y + m.tx, m.B * x + m.A * y + m.ty]);

  function align(canvas, kps, size) {
    const m = alignMatrix(kps, size);
    const out = makeCanvas(size, size);
    const g = ctx2d(out);
    g.fillStyle = "#000"; g.fillRect(0, 0, size, size);
    g.imageSmoothingQuality = "high";
    g.setTransform(m.A, m.B, -m.B, m.A, m.tx, m.ty);
    g.drawImage(canvas, 0, 0);
    g.setTransform(1, 0, 0, 1, 0, 0);
    return out;
  }

  const normalize = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
  async function rawEmbed(c112) {
    const input = chwTensor(ctx2d(c112).getImageData(0, 0, 112, 112).data, 112 * 112, 127.5, 127.5);
    const outs = await Engine.run(Engine.rec, { [Engine.rec.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, 112, 112]) });
    return normalize(Float32Array.from(outs[Engine.rec.outputNames[0]].data));
  }
  // เวกเตอร์ใบหน้า = ค่าเฉลี่ยของภาพปกติและภาพกลับด้านซ้าย-ขวา (ลดผลของมุมหน้า/แสงด้านเดียว)
  async function embed(canvas, face) {
    const crop = align(canvas, face.kps, 112);
    const flip = makeCanvas(112, 112);
    const g = ctx2d(flip);
    g.translate(112, 0); g.scale(-1, 1); g.drawImage(crop, 0, 0);
    const a = await rawEmbed(crop), b = await rawEmbed(flip);
    return normalize(a.map((x, i) => x + b[i]));
  }
  const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  // ============================================================ คุณภาพภาพ
  function pose(kps) {
    const [le, re, nose, lm, rm] = kps;
    const ev = [re[0] - le[0], re[1] - le[1]];
    const ed = Math.hypot(ev[0], ev[1]) + 1e-6;
    const roll = (Math.atan2(ev[1], ev[0]) * 180) / Math.PI;
    const m = mid(le, re);
    const axis = [ev[0] / ed, ev[1] / ed];
    const yawRatio = ((nose[0] - m[0]) * axis[0] + (nose[1] - m[1]) * axis[1]) / ed;
    const yaw = (Math.atan(yawRatio / 0.55) * 180) / Math.PI;
    const up = [-axis[1], axis[0]];
    const mm = mid(lm, rm);
    const eyeNose = (nose[0] - m[0]) * up[0] + (nose[1] - m[1]) * up[1];
    const noseMouth = (mm[0] - nose[0]) * up[0] + (mm[1] - nose[1]) * up[1];
    return { yaw, roll, pitchRatio: eyeNose / (eyeNose + noseMouth + 1e-6) };
  }

  function rgbToHsv(r, g, b) {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d) {
      if (mx === r) h = ((g - b) / d) % 6; else if (mx === g) h = (b - r) / d + 2; else h = (r - g) / d + 4;
      h *= 60; if (h < 0) h += 360;
    }
    return [h, mx ? d / mx : 0, mx / 255];
  }
  function inPoly(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  function regionStats(canvas, test, box) {
    const x0 = Math.max(0, Math.floor(box[0])), y0 = Math.max(0, Math.floor(box[1]));
    const x1 = Math.min(canvas.width, Math.ceil(box[2])), y1 = Math.min(canvas.height, Math.ceil(box[3]));
    if (x1 - x0 < 2 || y1 - y0 < 2) return null;
    const d = ctx2d(canvas).getImageData(x0, y0, x1 - x0, y1 - y0).data;
    let n = 0, s = 0, ss = 0, tint = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      if (!test(x + 0.5, y + 0.5)) continue;
      const i = ((y - y0) * (x1 - x0) + (x - x0)) * 4;
      const L = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      const [h, sat, v] = rgbToHsv(d[i], d[i + 1], d[i + 2]);
      if (h >= 150 && h <= 280 && sat > 0.24 && v > 0.12) tint++;
      n++; s += L; ss += L * L;
    }
    if (!n) return null;
    const mean = s / n;
    return { mean, std: Math.sqrt(Math.max(0, ss / n - mean * mean)), tint: tint / n };
  }
  // ตรวจสิ่งบดบังดวงตา (แว่นกันแดด): บริเวณตามีสีที่ไม่ใช่โทนผิว/ตา (เลนส์ปรอท/เลนส์สี) หรือมืดและเรียบผิดปกติ (เลนส์ดำ)
  function eyeOcclusion(canvas, P) {
    const ipd = dist(P[38], P[88]);
    const eyeMid = mid(P[38], P[88]), mouthMid = mid(P[52], P[61]);
    const dl = dist(eyeMid, mouthMid) || 1;
    const down = [(mouthMid[0] - eyeMid[0]) / dl, (mouthMid[1] - eyeMid[1]) / dl];
    const res = [];
    for (const idx of [EYE_L, EYE_R]) {
      const pts = idx.map((i) => P[i]);
      const c = [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];
      const poly = pts.map(([x, y]) => [c[0] + (x - c[0]) * 1.6, c[1] + (y - c[1]) * 1.6]);
      const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
      const eye = regionStats(canvas, (x, y) => inPoly(x, y, poly), [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]);
      const cc = [c[0] + down[0] * 0.55 * ipd, c[1] + down[1] * 0.55 * ipd], r = 0.18 * ipd;
      const cheek = regionStats(canvas, (x, y) => (x - cc[0]) ** 2 + (y - cc[1]) ** 2 <= r * r, [cc[0] - r, cc[1] - r, cc[0] + r, cc[1] + r]);
      if (eye && cheek) res.push({ tint: eye.tint, lum: eye.mean / Math.max(cheek.mean, 1), tex: eye.std / Math.max(cheek.std, 1) });
    }
    if (!res.length) return false;
    const avg = (k) => res.reduce((s, r) => s + r[k], 0) / res.length;
    return avg("tint") > 0.5 || (avg("lum") < 0.35 && avg("tex") < 0.8);
  }

  function assess(canvas, face) {
    const [x1, y1, x2, y2] = face.bbox;
    const fw = x2 - x1, fh = y2 - y1;
    const crop = align(canvas, face.kps, 112);
    const d = ctx2d(crop).getImageData(0, 0, 112, 112).data;
    const gray = new Float32Array(112 * 112);
    let sum = 0;
    for (let i = 0; i < gray.length; i++) { gray[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]; sum += gray[i]; }
    const bright = sum / gray.length;
    let vs = 0;
    for (const v of gray) vs += (v - bright) ** 2;
    const contrast = Math.sqrt(vs / gray.length);
    let ls = 0, lss = 0, ln = 0;
    for (let y = 1; y < 111; y++) for (let x = 1; x < 111; x++) {
      const i = y * 112 + x;
      const l = gray[i - 1] + gray[i + 1] + gray[i - 112] + gray[i + 112] - 4 * gray[i];
      ls += l; lss += l * l; ln++;
    }
    const sharp = lss / ln - (ls / ln) ** 2;
    const p = pose(face.kps);
    const edge = x1 < 2 || y1 < 2 || x2 > canvas.width - 2 || y2 > canvas.height - 2;
    const occluded = face.lmk ? eyeOcclusion(canvas, face.lmk) : false;
    const warnings = [];
    let penalty = 0;
    const small = Math.min(fw, fh);
    if (small < CFG.minFacePx) { warnings.push(`ใบหน้าเล็กมาก (~${Math.round(small)} px) ความละเอียดไม่พอ`); penalty += 2; }
    else if (small < 80) { warnings.push(`ใบหน้าค่อนข้างเล็ก (~${Math.round(small)} px)`); penalty += 1; }
    if (face.score < 0.65) { warnings.push("ความมั่นใจในการตรวจจับใบหน้าต่ำ"); penalty += 1; }
    if (occluded) { warnings.push("อาจมีแว่นกันแดดหรือสิ่งบดบังบริเวณดวงตา"); penalty += 2; }
    if (Math.abs(p.yaw) > 35) { warnings.push(`ใบหน้าหันข้างมาก (ประมาณ ${Math.round(Math.abs(p.yaw))}°)`); penalty += 2; }
    else if (Math.abs(p.yaw) > 20) { warnings.push(`ใบหน้าเอียงหันเล็กน้อย (ประมาณ ${Math.round(Math.abs(p.yaw))}°)`); penalty += 1; }
    if (Math.abs(p.roll) > 25) { warnings.push(`ศีรษะเอียงมาก (~${Math.round(Math.abs(p.roll))}°)`); penalty += 1; }
    if (p.pitchRatio < 0.33 || p.pitchRatio > 0.80) { warnings.push("ใบหน้าก้มหรือเงยมาก"); penalty += 1; }
    if (sharp < 40) { warnings.push("ภาพเบลอ/ไม่คมชัด"); penalty += 2; }
    else if (sharp < 90) { warnings.push("ภาพค่อนข้างไม่คมชัด"); penalty += 1; }
    if (bright < 55) { warnings.push("ภาพมืดเกินไป"); penalty += 1; }
    else if (bright > 200) { warnings.push("ภาพสว่างจ้าเกินไป"); penalty += 1; }
    if (contrast < 25) { warnings.push("คอนทราสต์ต่ำ"); penalty += 1; }
    if (edge) { warnings.push("ใบหน้าชิด/ถูกตัดที่ขอบภาพ"); penalty += 1; }
    return {
      det_score: face.score, face_px: [Math.round(fw), Math.round(fh)], sharpness: sharp, brightness: bright, contrast,
      yaw: p.yaw, roll: p.roll, pitch_ratio: p.pitchRatio, occluded, warnings, grade: penalty === 0 ? "good" : penalty <= 2 ? "fair" : "poor",
    };
  }

  const bandFor = (score) => BANDS.find((b) => score >= b.min);
  function reliability(qa, qb, sameImage) {
    const order = { good: 0, fair: 1, poor: 2 };
    const worst = order[qa.grade] >= order[qb.grade] ? qa.grade : qb.grade;
    const text = {
      good: "คุณภาพภาพโดยรวมเหมาะสมต่อการเปรียบเทียบ",
      fair: "คุณภาพภาพพอใช้ ผลอาจคลาดเคลื่อนได้ ควรใช้ด้วยความระมัดระวัง",
      poor: "คุณภาพภาพไม่ดี ผลการเปรียบเทียบเชื่อถือได้ต่ำ ควรหาภาพที่ดีกว่า",
    }[worst];
    const notes = [...qa.warnings.map((w) => `ภาพ A: ${w}`), ...qb.warnings.map((w) => `ภาพ B: ${w}`)];
    if (Math.abs(qa.yaw - qb.yaw) > 25) notes.push(`มุมหันหน้าของสองภาพต่างกันมาก (~${Math.round(Math.abs(qa.yaw - qb.yaw))}°) คะแนนมักต่ำกว่าความเป็นจริง`);
    if (sameImage) notes.unshift("เลือกภาพและใบหน้าเดียวกันทั้งสองฝั่ง คะแนนจึงสูงสุดโดยธรรมชาติ ไม่มีความหมายเชิงการพิสูจน์");
    return { grade: worst, text, notes };
  }

  function measures(fa, fb) {
    const ipdA = dist(fa.lmk[38], fa.lmk[88]), ipdB = dist(fb.lmk[38], fb.lmk[88]);
    return MEASURES.map(([label, fn, expr]) => {
      const a = fn(fa.lmk) / ipdA, b = fn(fb.lmk) / ipdB;
      const diff = (Math.abs(a - b) / ((a + b) / 2)) * 100;
      return { label, a, b, diff, expr: !!expr, status: diff <= 5 ? "close" : diff <= 10 ? "slight" : "large" };
    });
  }
  const STATUS_TH = { close: "ใกล้เคียง", slight: "ต่างเล็กน้อย", large: "ต่างชัด" };

  function cropRegion(aligned, key) {
    const [x1, y1, x2, y2] = REGIONS[key][1];
    const w = aligned.width, h = aligned.height;
    const sx = Math.floor(x1 * w), sy = Math.floor(y1 * h), sw = Math.floor(x2 * w) - sx, sh = Math.floor(y2 * h) - sy;
    const c = makeCanvas(sw, sh);
    ctx2d(c).drawImage(aligned, sx, sy, sw, sh, 0, 0, sw, sh);
    return c;
  }

  // เส้นโครงหน้าในพิกัดภาพใบหน้าที่จัดตำแหน่งแล้ว (256×256)
  function meshSvg(pts, cls) {
    const f = (i) => `${pts[i][0].toFixed(1)},${pts[i][1].toFixed(1)}`;
    let s = `<g class="mesh ${cls}">`;
    for (const [idx, closed] of MESH) s += `<${closed ? "polygon" : "polyline"} points="${idx.map(f).join(" ")}"/>`;
    s += `<circle cx="${pts[38][0].toFixed(1)}" cy="${pts[38][1].toFixed(1)}" r="2.6"/><circle cx="${pts[88][0].toFixed(1)}" cy="${pts[88][1].toFixed(1)}" r="2.6"/></g>`;
    return s;
  }
  function drawMesh(canvas, pts, color) {
    const c = makeCanvas(canvas.width, canvas.height);
    const g = ctx2d(c);
    g.drawImage(canvas, 0, 0);
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 1.6; g.lineJoin = "round";
    for (const [idx, closed] of MESH) {
      g.beginPath();
      idx.forEach((i, k) => (k ? g.lineTo(pts[i][0], pts[i][1]) : g.moveTo(pts[i][0], pts[i][1])));
      if (closed) g.closePath();
      g.stroke();
    }
    for (const i of [38, 88]) { g.beginPath(); g.arc(pts[i][0], pts[i][1], 2.8, 0, 7); g.fill(); }
    return c;
  }

  // ============================================================ UI: เลือกภาพ
  function showError(msg, detail) {
    const el = $("#error");
    el.textContent = msg || "";
    el.hidden = !msg;
    const d = $("#diag");
    d.textContent = detail || "";
    d.hidden = !detail;
  }
  function setStatus(slot, msg, bad = false) { const el = $(`#status-${slot}`); el.textContent = msg || ""; el.classList.toggle("bad", !!bad); }
  function refreshButtons() {
    const ready = SLOTS.every((s) => state[s] && state[s].entry.faces.length > 0);
    $("#go").disabled = !ready || state.comparing || state.busy;
    $("#swap").disabled = !(state.A && state.B);
    $("#clear").disabled = !(state.A || state.B);
    SLOTS.forEach((s) => ($(`#reset-${s}`).hidden = !state[s]));
  }

  const emptyHtml = (slot) => `<div class="empty"><div class="empty-icon" aria-hidden="true"></div><p>แตะเพื่อเลือกภาพ หรือลากภาพมาวางที่นี่</p><div class="btnrow">
      <label class="btn primary">เลือกรูป<input type="file" accept="image/*" data-pick="${slot}" hidden></label>
      <label class="btn">ถ่ายรูป<input type="file" accept="image/*" capture="user" data-pick="${slot}" hidden></label></div></div>`;

  function bindPickers(root) {
    root.querySelectorAll("input[data-pick]").forEach((inp) => {
      inp.addEventListener("change", () => { const f = inp.files && inp.files[0]; inp.value = ""; upload(inp.dataset.pick, f); });
    });
  }

  function dropEntry(slot) {
    const st = state[slot];
    if (st && st.entry.previewUrl) URL.revokeObjectURL(st.entry.previewUrl);
    state[slot] = null;
    hideResult();
  }

  async function upload(slot, file) {
    if (!file) return;
    showError("");
    dropEntry(slot);
    $(`#chips-${slot}`).innerHTML = "";
    const stage = $(`#stage-${slot}`);
    stage.innerHTML = '<div class="photo loading"><div class="spinner"></div></div>';
    state.busy = true;
    refreshButtons();
    try {
      setStatus(slot, Engine.det ? "กำลังตรวจจับใบหน้า…" : "กำลังโหลดโมเดลตรวจจับ (~10 MB) …");
      const [canvas] = await Promise.all([decodeFile(file), Engine.loadSmall()]);
      Engine.loadBig().catch(() => {}); // เริ่มโหลดโมเดลจดจำใบหน้าเบื้องหลัง
      setStatus(slot, "กำลังตรวจจับใบหน้าและจุดโครงหน้า…");
      const [faces, hash] = await Promise.all([detect(canvas), sha256(file)]);
      const s = Math.min(1, CFG.previewSide / Math.max(canvas.width, canvas.height));
      let pc = canvas;
      if (s < 1) { pc = makeCanvas(Math.round(canvas.width * s), Math.round(canvas.height * s)); ctx2d(pc).drawImage(canvas, 0, 0, pc.width, pc.height); }
      const previewUrl = URL.createObjectURL(await toBlob(pc));
      state[slot] = { entry: { filename: file.name || "image", size: file.size, sha256: hash, canvas, faces, previewUrl }, face: 0 };
      renderSlot(slot);
    } catch (e) {
      stage.innerHTML = emptyHtml(slot);
      bindPickers(stage);
      setStatus(slot, e.message || "เกิดข้อผิดพลาด", true);
      if (e.detail) showError("รายละเอียดทางเทคนิค (ส่งภาพหน้านี้ให้ผู้พัฒนาได้):", e.detail);
    } finally {
      state.busy = false;
      refreshButtons();
    }
  }

  const svgEl = (name, attrs) => { const e = document.createElementNS("http://www.w3.org/2000/svg", name); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };

  function renderSlot(slot) {
    const { entry, face } = state[slot];
    const stage = $(`#stage-${slot}`);
    stage.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "photo";
    const img = document.createElement("img");
    img.alt = `ภาพบุคคล ${slot}`;
    img.src = entry.previewUrl;
    wrap.append(img);
    const W = entry.canvas.width, H = entry.canvas.height, M = Math.max(W, H);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
    entry.faces.forEach((f, i) => {
      const [x1, y1, x2, y2] = f.bbox;
      const sel = i === face;
      svg.append(svgEl("rect", { class: "face" + (sel ? " sel" : ""), x: x1, y: y1, width: x2 - x1, height: y2 - y1, rx: 6 }));
      if (entry.faces.length > 1) {
        const t = svgEl("text", { x: x1 + 6, y: y1 + M * 0.035, "font-size": M * 0.032 });
        t.textContent = String(i + 1);
        const hit = svgEl("rect", { class: "hit", x: x1, y: y1, width: x2 - x1, height: y2 - y1 });
        hit.addEventListener("click", () => selectFace(slot, i));
        svg.append(t, hit);
      }
      if (sel && f.lmk) f.lmk.forEach(([lx, ly]) => svg.append(svgEl("circle", { cx: lx, cy: ly, r: Math.max(1.6, M * 0.0022) })));
    });
    wrap.append(svg);
    stage.append(wrap);
    const n = entry.faces.length;
    if (n === 0) setStatus(slot, "ไม่พบใบหน้าในภาพนี้ ลองภาพที่เห็นหน้าชัดและตรงมากขึ้น", true);
    else if (n === 1) setStatus(slot, `${entry.filename} · พบ 1 ใบหน้า · จุดโครงหน้า 106 จุด`);
    else setStatus(slot, `${entry.filename} · พบ ${n} ใบหน้า — แตะกรอบเพื่อเลือก (กำลังเลือกลำดับ ${face + 1})`);
    renderChips(slot);
  }

  function renderChips(slot) {
    const box = $(`#chips-${slot}`);
    box.innerHTML = "";
    const st = state[slot];
    if (!st || !st.entry.faces.length) return;
    const q = st.entry.faces[st.face].quality;
    const add = (text, cls) => { const c = document.createElement("span"); c.className = "chip " + cls; c.textContent = text; box.append(c); };
    add(`คุณภาพ: ${GRADE_TH[q.grade]}`, "g-" + q.grade);
    q.warnings.forEach((w) => add(w, "warn"));
  }

  function selectFace(slot, i) { state[slot].face = i; hideResult(); renderSlot(slot); }

  function resetSlot(slot) {
    dropEntry(slot);
    const stage = $(`#stage-${slot}`);
    stage.innerHTML = emptyHtml(slot);
    bindPickers(stage);
    setStatus(slot, "");
    $(`#chips-${slot}`).innerHTML = "";
    refreshButtons();
  }

  bindPickers(document);
  SLOTS.forEach((slot) => {
    const stage = $(`#stage-${slot}`);
    ["dragenter", "dragover"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.remove("drag"); }));
    stage.addEventListener("drop", (e) => { const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]; if (f) upload(slot, f); });
    $(`#reset-${slot}`).addEventListener("click", () => resetSlot(slot));
  });
  ["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => e.preventDefault()));
  $("#clear").addEventListener("click", () => { SLOTS.forEach(resetSlot); showError(""); });
  $("#swap").addEventListener("click", () => {
    [state.A, state.B] = [state.B, state.A];
    hideResult();
    SLOTS.forEach((s) => (state[s] ? renderSlot(s) : null));
    refreshButtons();
  });

  // ============================================================ เปรียบเทียบ
  function hideResult() { $("#result").hidden = true; state.last = null; clearReport(); }

  $("#go").addEventListener("click", async () => {
    showError("");
    state.comparing = true;
    const btn = $("#go");
    refreshButtons();
    try {
      if (!Engine.rec) { btn.textContent = "กำลังโหลดโมเดลจดจำใบหน้า…"; await Engine.loadBig(); }
      btn.textContent = "กำลังวิเคราะห์…";
      const a = state.A, b = state.B;
      const fa = a.entry.faces[a.face], fb = b.entry.faces[b.face];
      const ea = await embed(a.entry.canvas, fa);
      const eb = await embed(b.entry.canvas, fb);
      const c = cosine(ea, eb);
      const score = Math.round(c * 1000) / 10;
      const sameImage = a.entry.sha256 === b.entry.sha256 && a.face === b.face;
      const S = CFG.alignSize;
      const mA = alignMatrix(fa.kps, S), mB = alignMatrix(fb.kps, S);
      state.last = {
        score, cosine: c, band: bandFor(score), rel: reliability(fa.quality, fb.quality, sameImage),
        alignedA: align(a.entry.canvas, fa.kps, S), alignedB: align(b.entry.canvas, fb.kps, S),
        lmkA: applyM(mA, fa.lmk), lmkB: applyM(mB, fb.lmk), measures: measures(fa, fb),
        fa, fb, a, b, sameImage,
      };
      renderResult(state.last);
    } catch (e) {
      showError(e.message || "เปรียบเทียบไม่สำเร็จ", e.detail);
    } finally {
      state.comparing = false;
      btn.textContent = "เปรียบเทียบ";
      refreshButtons();
    }
  });

  const RING_C = 2 * Math.PI * 52;
  function renderResult(r) {
    $("#result").hidden = false;
    clearReport();
    const lv = r.band.level;
    $("#score-card").dataset.level = lv;
    $("#score").textContent = r.score.toFixed(1);
    $("#ring-fg").style.strokeDasharray = `${(RING_C * Math.max(0, Math.min(100, r.score))) / 100} ${RING_C}`;
    $("#band").textContent = r.band.label;
    $("#band-text").textContent = r.band.text;
    $("#cosine").textContent = `cosine similarity = ${r.cosine.toFixed(4)}`;
    $("#marker").style.left = Math.max(0, Math.min(100, r.score)) + "%";
    $("#disclaimer").textContent = DISCLAIMER;

    const rel = $("#reliability");
    rel.className = "reliability " + r.rel.grade;
    rel.textContent = "";
    const bold = document.createElement("strong");
    bold.textContent = `คุณภาพภาพโดยรวม: ${GRADE_TH[r.rel.grade]}`;
    rel.append(bold, " — " + r.rel.text);
    if (r.rel.notes.length) {
      const ul = document.createElement("ul");
      r.rel.notes.forEach((n) => { const li = document.createElement("li"); li.textContent = n; ul.append(li); });
      rel.append(ul);
    }
    const ua = r.alignedA.toDataURL("image/png"), ub = r.alignedB.toDataURL("image/png");
    for (const id of ["aligned-A", "wipe-A", "blend-A"]) $("#" + id).src = ua;
    for (const id of ["aligned-B", "wipe-B", "blend-B"]) $("#" + id).src = ub;
    const mA = meshSvg(r.lmkA, "ma"), mB = meshSvg(r.lmkB, "mb");
    $("#mesh-A").innerHTML = mA;
    $("#mesh-B").innerHTML = mB;
    $("#mesh-wipe").innerHTML = mA + mB;
    $("#mesh-blend").innerHTML = mA + mB;

    const regions = $("#regions");
    regions.textContent = "";
    Object.keys(REGIONS).forEach((k) => {
      const row = document.createElement("div");
      row.className = "region";
      [["A", r.alignedA], ["B", r.alignedB]].forEach(([who, al]) => {
        const fig = document.createElement("figure");
        fig.className = "side-" + who.toLowerCase();
        const im = document.createElement("img");
        im.src = cropRegion(al, k).toDataURL("image/png");
        im.alt = `${REGIONS[k][0]} ${who}`;
        const cap = document.createElement("figcaption");
        cap.textContent = `${who} · ${REGIONS[k][0]}`;
        fig.append(im, cap);
        row.append(fig);
      });
      regions.append(row);
    });
    renderMeasures(r);
    renderQuality(r);
    $("#model-name").textContent = "โมเดล: " + CFG.modelName;
    setMode("side");
    $("#result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  function renderMeasures(r) {
    const body = $("#measure-body");
    body.textContent = "";
    r.measures.forEach((m) => {
      const tr = document.createElement("tr");
      const max = Math.max(m.a, m.b) || 1;
      tr.innerHTML = `<th scope="row"></th><td class="num va"></td><td class="num vb"></td>
        <td class="bars"><span class="bar a" style="width:${((m.a / max) * 100).toFixed(1)}%"></span><span class="bar b" style="width:${((m.b / max) * 100).toFixed(1)}%"></span></td>
        <td><span class="st st-${m.status}"></span></td>`;
      tr.querySelector("th").textContent = m.label + (m.expr ? " *" : "");
      tr.querySelector(".va").textContent = m.a.toFixed(2);
      tr.querySelector(".vb").textContent = m.b.toFixed(2);
      tr.querySelector(".st").textContent = `${STATUS_TH[m.status]} ${m.diff.toFixed(1)}%`;
      body.append(tr);
    });
    const counts = { close: 0, slight: 0, large: 0 };
    r.measures.forEach((m) => counts[m.status]++);
    $("#measure-sum").textContent = `ใกล้เคียง ${counts.close} · ต่างเล็กน้อย ${counts.slight} · ต่างชัด ${counts.large} (จาก ${r.measures.length} รายการ)`;
    const warn = measureWarnings(r);
    const w = $("#measure-warn");
    w.textContent = warn.join(" · ");
    w.hidden = !warn.length;
  }

  function measureWarnings(r) {
    const warn = [];
    const qa = r.fa.quality, qb = r.fb.quality;
    if (Math.abs(qa.yaw) > 20 || Math.abs(qb.yaw) > 20 || Math.abs(qa.yaw - qb.yaw) > 15) warn.push("มุมหันหน้าของสองภาพต่างกัน ค่าความกว้าง (แนวนอน) อาจคลาดเคลื่อนมาก");
    if (Math.abs(qa.pitch_ratio - qb.pitch_ratio) > 0.08) warn.push("มุมก้ม/เงยของสองภาพต่างกัน ค่าความยาว (แนวตั้ง) อาจคลาดเคลื่อน");
    if (qa.occluded || qb.occluded) warn.push("มีสิ่งบดบังดวงตา ค่าที่เกี่ยวกับตาและคิ้วอาจไม่ถูกต้อง");
    return warn;
  }

  const signed = (v) => { const n = Math.round(v); return n === 0 ? "0°" : `${n > 0 ? "+" : ""}${n}°`; };
  const QROWS = [
    ["ความมั่นใจการตรวจจับ", (q) => q.det_score.toFixed(2)],
    ["ขนาดใบหน้า (px)", (q) => `${q.face_px[0]}×${q.face_px[1]}`],
    ["ความคมชัด", (q) => String(Math.round(q.sharpness))],
    ["ความสว่างเฉลี่ย", (q) => String(Math.round(q.brightness))],
    ["คอนทราสต์", (q) => String(Math.round(q.contrast))],
    ["มุมหันหน้า (ประมาณ)", (q) => signed(q.yaw)],
    ["มุมเอียงศีรษะ (ประมาณ)", (q) => signed(q.roll)],
    ["สิ่งบดบังดวงตา", (q) => (q.occluded ? "อาจมี" : "ไม่พบ")],
    ["เกรดคุณภาพ", (q) => GRADE_TH[q.grade]],
  ];
  function renderQuality(r) {
    const body = $("#quality-body");
    body.textContent = "";
    QROWS.forEach(([label, fn]) => {
      const tr = document.createElement("tr");
      const t0 = document.createElement("th"), ta = document.createElement("td"), tb = document.createElement("td");
      t0.scope = "row"; t0.textContent = label; ta.textContent = fn(r.fa.quality); tb.textContent = fn(r.fb.quality);
      if (label === "เกรดคุณภาพ") { ta.className = "g-" + r.fa.quality.grade; tb.className = "g-" + r.fb.quality.grade; }
      if (label === "สิ่งบดบังดวงตา") { if (r.fa.quality.occluded) ta.className = "g-poor"; if (r.fb.quality.occluded) tb.className = "g-poor"; }
      tr.append(t0, ta, tb);
      body.append(tr);
    });
  }

  // ---------- โหมดแสดงผล
  function setMode(mode) {
    document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
    $("#view-side").hidden = mode !== "side";
    $("#view-wipe").hidden = mode !== "wipe";
    $("#view-blend").hidden = mode !== "blend";
  }
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  const wipe = $("#wipe-range");
  const applyWipe = () => { $("#wipe-A").style.clipPath = `inset(0 ${100 - wipe.value}% 0 0)`; $("#wipe-handle").style.left = wipe.value + "%"; };
  wipe.addEventListener("input", applyWipe);
  applyWipe();
  const blend = $("#blend-range");
  blend.addEventListener("input", () => ($("#blend-A").style.opacity = blend.value / 100));
  $("#mesh-toggle").addEventListener("change", (e) => $("#compare-card").classList.toggle("meshes-off", !e.target.checked));

  // ============================================================ รายงาน PDF (วาดเป็นหน้า A4 แล้วรวมเป็น PDF)
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const LEVEL_HEX = { high: "#a21caf", "mid-high": "#7c3aed", mid: "#d97706", low: "#0284c7", "very-low": "#64748b" };
  const ST_HEX = { close: "#0284c7", slight: "#d97706", large: "#c026d3" };
  const COL_A = "#0d9488", COL_B = "#ea580c";
  const PAGE_W = 794, PAGE_H = 1123, MARGIN = 48;

  function annotated(entry, chosen) {
    const src = entry.canvas;
    const s = Math.min(1, 1100 / Math.max(src.width, src.height));
    const c = makeCanvas(Math.round(src.width * s), Math.round(src.height * s));
    const g = ctx2d(c);
    g.drawImage(src, 0, 0, c.width, c.height);
    const lw = Math.max(2, Math.round(Math.max(c.width, c.height) / 400));
    entry.faces.forEach((f, i) => {
      g.lineWidth = lw;
      g.strokeStyle = i === chosen ? "#22c55e" : "#aaaaaa";
      g.strokeRect(f.bbox[0] * s, f.bbox[1] * s, (f.bbox[2] - f.bbox[0]) * s, (f.bbox[3] - f.bbox[1]) * s);
      if (i === chosen) { g.fillStyle = "#f59e0b"; f.kps.forEach(([x, y]) => { g.beginPath(); g.arc(x * s, y * s, lw + 1, 0, 7); g.fill(); }); }
    });
    return c.toDataURL("image/jpeg", 0.88);
  }

  function thaiDate() {
    const d = new Date();
    const date = new Intl.DateTimeFormat("th-TH-u-ca-buddhist", { timeZone: "Asia/Bangkok", day: "numeric", month: "long", year: "numeric" }).format(d);
    const time = new Intl.DateTimeFormat("th-TH", { timeZone: "Asia/Bangkok", hour: "2-digit", minute: "2-digit", hour12: false }).format(d);
    return `${date} เวลา ${time} น.`;
  }

  function reportBlocks(r, info) {
    const cv = (k) => esc((info[k] || "").trim() || "-");
    const col = LEVEL_HEX[r.band.level];
    const pos = Math.max(0, Math.min(100, r.score));
    const blocks = [];
    blocks.push(`<div class="rp-hero"><div class="rp-title">รายงานการเปรียบเทียบใบหน้า</div><div class="rp-sub">Face Comparison Report — รายงานผลวิเคราะห์เชิงเทคนิคเบื้องต้น</div></div>
      <table class="rp-kv"><tr><th>เลขที่เรื่อง / อ้างอิง</th><td>${cv("case_no")}</td></tr><tr><th>วัตถุประสงค์</th><td>${cv("purpose")}</td></tr>
      <tr><th>ผู้จัดทำ / ผู้ตรวจ</th><td>${cv("examiner")}</td></tr><tr><th>วันที่และเวลาที่ออกรายงาน</th><td>${esc(thaiDate())}</td></tr></table>`);
    blocks.push(`<div class="rp-h">1. ผลการเปรียบเทียบ</div>
      <div class="rp-score" style="border-color:${col}"><div><div class="rp-num" style="color:${col}">${r.score.toFixed(1)}</div><div class="rp-small">คะแนนความคล้าย (0–100)</div></div>
      <div><div class="rp-band" style="background:${col}">ระดับ: ${esc(r.band.label)}</div><div>${esc(r.band.text)}</div><div class="rp-small">ค่า cosine similarity = ${r.cosine.toFixed(4)}</div></div></div>
      <div class="rp-gauge"><div class="rp-zones"><i style="flex:20;background:${LEVEL_HEX["very-low"]}"></i><i style="flex:10;background:${LEVEL_HEX.low}"></i><i style="flex:10;background:${LEVEL_HEX.mid}"></i><i style="flex:15;background:${LEVEL_HEX["mid-high"]}"></i><i style="flex:45;background:${LEVEL_HEX.high}"></i></div>
      <div class="rp-mark" style="left:${pos}%"></div>
      <div class="rp-ticks"><span style="left:0%">0</span><span style="left:20%">20</span><span style="left:30%">30</span><span style="left:40%">40</span><span style="left:55%">55</span><span style="left:100%">100</span></div></div>
      <div class="rp-small">ช่วงสี: ต่ำมาก &lt;20 · ต่ำ 20–30 · ก้ำกึ่ง 30–40 · สูง 40–55 · สูงมาก ≥55 (เกณฑ์เชิงแนวทางทั่วไป ไม่ได้สอบเทียบกับข้อมูลของงานนี้ สีใช้แสดงระดับเท่านั้น ไม่ได้บอกว่าผ่านหรือไม่ผ่าน)</div>`);
    blocks.push(`<div class="rp-b"><b>คุณภาพภาพโดยรวม: ${GRADE_TH[r.rel.grade]}</b> — ${esc(r.rel.text)}${r.rel.notes.length ? "<ul>" + r.rel.notes.map((n) => `<li>${esc(n)}</li>`).join("") + "</ul>" : ""}</div>
      <div class="rp-h">ข้อควรระวังสำคัญ</div><div class="rp-warn">${esc(DISCLAIMER)}</div>`);
    const src = (label, p, color) => `<div class="rp-src"><img src="${annotated(p.entry, p.face)}"><div class="rp-cell"><b style="color:${color}">${label}</b> · ${esc(p.entry.filename)}</div>
      <div class="rp-small">ใบหน้าที่เลือก: ลำดับ ${p.face + 1} จากที่ตรวจพบ ${p.entry.faces.length} ใบหน้า<br>ขนาดไฟล์ ${Math.round(p.entry.size / 1024)} KB · ${p.entry.canvas.width}×${p.entry.canvas.height} px<br>SHA-256: <span class="rp-hash">${esc(p.entry.sha256)}</span></div></div>`;
    blocks.push(`<div class="rp-h">2. ภาพที่นำมาเปรียบเทียบ</div><div class="rp-two">${src("ภาพ A", r.a, COL_A)}${src("ภาพ B", r.b, COL_B)}</div>
      <div class="rp-small">กรอบสีเขียว = ใบหน้าที่ใช้เปรียบเทียบ · กรอบสีเทา = ใบหน้าอื่นในภาพ · จุดสีส้ม = จุดอ้างอิง 5 จุด (ตา จมูก มุมปาก)</div>`);
    const ma = drawMesh(r.alignedA, r.lmkA, COL_A).toDataURL("image/png"), mb = drawMesh(r.alignedB, r.lmkB, COL_B).toDataURL("image/png");
    blocks.push(`<div class="rp-h">3. เปรียบเทียบใบหน้าเคียงข้างกัน (จัดตำแหน่งอัตโนมัติ + จุดโครงหน้า 106 จุด)</div><div class="rp-two rp-faces"><div><img src="${ma}"><div class="rp-cap" style="color:${COL_A}">ภาพ A</div></div><div><img src="${mb}"><div class="rp-cap" style="color:${COL_B}">ภาพ B</div></div></div>`);
    Object.keys(REGIONS).forEach((k, i) => blocks.push((i === 0 ? `<div class="rp-small">ส่วนประกอบของใบหน้า (ใช้ประกอบการพิจารณาด้วยสายตา โมเดลไม่ได้ให้คะแนนแยกรายส่วน)</div>` : "") +
      `<div class="rp-two rp-reg"><div><img src="${cropRegion(r.alignedA, k).toDataURL("image/png")}"><div class="rp-cap">A · ${REGIONS[k][0]}</div></div><div><img src="${cropRegion(r.alignedB, k).toDataURL("image/png")}"><div class="rp-cap">B · ${REGIONS[k][0]}</div></div></div>`));
    blocks.push(`<div class="rp-h">4. สัดส่วนใบหน้าจากจุดโครงหน้า 106 จุด</div>
      ${measureWarnings(r).map((w) => `<div class="rp-mwarn">${esc(w)}</div>`).join("")}
      <div class="rp-small">ค่า = ระยะทาง ÷ ระยะระหว่างรูม่านตา · ความต่าง = |A−B| ÷ ค่าเฉลี่ย · ≤5% ใกล้เคียง · 5–10% ต่างเล็กน้อย · &gt;10% ต่างชัด · * ขึ้นกับการแสดงสีหน้า · ไม่ได้นำไปคิดคะแนน</div>
      <table class="rp-q"><tr><th>รายการ</th><th style="color:${COL_A}">A</th><th style="color:${COL_B}">B</th><th>ความต่าง</th></tr>${r.measures.map((m) => `<tr><td>${esc(m.label)}${m.expr ? " *" : ""}</td><td>${m.a.toFixed(2)}</td><td>${m.b.toFixed(2)}</td><td style="color:${ST_HEX[m.status]};font-weight:700">${STATUS_TH[m.status]} ${m.diff.toFixed(1)}%</td></tr>`).join("")}</table>`);
    blocks.push(`<div class="rp-h">5. คุณภาพภาพและท่าทางใบหน้า</div><table class="rp-q"><tr><th>รายการ</th><th style="color:${COL_A}">ภาพ A</th><th style="color:${COL_B}">ภาพ B</th></tr>${QROWS.map(([l, f]) => `<tr><td>${l}</td><td>${esc(f(r.fa.quality))}</td><td>${esc(f(r.fb.quality))}</td></tr>`).join("")}</table>`);
    const method = [
      `โมเดล: ${CFG.modelName}`,
      "ขั้นตอน: (1) ตรวจจับใบหน้าและจุดอ้างอิง 5 จุด (2) หาจุดโครงหน้า 106 จุด (3) จัดตำแหน่งใบหน้าให้ตรงแม่แบบ 112×112 พิกเซล (4) สกัดเวกเตอร์ลักษณะเฉพาะ 512 มิติจากภาพปกติและภาพกลับด้านซ้าย-ขวาแล้วเฉลี่ย (5) คำนวณ cosine similarity",
      "คะแนนความคล้าย = cosine similarity × 100 (ค่ายิ่งสูงยิ่งคล้าย; คนละคนมักได้ใกล้ 0) ไม่ใช่เปอร์เซ็นต์ความน่าจะเป็น",
      "โมเดลจดจำใบหน้าถูกบีบอัดน้ำหนักเป็น 8 บิตเพื่อให้รันบนเบราว์เซอร์ได้ ผลต่างจากโมเดลเต็มไม่เกินประมาณ 1 คะแนนในการทดสอบ",
      "ภาพถูกหมุนตามข้อมูล EXIF และอาจถูกย่อเพื่อประมวลผล; ค่า SHA-256 คำนวณจากไฟล์ที่เลือกต้นฉบับ",
      "การประมวลผลทั้งหมดเกิดในเบราว์เซอร์ของผู้ใช้ ภาพไม่ถูกส่งออกจากอุปกรณ์",
    ];
    blocks.push(`<div class="rp-h">6. วิธีการ</div><ul class="rp-b">${method.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>`);
    if ((info.notes || "").trim()) blocks.push(`<div class="rp-h">7. หมายเหตุของผู้จัดทำ</div><div class="rp-b" style="white-space:pre-wrap">${esc(info.notes.trim())}</div>`);
    blocks.push(`<div class="rp-sig"><div>ลงชื่อ ........................................ ผู้ตรวจสอบ<br><span>(........................................)</span></div><div>ลงชื่อ ........................................ ผู้รับรอง<br><span>(........................................)</span></div></div>`);
    return blocks;
  }

  async function renderReportPages(r, info) {
    const stage = $("#report-stage");
    stage.textContent = "";
    const inner = PAGE_W - 2 * MARGIN, usable = PAGE_H - MARGIN - 70;
    const blocks = reportBlocks(r, info);
    // วัดความสูงแต่ละบล็อก แล้วจัดลงหน้า A4
    const meas = document.createElement("div");
    meas.className = "rp rp-measure";
    meas.style.width = inner + "px";
    meas.innerHTML = blocks.map((b) => `<div class="rp-block">${b}</div>`).join("");
    stage.append(meas);
    await Promise.all([...meas.querySelectorAll("img")].map((im) => im.decode().catch(() => {})));
    const heights = [...meas.children].map((el) => el.getBoundingClientRect().height);
    meas.remove();
    const pages = [[]];
    let used = 0;
    blocks.forEach((b, i) => {
      if (used + heights[i] > usable && pages[pages.length - 1].length) { pages.push([]); used = 0; }
      pages[pages.length - 1].push(b);
      used += heights[i];
    });
    const canvases = [];
    for (let p = 0; p < pages.length; p++) {
      const el = document.createElement("div");
      el.className = "rp rp-page";
      el.style.width = PAGE_W + "px"; el.style.height = PAGE_H + "px";
      el.innerHTML = `<div class="rp-strip"></div><div class="rp-body" style="padding:${MARGIN}px ${MARGIN}px 0">${pages[p].map((b) => `<div class="rp-block">${b}</div>`).join("")}</div>
        <div class="rp-foot"><span>คะแนนความคล้ายไม่ใช่การยืนยันตัวบุคคล — ต้องมีการตรวจสอบโดยผู้เชี่ยวชาญและใช้หลักฐานอื่นประกอบ</span><span>หน้า ${p + 1}/${pages.length}</span></div>`;
      stage.append(el);
      await Promise.all([...el.querySelectorAll("img")].map((im) => im.decode().catch(() => {})));
      canvases.push(await html2canvas(el, { scale: 2, backgroundColor: "#ffffff", logging: false, useCORS: true, width: PAGE_W, height: PAGE_H, windowWidth: PAGE_W, scrollX: 0, scrollY: 0, x: 0, y: 0 }));
      el.remove();
    }
    return canvases;
  }

  function clearReport() {
    $("#report-pages").textContent = "";
    $("#report-preview").hidden = true;
    $("#report-status").textContent = "";
    $("#report-status").className = "status";
  }

  async function getDownloads() {
    try { return window.claude && window.claude.use ? await window.claude.use("downloads") : null; } catch { return null; }
  }

  $("#report-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (!state.last) return;
    const btn = $("#report-btn"), status = $("#report-status");
    btn.disabled = true;
    status.className = "status";
    status.textContent = "กำลังสร้างรายงาน…";
    try {
      if (!(window.html2canvas && window.jspdf)) throw new Error("โหลดไลบรารีสร้าง PDF ไม่สำเร็จ ตรวจสอบอินเทอร์เน็ตแล้วลองใหม่");
      try {
        await Promise.all(["400 14px Sarabun", "700 14px Sarabun", "600 14px 'Chakra Petch'", "700 14px 'Chakra Petch'"].map((f) => document.fonts.load(f)));
      } catch { /* ใช้ฟอนต์สำรอง */ }
      const info = Object.fromEntries(new FormData(ev.target).entries());
      const pages = await renderReportPages(state.last, info);
      const prev = $("#report-pages");
      prev.textContent = "";
      pages.forEach((c, i) => { const im = document.createElement("img"); im.src = c.toDataURL("image/jpeg", 0.8); im.alt = `หน้า ${i + 1} ของรายงาน`; prev.append(im); });
      $("#report-preview").hidden = false;

      const { jsPDF } = window.jspdf;
      const pdf = new jsPDF({ unit: "mm", format: "a4", compress: true });
      pages.forEach((c, i) => { if (i) pdf.addPage(); pdf.addImage(c.toDataURL("image/jpeg", 0.92), "JPEG", 0, 0, 210, 297, undefined, "FAST"); });
      const blob = pdf.output("blob");
      const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
      const filename = `face-comparison-${stamp}.pdf`;
      const dl = await getDownloads();
      if (dl) {
        try {
          await dl.save({ filename, data: blob });
          status.textContent = "บันทึกรายงานแล้ว";
        } catch (e) {
          status.textContent = e && e.code === "declined" ? "ยกเลิกการบันทึก — กดปุ่มอีกครั้งเพื่อบันทึก หรือดูตัวอย่างหน้ารายงานด้านล่าง" : "บันทึกไฟล์ไม่ได้ในมุมมองนี้ ดูตัวอย่างหน้ารายงานด้านล่าง (กดค้างที่ภาพเพื่อบันทึก)";
        }
      } else {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = filename;
        document.body.append(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 60000);
        status.textContent = "สร้างรายงานแล้ว หากไม่มีไฟล์ดาวน์โหลด ให้กดค้างที่ภาพตัวอย่างด้านล่างเพื่อบันทึก";
      }
    } catch (e) {
      status.className = "status bad";
      status.textContent = e.message || "สร้างรายงานไม่สำเร็จ";
    } finally {
      btn.disabled = false;
    }
  });

  refreshButtons();
  modelStatus("idle");
  // โหลดโมเดลเล็กล่วงหน้าเงียบ ๆ เพื่อให้ภาพแรกเร็วขึ้น (ข้อผิดพลาดจะแสดงเมื่อผู้ใช้เลือกภาพ)
  setTimeout(() => Engine.loadSmall().catch(() => {}), 300);
  window.__faceCompare = { Engine, state, detect, embed, cosine, decodeFile, Assets };
})();
