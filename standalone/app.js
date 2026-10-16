"use strict";
/* เปรียบเทียบใบหน้า — ทำงานในเบราว์เซอร์ทั้งหมด (ONNX Runtime Web + SCRFD + ArcFace)
   ภาพไม่ถูกส่งออกจากเครื่อง ไม่มีเซิร์ฟเวอร์ */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const SLOTS = ["A", "B"];

  const CFG = {
    // โมเดลเก็บเป็น base64 ในไฟล์ .txt (บริการโฮสต์ไม่รับ .onnx) แบ่งเป็นส่วน ๆ ละ < 16 MB
    detModel: ["models/det_500m.b64.0.txt"],
    recModel: ["models/w600k_mbf.b64.0.txt", "models/w600k_mbf.b64.1.txt"],
    modelName: "InsightFace buffalo_s (SCRFD det_500m + ArcFace MobileFaceNet w600k_mbf)",
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

  const state = { A: null, B: null, busy: false, comparing: false, last: null, pages: null };

  // ============================================================ ภาพ
  const makeCanvas = (w, h) => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
  const ctx2d = (c, opts) => c.getContext("2d", opts);
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
      g.drawImage(img, 0, 0, w, h);   // เบราว์เซอร์หมุนตาม EXIF ให้แล้ว
      return canvas;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // ============================================================ โหลดไฟล์ประกอบของหน้า (โมเดล/wasm)
  // ที่อยู่ไฟล์ที่เผยแพร่คู่กับหน้า อาจต้องอ้างอิงต่างกันตามที่หน้าถูกเสิร์ฟ จึงลองหลายรูปแบบและจำแบบที่ใช้ได้
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
    async fetchText(path) {
      const order = this.mode != null ? [this.mode] : [0, 1, 2, 3];
      for (const m of order) {
        const url = this.candidates(path)[m];
        try {
          const r = await fetch(url);
          this.log.push(`${r.status} ${url}`);
          if (r.ok) { this.mode = m; return await r.text(); }
        } catch (e) { this.log.push(`ERR ${url} (${e && e.message})`); }
      }
      // ลองไฟล์เล็ก (index.html) เพื่อดูว่าปัญหาอยู่ที่ที่อยู่หรือที่ตัวไฟล์
      this.log.push("— ทดสอบ index.html —");
      for (const url of this.candidates("index.html")) {
        try { const r = await fetch(url, { method: "GET" }); this.log.push(`${r.status} ${url}`); } catch (e) { this.log.push(`ERR ${url}`); }
      }
      this.log.push(`page: ${window.location.href.slice(0, 160)} | base: ${document.baseURI.slice(0, 160)}`);
      const err = new Error(`โหลดโมเดลไม่สำเร็จ (${path})`);
      err.detail = this.log.join("\n");
      throw err;
    },
  };

  // ============================================================ โมเดล
  const Engine = {
    det: null, rec: null, loading: null, chain: Promise.resolve(),
    load() {
      if (!this.loading) {
        this.loading = (async () => {
          if (typeof ort === "undefined") throw new Error("โหลดไลบรารี ONNX Runtime ไม่สำเร็จ");
          ort.env.wasm.numThreads = 1;
          ort.env.wasm.proxy = false;
          const get = async (parts) => {
            const chunks = [];
            let total = 0;
            for (const p of await Promise.all(parts.map(async (p) => (await Assets.fetchText(p)).trim()))) {
              const bin = atob(p);
              const u8 = new Uint8Array(bin.length);
              for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
              chunks.push(u8); total += u8.length;
            }
            const out = new Uint8Array(total);
            let off = 0;
            for (const c of chunks) { out.set(c, off); off += c.length; }
            return out;
          };
          const d = await get(CFG.detModel);   // ไฟล์แรกใช้หาวิธีอ้างอิงที่อยู่ที่ใช้ได้ ก่อนโหลดไฟล์ใหญ่
          const r = await get(CFG.recModel);
          ort.env.wasm.wasmPaths = { "ort-wasm-simd.wasm": Assets.urlFor("ort/ort-wasm-simd.wasm"), "ort-wasm.wasm": Assets.urlFor("ort/ort-wasm.wasm") };
          const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
          try {
            this.det = await ort.InferenceSession.create(d, opts);
            this.rec = await ort.InferenceSession.create(r, opts);
          } catch (e) {
            const err = new Error("เริ่มตัวรันโมเดลไม่สำเร็จ (WebAssembly)");
            err.detail = `${e && e.message}\n${Assets.log.join("\n")}\nwasm: ${JSON.stringify(ort.env.wasm.wasmPaths)}`;
            throw err;
          }
        })().catch((e) => { this.loading = null; throw e; });
      }
      return this.loading;
    },
    // ให้ทำงานทีละงาน (หลาย session บน wasm เธรดเดียว)
    run(session, feeds) {
      const p = this.chain.then(() => session.run(feeds));
      this.chain = p.catch(() => {});
      return p;
    },
  };

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

  async function detectOnce(canvas, size) {
    const w = canvas.width, h = canvas.height;
    const ratio = h / w;
    let nw, nh;
    if (ratio > 1) { nh = size; nw = Math.floor(size / ratio); } else { nw = size; nh = Math.floor(size * ratio); }
    const scale = nh / h;
    const tmp = makeCanvas(size, size);
    const g = ctx2d(tmp, { willReadFrequently: true });
    g.fillStyle = "#000"; g.fillRect(0, 0, size, size);
    g.imageSmoothingQuality = "high";
    g.drawImage(canvas, 0, 0, nw, nh);
    const px = g.getImageData(0, 0, size, size).data;
    const n = size * size;
    const input = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      input[i] = (px[i * 4] - 127.5) / 128;
      input[n + i] = (px[i * 4 + 1] - 127.5) / 128;
      input[2 * n + i] = (px[i * 4 + 2] - 127.5) / 128;
    }
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
    return dets.map((d) => {
      const face = { bbox: d.box, kps: d.kps, score: d.score };
      face.quality = assess(canvas, face);
      return face;
    });
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

  function align(canvas, kps, size) {
    const k = size / 112;
    const m = similarity(kps, ARCFACE_DST.map(([x, y]) => [x * k, y * k]));
    const out = makeCanvas(size, size);
    const g = ctx2d(out, { willReadFrequently: true });
    g.fillStyle = "#000"; g.fillRect(0, 0, size, size);
    g.imageSmoothingQuality = "high";
    g.setTransform(m.A, m.B, -m.B, m.A, m.tx, m.ty);
    g.drawImage(canvas, 0, 0);
    g.setTransform(1, 0, 0, 1, 0, 0);
    return out;
  }

  async function embed(canvas, face) {
    const crop = align(canvas, face.kps, 112);
    const px = ctx2d(crop).getImageData(0, 0, 112, 112).data;
    const n = 112 * 112;
    const input = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      input[i] = (px[i * 4] - 127.5) / 127.5;
      input[n + i] = (px[i * 4 + 1] - 127.5) / 127.5;
      input[2 * n + i] = (px[i * 4 + 2] - 127.5) / 127.5;
    }
    const outs = await Engine.run(Engine.rec, { [Engine.rec.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, 112, 112]) });
    const v = Float32Array.from(outs[Engine.rec.outputNames[0]].data);
    let norm = 0;
    for (const x of v) norm += x * x;
    norm = Math.sqrt(norm);
    return v.map((x) => x / norm);
  }

  const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  // ============================================================ คุณภาพภาพ
  function pose(kps) {
    const [le, re, nose, lm, rm] = kps;
    const ev = [re[0] - le[0], re[1] - le[1]];
    const ed = Math.hypot(ev[0], ev[1]) + 1e-6;
    const roll = (Math.atan2(ev[1], ev[0]) * 180) / Math.PI;
    const mid = [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2];
    const axis = [ev[0] / ed, ev[1] / ed];
    const yawRatio = ((nose[0] - mid[0]) * axis[0] + (nose[1] - mid[1]) * axis[1]) / ed;
    const yaw = (Math.atan(yawRatio / 0.55) * 180) / Math.PI;
    const up = [-axis[1], axis[0]];
    const mm = [(lm[0] + rm[0]) / 2, (lm[1] + rm[1]) / 2];
    const eyeNose = (nose[0] - mid[0]) * up[0] + (nose[1] - mid[1]) * up[1];
    const noseMouth = (mm[0] - nose[0]) * up[0] + (mm[1] - nose[1]) * up[1];
    return { yaw, roll, pitchRatio: eyeNose / (eyeNose + noseMouth + 1e-6) };
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
    const warnings = [];
    let penalty = 0;
    const small = Math.min(fw, fh);
    if (small < CFG.minFacePx) { warnings.push(`ใบหน้าเล็กมาก (~${Math.round(small)} px) ความละเอียดไม่พอ`); penalty += 2; }
    else if (small < 80) { warnings.push(`ใบหน้าค่อนข้างเล็ก (~${Math.round(small)} px)`); penalty += 1; }
    if (face.score < 0.65) { warnings.push("ความมั่นใจในการตรวจจับใบหน้าต่ำ"); penalty += 1; }
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
      yaw: p.yaw, roll: p.roll, pitch_ratio: p.pitchRatio, warnings, grade: penalty === 0 ? "good" : penalty <= 2 ? "fair" : "poor",
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
    if (sameImage) notes.unshift("เลือกภาพและใบหน้าเดียวกันทั้งสองฝั่ง คะแนนจึงสูงสุดโดยธรรมชาติ ไม่มีความหมายเชิงการพิสูจน์");
    return { grade: worst, text, notes };
  }

  function cropRegion(aligned, key) {
    const [x1, y1, x2, y2] = REGIONS[key][1];
    const w = aligned.width, h = aligned.height;
    const sx = Math.floor(x1 * w), sy = Math.floor(y1 * h), sw = Math.floor(x2 * w) - sx, sh = Math.floor(y2 * h) - sy;
    const c = makeCanvas(sw, sh);
    ctx2d(c).drawImage(aligned, sx, sy, sw, sh, 0, 0, sw, sh);
    return c;
  }

  // ============================================================ UI: การอัปโหลด
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

  const emptyHtml = (slot) => `<div class="empty"><p>แตะเพื่อเลือกภาพ หรือลากภาพมาวางที่นี่</p><div class="btnrow">
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
      if (!Engine.det) setStatus(slot, "กำลังโหลดโมเดลครั้งแรก (~16 MB) …");
      else setStatus(slot, "กำลังตรวจจับใบหน้า…");
      const [canvas] = await Promise.all([decodeFile(file), Engine.load()]);
      setStatus(slot, "กำลังตรวจจับใบหน้า…");
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
      svg.append(svgEl("rect", { class: "face" + (sel ? " sel" : ""), x: x1, y: y1, width: x2 - x1, height: y2 - y1, rx: 4 }));
      if (entry.faces.length > 1) {
        const t = svgEl("text", { x: x1 + 6, y: y1 + M * 0.035, "font-size": M * 0.032 });
        t.textContent = String(i + 1);
        const hit = svgEl("rect", { class: "hit", x: x1, y: y1, width: x2 - x1, height: y2 - y1 });
        hit.addEventListener("click", () => selectFace(slot, i));
        svg.append(t, hit);
      }
      if (sel) f.kps.forEach(([lx, ly]) => svg.append(svgEl("circle", { cx: lx, cy: ly, r: Math.max(2.5, M * 0.004) })));
    });
    wrap.append(svg);
    stage.append(wrap);
    const n = entry.faces.length;
    if (n === 0) setStatus(slot, "ไม่พบใบหน้าในภาพนี้ ลองภาพที่เห็นหน้าชัดและตรงมากขึ้น", true);
    else if (n === 1) setStatus(slot, `${entry.filename} · พบ 1 ใบหน้า`);
    else setStatus(slot, `${entry.filename} · พบ ${n} ใบหน้า — แตะกรอบเพื่อเลือกใบหน้าที่ต้องการ (กำลังเลือกลำดับ ${face + 1})`);
    renderChips(slot);
  }

  function renderChips(slot) {
    const box = $(`#chips-${slot}`);
    box.innerHTML = "";
    const st = state[slot];
    if (!st || !st.entry.faces.length) return;
    const q = st.entry.faces[st.face].quality;
    const add = (text, warn) => { const c = document.createElement("span"); c.className = "chip" + (warn ? " warn" : ""); c.textContent = text; box.append(c); };
    add(`คุณภาพ: ${GRADE_TH[q.grade]}`, q.grade !== "good");
    q.warnings.forEach((w) => add(w, true));
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
    btn.textContent = "กำลังเปรียบเทียบ…";
    refreshButtons();
    try {
      const a = state.A, b = state.B;
      const fa = a.entry.faces[a.face], fb = b.entry.faces[b.face];
      const [ea, eb] = [await embed(a.entry.canvas, fa), await embed(b.entry.canvas, fb)];
      const c = cosine(ea, eb);
      const score = Math.round(c * 1000) / 10;
      const sameImage = a.entry.sha256 === b.entry.sha256 && a.face === b.face;
      const alignedA = align(a.entry.canvas, fa.kps, CFG.alignSize), alignedB = align(b.entry.canvas, fb.kps, CFG.alignSize);
      state.last = { score, cosine: c, band: bandFor(score), rel: reliability(fa.quality, fb.quality, sameImage), alignedA, alignedB, fa, fb, a, b, sameImage };
      renderResult(state.last);
    } catch (e) {
      showError(e.message || "เปรียบเทียบไม่สำเร็จ");
    } finally {
      state.comparing = false;
      btn.textContent = "เปรียบเทียบ";
      refreshButtons();
    }
  });

  function renderResult(r) {
    $("#result").hidden = false;
    clearReport();
    $("#score").textContent = r.score.toFixed(1);
    $("#score").className = "score lv-" + r.band.level;
    $("#band").textContent = "ระดับ: " + r.band.label;
    $("#band").className = "band lv-" + r.band.level;
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

    const regions = $("#regions");
    regions.textContent = "";
    Object.keys(REGIONS).forEach((k) => {
      const row = document.createElement("div");
      row.className = "region";
      [["A", r.alignedA], ["B", r.alignedB]].forEach(([who, al]) => {
        const fig = document.createElement("figure");
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
    renderQuality(r);
    $("#model-name").textContent = "โมเดล: " + CFG.modelName;
    setMode("side");
    $("#result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
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
    ["เกรดคุณภาพ", (q) => GRADE_TH[q.grade]],
  ];
  function renderQuality(r) {
    const body = $("#quality-body");
    body.textContent = "";
    QROWS.forEach(([label, fn]) => {
      const tr = document.createElement("tr");
      const t0 = document.createElement("td"), ta = document.createElement("td"), tb = document.createElement("td");
      t0.textContent = label; ta.textContent = fn(r.fa.quality); tb.textContent = fn(r.fb.quality);
      if (label === "เกรดคุณภาพ") { if (r.fa.quality.grade === "poor") ta.className = "bad"; if (r.fb.quality.grade === "poor") tb.className = "bad"; }
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

  // ============================================================ รายงาน PDF (วาดเป็นหน้า A4 แล้วรวมเป็น PDF)
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const LEVEL_HEX = { high: "#1e1b6e", "mid-high": "#2f3fa8", mid: "#3f66c4", low: "#6b8cc7", "very-low": "#64748b" };
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
      g.strokeStyle = i === chosen ? "#3ca01e" : "#aaaaaa";
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
    blocks.push(`<div class="rp-title">รายงานการเปรียบเทียบใบหน้า</div><div class="rp-sub">Face Comparison Report — รายงานผลวิเคราะห์เชิงเทคนิคเบื้องต้น</div>
      <table class="rp-kv"><tr><th>เลขที่เรื่อง / อ้างอิง</th><td>${cv("case_no")}</td></tr><tr><th>วัตถุประสงค์</th><td>${cv("purpose")}</td></tr>
      <tr><th>ผู้จัดทำ / ผู้ตรวจ</th><td>${cv("examiner")}</td></tr><tr><th>วันที่และเวลาที่ออกรายงาน</th><td>${esc(thaiDate())}</td></tr></table>`);
    blocks.push(`<div class="rp-h">1. ผลการเปรียบเทียบ</div>
      <div class="rp-score"><div><div class="rp-num" style="color:${col}">${r.score.toFixed(1)}</div><div class="rp-small">คะแนนความคล้าย (0–100)</div></div>
      <div><div class="rp-band" style="color:${col}">ระดับ: ${esc(r.band.label)}</div><div>${esc(r.band.text)}</div><div class="rp-small">ค่า cosine similarity = ${r.cosine.toFixed(4)}</div></div></div>
      <div class="rp-gauge"><div class="rp-zones"><i style="flex:20;background:${LEVEL_HEX["very-low"]}"></i><i style="flex:10;background:${LEVEL_HEX.low}"></i><i style="flex:10;background:${LEVEL_HEX.mid}"></i><i style="flex:15;background:${LEVEL_HEX["mid-high"]}"></i><i style="flex:45;background:${LEVEL_HEX.high}"></i></div>
      <div class="rp-mark" style="left:${pos}%"></div>
      <div class="rp-ticks"><span style="left:0%">0</span><span style="left:20%">20</span><span style="left:30%">30</span><span style="left:40%">40</span><span style="left:55%">55</span><span style="left:100%">100</span></div></div>
      <div class="rp-small">ช่วงสี: ต่ำมาก &lt;20 · ต่ำ 20–30 · ก้ำกึ่ง 30–40 · สูง 40–55 · สูงมาก ≥55 (เกณฑ์เชิงแนวทางทั่วไป ไม่ได้สอบเทียบกับข้อมูลของงานนี้ สีใช้แสดงระดับเท่านั้น ไม่ได้บอกว่าผ่านหรือไม่ผ่าน)</div>`);
    blocks.push(`<div class="rp-b"><b>คุณภาพภาพโดยรวม: ${GRADE_TH[r.rel.grade]}</b> — ${esc(r.rel.text)}${r.rel.notes.length ? "<ul>" + r.rel.notes.map((n) => `<li>${esc(n)}</li>`).join("") + "</ul>" : ""}</div>
      <div class="rp-h">ข้อควรระวังสำคัญ</div><div class="rp-warn">${esc(DISCLAIMER)}</div>`);
    const src = (label, p) => `<div class="rp-src"><img src="${annotated(p.entry, p.face)}"><div class="rp-cell">${label} · ${esc(p.entry.filename)}</div>
      <div class="rp-small">ใบหน้าที่เลือก: ลำดับ ${p.face + 1} จากที่ตรวจพบ ${p.entry.faces.length} ใบหน้า<br>ขนาดไฟล์ ${Math.round(p.entry.size / 1024)} KB · ${p.entry.canvas.width}×${p.entry.canvas.height} px<br>SHA-256: <span class="rp-hash">${esc(p.entry.sha256)}</span></div></div>`;
    blocks.push(`<div class="rp-h">2. ภาพที่นำมาเปรียบเทียบ</div><div class="rp-two">${src("ภาพ A", r.a)}${src("ภาพ B", r.b)}</div>
      <div class="rp-small">กรอบสีเขียว = ใบหน้าที่ใช้เปรียบเทียบ · กรอบสีเทา = ใบหน้าอื่นในภาพ · จุดสีส้ม = จุดอ้างอิง 5 จุด (ตา จมูก มุมปาก)</div>`);
    const ua = r.alignedA.toDataURL("image/png"), ub = r.alignedB.toDataURL("image/png");
    blocks.push(`<div class="rp-h">3. เปรียบเทียบใบหน้าเคียงข้างกัน (จัดตำแหน่งอัตโนมัติ)</div><div class="rp-two rp-faces"><div><img src="${ua}"><div class="rp-cap">ภาพ A</div></div><div><img src="${ub}"><div class="rp-cap">ภาพ B</div></div></div>`);
    Object.keys(REGIONS).forEach((k, i) => blocks.push((i === 0 ? `<div class="rp-small">ส่วนประกอบของใบหน้า (ใช้ประกอบการพิจารณาด้วยสายตา โมเดลไม่ได้ให้คะแนนแยกรายส่วน)</div>` : "") +
      `<div class="rp-two rp-reg"><div><img src="${cropRegion(r.alignedA, k).toDataURL("image/png")}"><div class="rp-cap">A · ${REGIONS[k][0]}</div></div><div><img src="${cropRegion(r.alignedB, k).toDataURL("image/png")}"><div class="rp-cap">B · ${REGIONS[k][0]}</div></div></div>`));
    blocks.push(`<div class="rp-h">4. คุณภาพภาพและท่าทางใบหน้า</div><table class="rp-q"><tr><th>รายการ</th><th>ภาพ A</th><th>ภาพ B</th></tr>${QROWS.map(([l, f]) => `<tr><td>${l}</td><td>${esc(f(r.fa.quality))}</td><td>${esc(f(r.fb.quality))}</td></tr>`).join("")}</table>`);
    const method = [
      `โมเดล: ${CFG.modelName}`,
      "ขั้นตอน: (1) ตรวจจับใบหน้าและจุดอ้างอิง 5 จุด (2) จัดตำแหน่งใบหน้าให้ตรงแม่แบบ 112×112 พิกเซล (3) สกัดเวกเตอร์ลักษณะเฉพาะ 512 มิติ (4) คำนวณ cosine similarity ระหว่างเวกเตอร์ของสองภาพ",
      "คะแนนความคล้าย = cosine similarity × 100 (ค่ายิ่งสูงยิ่งคล้าย; คนละคนมักได้ใกล้ 0 หรือติดลบเล็กน้อย) ไม่ใช่เปอร์เซ็นต์ความน่าจะเป็น",
      "ภาพถูกหมุนตามข้อมูล EXIF และอาจถูกย่อเพื่อประมวลผล; ค่า SHA-256 คำนวณจากไฟล์ที่เลือกต้นฉบับ",
      "การประมวลผลทั้งหมดเกิดในเบราว์เซอร์ของผู้ใช้ ภาพไม่ถูกส่งออกจากอุปกรณ์",
    ];
    blocks.push(`<div class="rp-h">5. วิธีการ</div><ul class="rp-b">${method.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>`);
    if ((info.notes || "").trim()) blocks.push(`<div class="rp-h">6. หมายเหตุของผู้จัดทำ</div><div class="rp-b" style="white-space:pre-wrap">${esc(info.notes.trim())}</div>`);
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
      el.innerHTML = `<div class="rp-body" style="padding:${MARGIN}px ${MARGIN}px 0">${pages[p].map((b) => `<div class="rp-block">${b}</div>`).join("")}</div>
        <div class="rp-foot"><span>คะแนนความคล้ายไม่ใช่การยืนยันตัวบุคคล — ต้องมีการตรวจสอบโดยผู้เชี่ยวชาญและใช้หลักฐานอื่นประกอบ</span><span>หน้า ${p + 1}/${pages.length}</span></div>`;
      stage.append(el);
      await Promise.all([...el.querySelectorAll("img")].map((im) => im.decode().catch(() => {})));
      canvases.push(await html2canvas(el, { scale: 2, backgroundColor: "#ffffff", logging: false, useCORS: true, width: PAGE_W, height: PAGE_H, windowWidth: PAGE_W, scrollX: 0, scrollY: 0, x: 0, y: 0 }));
      el.remove();
    }
    return canvases;
  }

  function clearReport() {
    state.pages = null;
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
      try { await Promise.all([document.fonts.load("400 14px Sarabun"), document.fonts.load("700 14px Sarabun")]); } catch { /* ใช้ฟอนต์สำรอง */ }
      const info = Object.fromEntries(new FormData(ev.target).entries());
      const pages = await renderReportPages(state.last, info);
      state.pages = pages;
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
  // โหลดโมเดลล่วงหน้าเงียบ ๆ เพื่อให้ภาพแรกเร็วขึ้น (ไม่แสดงข้อผิดพลาดจนกว่าผู้ใช้จะเลือกภาพ)
  setTimeout(() => Engine.load().catch(() => {}), 300);
  window.__faceCompare = { Engine, state, detect, embed, cosine, decodeFile };
})();
