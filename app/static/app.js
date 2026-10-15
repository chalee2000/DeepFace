"use strict";
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const SLOTS = ["A", "B"];
  const state = { A: null, B: null, busy: { A: false, B: false }, comparing: false, pdfUrl: null };
  // state[slot] = { info, face }

  const GRADE_TH = { good: "ดี", fair: "พอใช้", poor: "ต่ำ" };

  // ---------- helpers
  async function api(path, opts) {
    let res;
    try {
      res = await fetch(path, opts);
    } catch {
      throw new Error("เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ ตรวจสอบเครือข่ายแล้วลองใหม่");
    }
    if (!res.ok) {
      let msg = `เกิดข้อผิดพลาด (${res.status})`;
      try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : msg; } catch { /* ไม่ใช่ JSON */ }
      const err = new Error(msg); err.status = res.status; throw err;
    }
    return res;
  }

  function showError(msg) {
    const el = $("#error");
    el.textContent = msg || "";
    el.hidden = !msg;
  }

  function setStatus(slot, msg, bad = false) {
    const el = $(`#status-${slot}`);
    el.textContent = msg || "";
    el.classList.toggle("bad", !!bad);
  }

  function refreshButtons() {
    const ready = SLOTS.every((s) => state[s] && state[s].info.faces.length > 0);
    $("#go").disabled = !ready || state.comparing;
    $("#swap").disabled = !(state.A && state.B);
    $("#clear").disabled = !(state.A || state.B);
    SLOTS.forEach((s) => ($(`#reset-${s}`).hidden = !state[s]));
  }

  // ---------- upload + preview
  async function upload(slot, file) {
    if (!file) return;
    showError("");
    await discard(slot);
    setStatus(slot, "กำลังตรวจจับใบหน้า…");
    const stage = $(`#stage-${slot}`);
    stage.innerHTML = '<div class="photo"><div class="spinner"></div></div>';
    stage.querySelector(".photo").style.minHeight = "200px";
    state.busy[slot] = true;
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      const info = await (await api("/api/upload", { method: "POST", body: fd })).json();
      state[slot] = { info, face: 0 };
      renderSlot(slot);
    } catch (e) {
      stage.innerHTML = emptyHtml(slot);
      bindPickers(stage);
      setStatus(slot, e.message, true);
    } finally {
      state.busy[slot] = false;
      refreshButtons();
    }
  }

  function emptyHtml(slot) {
    return `<div class="empty"><p>ลากภาพมาวางที่นี่ หรือเลือกจากเครื่อง</p><div class="btnrow">
      <label class="btn primary">เลือกรูป<input type="file" accept="image/*" data-pick="${slot}" hidden></label>
      <label class="btn">ถ่ายรูป<input type="file" accept="image/*" capture="user" data-pick="${slot}" hidden></label></div></div>`;
  }

  function svgEl(name, attrs) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  function renderSlot(slot) {
    const { info, face } = state[slot];
    const stage = $(`#stage-${slot}`);
    stage.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "photo";
    const img = document.createElement("img");
    img.alt = `ภาพบุคคล ${slot}`;
    img.src = `/api/image/${info.id}`;
    wrap.append(img);

    const W = info.width, H = info.height;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
    info.faces.forEach((f, i) => {
      const [x1, y1, x2, y2] = f.bbox;
      const sel = i === face;
      const r = svgEl("rect", {
        class: "face" + (sel ? " sel" : ""), x: x1 * W, y: y1 * H, width: (x2 - x1) * W, height: (y2 - y1) * H, rx: 4,
      });
      svg.append(r);
      if (info.faces.length > 1) {
        const hit = svgEl("rect", { class: "hit", x: x1 * W, y: y1 * H, width: (x2 - x1) * W, height: (y2 - y1) * H });
        hit.addEventListener("click", () => selectFace(slot, i));
        const t = svgEl("text", { x: x1 * W + 6, y: y1 * H + Math.max(W, H) * 0.035, "font-size": Math.max(W, H) * 0.032 });
        t.textContent = String(i + 1);
        svg.append(t, hit);
      }
      if (sel) f.landmarks.forEach(([lx, ly]) => svg.append(svgEl("circle", { cx: lx * W, cy: ly * H, r: Math.max(2.5, Math.max(W, H) * 0.004) })));
    });
    wrap.append(svg);
    stage.append(wrap);

    const n = info.faces.length;
    if (n === 0) setStatus(slot, "ไม่พบใบหน้าในภาพนี้ ลองภาพที่เห็นหน้าชัดและตรงมากขึ้น", true);
    else if (n === 1) setStatus(slot, `${info.filename} · พบ 1 ใบหน้า`);
    else setStatus(slot, `${info.filename} · พบ ${n} ใบหน้า — แตะกรอบเพื่อเลือกใบหน้าที่ต้องการ (กำลังเลือกลำดับ ${face + 1})`);
    renderChips(slot);
  }

  function renderChips(slot) {
    const box = $(`#chips-${slot}`);
    box.innerHTML = "";
    const st = state[slot];
    if (!st || !st.info.faces.length) return;
    const q = st.info.faces[st.face].quality;
    const add = (text, warn) => {
      const c = document.createElement("span");
      c.className = "chip" + (warn ? " warn" : "");
      c.textContent = text;
      box.append(c);
    };
    add(`คุณภาพ: ${GRADE_TH[q.grade]}`, q.grade !== "good");
    q.warnings.forEach((w) => add(w, true));
  }

  function selectFace(slot, i) {
    state[slot].face = i;
    hideResult();
    renderSlot(slot);
  }

  async function discard(slot) {
    const st = state[slot];
    state[slot] = null;
    hideResult();
    if (st) fetch(`/api/image/${st.info.id}`, { method: "DELETE" }).catch(() => {});
  }

  async function resetSlot(slot) {
    await discard(slot);
    const stage = $(`#stage-${slot}`);
    stage.innerHTML = emptyHtml(slot);
    bindPickers(stage);
    setStatus(slot, "");
    $(`#chips-${slot}`).innerHTML = "";
    refreshButtons();
  }

  // ---------- pickers + drag/drop
  function bindPickers(root) {
    root.querySelectorAll("input[data-pick]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const f = inp.files && inp.files[0];
        inp.value = "";
        upload(inp.dataset.pick, f);
      });
    });
  }
  bindPickers(document);

  SLOTS.forEach((slot) => {
    const stage = $(`#stage-${slot}`);
    ["dragenter", "dragover"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.remove("drag"); }));
    stage.addEventListener("drop", (e) => {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) upload(slot, f);
    });
    $(`#reset-${slot}`).addEventListener("click", () => resetSlot(slot));
  });
  // กันเบราว์เซอร์เปิดไฟล์ทับหน้าเมื่อวางนอกกรอบ
  ["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => e.preventDefault()));

  $("#clear").addEventListener("click", async () => { await Promise.all(SLOTS.map(resetSlot)); showError(""); });
  $("#swap").addEventListener("click", () => {
    [state.A, state.B] = [state.B, state.A];
    hideResult();
    SLOTS.forEach((s) => (state[s] ? renderSlot(s) : null));
    refreshButtons();
  });

  // ---------- compare
  const pickBody = () => ({ a: { id: state.A.info.id, face: state.A.face }, b: { id: state.B.info.id, face: state.B.face } });

  function hideResult() {
    $("#result").hidden = true;
    clearPdf();
  }

  function handleExpired(e) {
    if (e.status === 410) {
      showError("ภาพหมดอายุจากหน่วยความจำของเซิร์ฟเวอร์ กรุณาอัปโหลดภาพใหม่อีกครั้ง");
      SLOTS.forEach((s) => (state[s] = null));
      SLOTS.forEach((s) => { $(`#stage-${s}`).innerHTML = emptyHtml(s); bindPickers($(`#stage-${s}`)); setStatus(s, ""); $(`#chips-${s}`).innerHTML = ""; });
      refreshButtons();
      return true;
    }
    return false;
  }

  $("#go").addEventListener("click", async () => {
    showError("");
    state.comparing = true;
    const btn = $("#go");
    btn.textContent = "กำลังเปรียบเทียบ…";
    refreshButtons();
    try {
      const res = await (await api("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(pickBody()) })).json();
      renderResult(res);
    } catch (e) {
      if (!handleExpired(e)) showError(e.message);
    } finally {
      state.comparing = false;
      btn.textContent = "เปรียบเทียบ";
      refreshButtons();
    }
  });

  function renderResult(r) {
    $("#result").hidden = false;
    clearPdf();
    $("#score").textContent = r.score.toFixed(1);
    $("#score").className = "score lv-" + r.band.level;
    $("#band").textContent = "ระดับ: " + r.band.label;
    $("#band").className = "band lv-" + r.band.level;
    $("#band-text").textContent = r.band.text;
    $("#cosine").textContent = `cosine similarity = ${r.cosine.toFixed(4)}`;
    $("#marker").style.left = Math.max(0, Math.min(100, r.score)) + "%";
    $("#disclaimer").textContent = r.disclaimer;

    const rel = $("#reliability");
    rel.className = "reliability " + r.reliability.grade;
    rel.textContent = "";
    const b = document.createElement("strong");
    b.textContent = `คุณภาพภาพโดยรวม: ${GRADE_TH[r.reliability.grade]}`;
    rel.append(b, " — " + r.reliability.text);
    if (r.reliability.notes.length) {
      const ul = document.createElement("ul");
      r.reliability.notes.forEach((n) => { const li = document.createElement("li"); li.textContent = n; ul.append(li); });
      rel.append(ul);
    }

    for (const id of ["aligned-A", "wipe-A", "blend-A"]) $("#" + id).src = r.a.aligned;
    for (const id of ["aligned-B", "wipe-B", "blend-B"]) $("#" + id).src = r.b.aligned;

    const regions = $("#regions");
    regions.textContent = "";
    Object.keys(r.a.regions).forEach((k) => {
      const row = document.createElement("div");
      row.className = "region";
      [["A", r.a.regions[k]], ["B", r.b.regions[k]]].forEach(([who, reg]) => {
        const fig = document.createElement("figure");
        const im = document.createElement("img");
        im.src = reg.image; im.alt = `${reg.label} ${who}`;
        const cap = document.createElement("figcaption");
        cap.textContent = `${who} · ${reg.label}`;
        fig.append(im, cap);
        row.append(fig);
      });
      regions.append(row);
    });

    renderQuality(r);
    $("#model-name").textContent = "โมเดล: " + r.model;
    setMode("side");
    $("#result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  const signed = (v) => { const n = Math.round(v); return n === 0 ? "0°" : `${n > 0 ? "+" : ""}${n}°`; };

  function renderQuality(r) {
    const rows = [
      ["ความมั่นใจการตรวจจับ", (q) => q.det_score.toFixed(2)],
      ["ขนาดใบหน้า (px)", (q) => `${q.face_px[0]}×${q.face_px[1]}`],
      ["ความคมชัด", (q) => Math.round(q.sharpness)],
      ["ความสว่างเฉลี่ย", (q) => Math.round(q.brightness)],
      ["คอนทราสต์", (q) => Math.round(q.contrast)],
      ["มุมหันหน้า (ประมาณ)", (q) => signed(q.yaw)],
      ["มุมเอียงศีรษะ (ประมาณ)", (q) => signed(q.roll)],
      ["เกรดคุณภาพ", (q) => GRADE_TH[q.grade]],
    ];
    const body = $("#quality-body");
    body.textContent = "";
    rows.forEach(([label, fn]) => {
      const tr = document.createElement("tr");
      const th = document.createElement("td"); th.textContent = label;
      const a = document.createElement("td"); a.textContent = fn(r.a.quality);
      const b = document.createElement("td"); b.textContent = fn(r.b.quality);
      if (label === "เกรดคุณภาพ") { if (r.a.quality.grade === "poor") a.className = "bad"; if (r.b.quality.grade === "poor") b.className = "bad"; }
      tr.append(th, a, b);
      body.append(tr);
    });
  }

  // ---------- view modes
  function setMode(mode) {
    document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
    $("#view-side").hidden = mode !== "side";
    $("#view-wipe").hidden = mode !== "wipe";
    $("#view-blend").hidden = mode !== "blend";
  }
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

  const wipe = $("#wipe-range");
  const applyWipe = () => {
    const v = wipe.value;
    $("#wipe-A").style.clipPath = `inset(0 ${100 - v}% 0 0)`;
    $("#wipe-handle").style.left = v + "%";
  };
  wipe.addEventListener("input", applyWipe);
  applyWipe();
  const blend = $("#blend-range");
  blend.addEventListener("input", () => ($("#blend-A").style.opacity = blend.value / 100));

  // ---------- PDF report
  function clearPdf() {
    if (state.pdfUrl) URL.revokeObjectURL(state.pdfUrl);
    state.pdfUrl = null;
    const link = $("#report-link");
    link.hidden = true;
    link.removeAttribute("href");
    $("#report-status").textContent = "";
  }

  $("#report-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (!(state.A && state.B)) return;
    clearPdf();
    const btn = $("#report-btn");
    const status = $("#report-status");
    btn.disabled = true;
    status.className = "status";
    status.textContent = "กำลังสร้างรายงาน…";
    const fd = new FormData(ev.target);
    const body = { ...pickBody(), case: Object.fromEntries(fd.entries()) };
    try {
      const blob = await (await api("/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).blob();
      state.pdfUrl = URL.createObjectURL(blob);
      const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
      const link = $("#report-link");
      link.href = state.pdfUrl;
      link.download = `face-comparison-${stamp}.pdf`;
      link.hidden = false;
      status.textContent = "สร้างรายงานเสร็จแล้ว แตะ “เปิด / บันทึก PDF”";
      // เดสก์ท็อปดาวน์โหลดอัตโนมัติ; บน iOS ให้ผู้ใช้แตะลิงก์เอง (ป้องกันป๊อปอัปถูกบล็อก)
      if (!/iPad|iPhone|iPod/.test(navigator.userAgent)) link.click();
    } catch (e) {
      if (!handleExpired(e)) { status.className = "status bad"; status.textContent = e.message; }
    } finally {
      btn.disabled = false;
    }
  });

  refreshButtons();
})();
