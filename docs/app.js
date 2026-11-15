/*
 * AgentKV visualizer.
 * Every simulator frame comes from docs/data.js, which benchmarks/export_visualizer.py
 * generates by running the real Python engine. Nothing here is a scripted animation.
 */
(function () {
  "use strict";

  const D = window.AGENTKV_DATA;
  const R = D.results;

  const AGENTS = ["Planner", "Retriever", "Coder", "Reviewer"];
  const AGENT_COLOR = {
    System: "#8a8272", Planner: "#2f5d8c", Retriever: "#7a8a2e",
    Coder: "#c58a17", Reviewer: "#a3472d", Tail: "#b8af9c",
  };
  const SHORT = { System: "S", Planner: "P", Retriever: "R", Coder: "C", Reviewer: "V", Tail: "t" };
  const POLICY_COLOR = {
    "LRU": "#cf4a2e", "LRU-TailFirst": "#b07d2b", "AgentKV": "#2d6a4f",
    "AgentKV-Markov": "#1f8a8a", "Belady": "#1d1a14",
    "LRU + CPU tier": "#cf4a2e", "AgentKV + prefetch": "#2d6a4f", "AgentKV-Markov + prefetch": "#1f8a8a",
  };
  const POLICY_RULE = {
    "LRU": "evicts whatever was used longest ago",
    "AgentKV": "evicts the agent that runs furthest away (BFS)",
    "AgentKV-Markov": "same idea with expected steps, learned on the fly",
    "Belady": "cheats: it knows the future (reference only)",
    "LRU + CPU tier": "offloads to CPU, copies back when needed",
    "AgentKV + prefetch": "copies the next agent back before it runs (BFS)",
    "AgentKV-Markov + prefetch": "prefetches using learned probabilities",
  };
  const pretty = (p) => (p === "Belady" ? "Bélády" : p);
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ helpers
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c.nodeType ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  const SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, ...children) {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v !== null && v !== undefined) el.setAttribute(k, v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined) continue;
      el.append(c.nodeType ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  const $ = (sel) => document.querySelector(sel);
  const fmt = (x, d = 1) => (x === null || x === undefined ? "∞" : Number(x).toFixed(d));
  const pill = (name) => `<span class="pill" style="--c:${AGENT_COLOR[name] || "#888"}">${name}</span>`;
  const tooltip = $("#tooltip");
  function showTip(evt, html) {
    tooltip.innerHTML = html;
    tooltip.style.opacity = 1;
    const x = Math.min(evt.clientX + 16, window.innerWidth - 240);
    tooltip.style.left = x + "px";
    tooltip.style.top = evt.clientY + 16 + "px";
  }
  const hideTip = () => { tooltip.style.opacity = 0; };

  // repo links + reveal-on-scroll
  document.querySelectorAll(".repo-link").forEach((a) => { a.href = document.body.dataset.repo; });
  const revealer = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("visible"); revealer.unobserve(e.target); } });
  }, { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((el) => revealer.observe(el));

  function onFirstView(el, fn) {
    const obs = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) { obs.disconnect(); fn(); }
    }, { threshold: 0.35 });
    obs.observe(el);
  }

  // ------------------------------------------------------------------ agent ring (shared drawing)
  const RING_POS = { Planner: -90, Retriever: 0, Coder: 90, Reviewer: 180 };
  function ringXY(agent, cx, cy, r) {
    const a = (RING_POS[agent] * Math.PI) / 180;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  }
  function drawRing(svgEl, opts) {
    const size = opts.size || 300;
    const cx = size / 2, cy = size * 0.47, r = size * 0.33, nodeR = size * 0.108;
    svgEl.innerHTML = "";
    const defs = s("defs", {},
      s("marker", { id: svgEl.id + "-arrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
        s("path", { d: "M0,0 L10,5 L0,10 z", fill: "#9a917f" })),
      s("filter", { id: svgEl.id + "-glow", x: "-50%", y: "-50%", width: "200%", height: "200%" },
        s("feGaussianBlur", { stdDeviation: 6, result: "b" }),
        s("feMerge", {}, s("feMergeNode", { in: "b" }), s("feMergeNode", { in: "SourceGraphic" }))));
    svgEl.append(defs);

    for (const e of opts.edges) {
      const [x1, y1] = ringXY(e.from, cx, cy, r);
      const [x2, y2] = ringXY(e.to, cx, cy, r);
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
      const ux = dx / len, uy = dy / len;
      const sx = x1 + ux * (nodeR + 4), sy = y1 + uy * (nodeR + 4);
      const ex = x2 - ux * (nodeR + 8), ey = y2 - uy * (nodeR + 8);
      const bend = e.bend === undefined ? 22 : e.bend;
      const mx = (sx + ex) / 2 - uy * bend, my = (sy + ey) / 2 + ux * bend;
      svgEl.append(s("path", {
        d: `M${sx},${sy} Q${mx},${my} ${ex},${ey}`, fill: "none",
        stroke: e.color || "#bfb39a", "stroke-width": e.width || 2,
        "marker-end": `url(#${svgEl.id}-arrow)`, opacity: e.opacity || 1,
      }));
      if (e.label) {
        const off = bend + (bend < 0 ? -14 : 12);
        const lx = (sx + ex) / 2 - uy * off, ly = (sy + ey) / 2 + ux * off;
        svgEl.append(s("text", { x: lx, y: ly + 4, "text-anchor": "middle", fill: "#3d382e", "font-size": 12, "font-weight": 700, "font-family": "JetBrains Mono, monospace" }, e.label));
      }
    }

    for (const agent of AGENTS) {
      const [x, y] = ringXY(agent, cx, cy, r);
      const color = AGENT_COLOR[agent];
      const isNow = agent === opts.current;
      const g = s("g", {});
      if (isNow) {
        g.append(s("circle", { cx: x, cy: y, r: nodeR + 6, fill: "none", stroke: color, "stroke-width": 2, opacity: 0.6 },
          s("animate", { attributeName: "r", values: `${nodeR + 4};${nodeR + 14};${nodeR + 4}`, dur: "1.6s", repeatCount: "indefinite" }),
          s("animate", { attributeName: "opacity", values: "0.7;0;0.7", dur: "1.6s", repeatCount: "indefinite" })));
      }
      g.append(s("circle", {
        cx: x, cy: y, r: nodeR, fill: isNow ? color : "#fbf7ee", stroke: color, "stroke-width": 2.5,
        filter: isNow ? `url(#${svgEl.id}-glow)` : null,
      }));
      g.append(s("text", {
        x, y: y + 4, "text-anchor": "middle", "font-size": size * 0.034, "font-weight": 800,
        fill: isNow ? "#fbf7ee" : color, "font-family": "Inter, sans-serif",
      }, agent));
      const badge = opts.badges && opts.badges[agent];
      if (badge) {
        const by = y + nodeR + 16;
        const w = Math.max(38, badge.length * 7.2 + 14);
        g.append(s("rect", { x: x - w / 2, y: by - 12, width: w, height: 19, rx: 9.5, fill: "#efe8d8", stroke: "#c9bda5" }));
        g.append(s("text", { x, y: by + 2, "text-anchor": "middle", "font-size": 11, fill: "#1d1a14", "font-family": "JetBrains Mono, monospace", "font-weight": 600 }, badge));
      }
      svgEl.append(g);
    }
    if (opts.center) {
      svgEl.append(s("text", { x: cx, y: cy - 4, "text-anchor": "middle", "font-size": 12, fill: "#6a6355", "font-family": "Inter" }, opts.center[0]));
      svgEl.append(s("text", { x: cx, y: cy + 14, "text-anchor": "middle", "font-size": 12, fill: "#1d1a14", "font-weight": 700, "font-family": "Inter" }, opts.center[1]));
    }
  }
  const CYCLE_EDGES = [
    { from: "Planner", to: "Retriever" }, { from: "Retriever", to: "Coder" },
    { from: "Coder", to: "Reviewer" }, { from: "Reviewer", to: "Planner" },
  ];
  function branchEdges(p) {
    const edges = [
      { from: "Planner", to: "Retriever" }, { from: "Retriever", to: "Coder" },
      { from: "Coder", to: "Reviewer" },
      { from: "Reviewer", to: "Planner", label: Math.round((1 - p) * 100) + "%" },
    ];
    if (p > 0) edges.push({ from: "Reviewer", to: "Coder", label: Math.round(p * 100) + "%", bend: -26, color: "#c58a17", width: 1.5 + p * 3 });
    return edges;
  }

  // ------------------------------------------------------------------ hero
  function initHero() {
    const sweep = R.sweep;
    const cyc = sweep.find((r) => r.workflow === "cyclic" && r.capacity_blocks === 360);
    const br = sweep.find((r) => r.workflow === "branching" && r.capacity_blocks === 360);
    const cut = (100 * (br.LRU_recomputed_tokens - br.AgentKV_recomputed_tokens)) / br.LRU_recomputed_tokens;
    const tflops = ((br.LRU_recomputed_tokens - br.AgentKV_recomputed_tokens) * 16e9) / 1e12;
    const t = R.headline.tiered.cyclic_300;
    const seeds = R.headline.seeds_experiment;
    const stats = [
      { from: fmt(cyc.LRU_hit_rate) + "% LRU", value: cyc.AgentKV_hit_rate, dec: 1, suffix: "%", accent: "#2d6a4f",
        label: "block hit rate on a 4-agent loop (720 MiB), equal to the Bélády oracle" },
      { from: "recomputed prefill", plain: true, value: cut, dec: 0, prefix: "−", suffix: "%", accent: "#2f5d8c",
        label: `on a branching loop: ~${Math.round(tflops).toLocaleString()} TFLOPs of Llama-3-8B prefill saved per trace` },
      { from: fmt(t["LRU, GPU only"].stall_ms_per_request) + " ms LRU", value: t["AgentKV + CPU tier + prefetch"].stall_ms_per_request, dec: 0, suffix: " ms", accent: "#1f8a8a",
        label: "modeled stall per request with a CPU tier and workflow-aware prefetching" },
      { from: "random seeds", plain: true, value: seeds.agentkv_wins, dec: 0, suffix: "/" + seeds.seeds, accent: "#1d1a14",
        label: `where AgentKV beat LRU (${fmt(seeds.agentkv_mean)} ± ${fmt(seeds.agentkv_std)}% vs ${fmt(seeds.lru_mean)}%)` },
    ];
    const box = $("#hero-stats");
    for (const st of stats) {
      const val = h("div", { class: "value" }, (st.prefix || "") + "0" + st.suffix);
      box.append(h("div", { class: "stat", style: `--accent:${st.accent}` },
        h("div", { class: "from" + (st.plain ? " plain" : "") }, st.from), val, h("div", { class: "label" }, st.label)));
      st.el = val;
    }
    const finalValues = () => { for (const st of stats) st.el.textContent = (st.prefix || "") + st.value.toFixed(st.dec) + st.suffix; };
    if (REDUCED) finalValues();
    else onFirstView(box, () => {
      const start = performance.now();
      const tick = (now) => {
        const k = Math.min(1, (now - start) / 1300);
        const ease = 1 - Math.pow(1 - k, 3);
        for (const st of stats) st.el.textContent = (st.prefix || "") + (st.value * ease).toFixed(st.dec) + st.suffix;
        if (k < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });

    // animated ring: show the LRU thrashing story on a loop
    const ring = $("#hero-ring");
    const steps = D.eviction.cyclic.lanes[0].steps;
    let i = 0;
    const paint = () => {
      const st = steps[i];
      const cached = new Set(st.gpu.filter(Boolean));
      const badges = {};
      for (const a of AGENTS) badges[a] = cached.has(a) ? "cached" : "—";
      const verdict = st.hit === st.total ? "HIT" : i < 3 ? "first run" : "MISS (thrashing)";
      drawRing(ring, { size: 360, current: st.agent, edges: CYCLE_EDGES, badges, center: ["LRU · step " + (i + 1), verdict] });
      i = (i + 1) % 8;
    };
    paint();
    setInterval(paint, 1500);
  }

  // ------------------------------------------------------------------ chapter 1 story
  function initProblem() {
    const lanes = D.eviction.cyclic.lanes;
    const fill = (ol, lane) => {
      lane.steps.slice(0, 8).forEach((st, i) => {
        const ev = st.events.find((e) => e.type === "evict");
        const firstRun = !lane.steps.slice(0, i).some((p) => p.agent === st.agent);
        let res, cls;
        if (firstRun) { res = "first run"; cls = "cold"; }
        else if (st.hit === st.total) { res = "HIT"; cls = "hit"; }
        else { res = "MISS"; cls = "miss"; }
        ol.append(h("li", {},
          h("span", { class: "step" }, "step " + (i + 1)),
          h("span", { html: pill(st.agent) + (ev ? ` <span class="muted">→ evicts</span> ${pill(ev.victim)}` : "") }),
          h("span", { class: "res " + cls }, res)));
      });
    };
    fill($("#lru-story"), lanes.find((l) => l.policy === "LRU"));
    fill($("#akv-story"), lanes.find((l) => l.policy === "AgentKV"));
  }

  // ------------------------------------------------------------------ chapter 2: prompt strip + calculator
  function initKV() {
    const strip = $("#prompt-strip");
    const parts = [["System", 16], ["Planner", 96], ["Tail", 10]];
    for (const [name, n] of parts) {
      for (let k = 0; k < n; k++) strip.append(h("div", { class: "pblock", style: `--c:${AGENT_COLOR[name]}`, title: name }));
    }
    strip.append(h("div", { class: "pblock partial", title: "10 leftover tokens: not a full block, never cached" }));
    $("#prompt-legend").innerHTML =
      `<span>${pill("System")} <b>256 tokens</b> = 16 blocks, shared by every agent</span>` +
      `<span>${pill("Planner")} <b>1,536 tokens</b> = 96 blocks, the same every time Planner runs</span>` +
      `<span>${pill("Tail")} <b>170 tokens</b> of fresh text = 10 blocks + 10 leftover tokens (never cached)</span>`;

    const presets = {
      "Llama-3-8B": [32, 8, 128], "Llama-3-70B": [80, 8, 128],
      "Qwen2.5-7B": [28, 4, 128], "Llama-2-7B (no GQA)": [32, 32, 128],
    };
    const inputs = { layers: $("#c-layers"), heads: $("#c-heads"), dim: $("#c-dim"), tokens: $("#c-tokens") };
    const chips = $("#calc-presets");
    for (const [name, [l, hh, d]] of Object.entries(presets)) {
      chips.append(h("button", {
        class: "chip" + (name === "Llama-3-8B" ? " active" : ""),
        onclick: (e) => {
          inputs.layers.value = l; inputs.heads.value = hh; inputs.dim.value = d;
          chips.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
          e.target.classList.add("active");
          update();
        },
      }, name));
    }
    const size = (bytes) => bytes >= 1024 ** 3 ? (bytes / 1024 ** 3).toFixed(2) + " GiB"
      : bytes >= 1024 ** 2 ? (bytes / 1024 ** 2).toFixed(bytes >= 100 * 1024 ** 2 ? 0 : 1) + " MiB"
      : (bytes / 1024).toFixed(0) + " KiB";
    function update() {
      const L = +inputs.layers.value, H = +inputs.heads.value, Dm = +inputs.dim.value, T = +inputs.tokens.value;
      $("#o-layers").textContent = L; $("#o-heads").textContent = H; $("#o-dim").textContent = Dm; $("#o-tokens").textContent = T.toLocaleString();
      const perTok = 2 * L * H * Dm * 2;
      $("#calc-formula").innerHTML = `2 (K,V) × <span class="hl">${L}</span> layers × <span class="hl">${H}</span> KV heads × <span class="hl">${Dm}</span> dims × 2 bytes = <span class="hl">${perTok.toLocaleString()}</span> bytes`;
      $("#calc-token").textContent = size(perTok);
      $("#calc-block").textContent = size(perTok * 16);
      $("#calc-prompt").textContent = size(perTok * T);
      const fit = Math.floor((16 * 1024 ** 3) / (perTok * T));
      $("#calc-fit").innerHTML = `A 16 GiB KV budget holds about <b>${fit.toLocaleString()}</b> prompts like this at once. That's why the cache fills up fast, and why <b>what to evict</b> matters.`;
      $("#calc-note").textContent = H >= 32
        ? "No Grouped Query Attention: every attention head stores its own K and V, so the cache is huge."
        : "Grouped Query Attention (GQA): many query heads share a few KV heads, which shrinks the cache a lot.";
      chips.querySelectorAll(".chip").forEach((c) => {
        const p = presets[c.textContent];
        c.classList.toggle("active", p[0] === L && p[1] === H && p[2] === Dm);
      });
    }
    Object.values(inputs).forEach((inp) => inp.addEventListener("input", update));
    update();
  }

  // ------------------------------------------------------------------ chapter 3: interactive radix tree
  function initRadix() {
    const PROMPTS = {
      Planner: ["S1", "S2", "P1", "P2", "P3"],
      Coder: ["S1", "S2", "C1", "C2", "C3"],
      Reviewer: ["S1", "S2", "V1", "V2", "V3"],
    };
    const colorOf = (b) => b[0] === "S" ? AGENT_COLOR.System : b[0] === "P" ? AGENT_COLOR.Planner
      : b[0] === "C" ? AGENT_COLOR.Coder : b[0] === "V" ? AGENT_COLOR.Reviewer : AGENT_COLOR.Tail;
    let root, nextId, tailCount, saved;
    function reset() {
      root = { id: 0, blocks: [], children: new Map() };
      nextId = 1; tailCount = 0; saved = 0;
      $("#radix-saved").textContent = 0;
      $("#radix-msg").textContent = "The tree is empty. Send a prompt.";
      render({});
    }
    function send(agent) {
      tailCount += 1;
      const prompt = PROMPTS[agent].concat(["t" + tailCount]);
      let node = root, i = 0, hits = 0, split = null;
      const path = [];
      while (i < prompt.length) {
        let child = node.children.get(prompt[i]);
        if (!child) break;
        let k = 0;
        while (k < child.blocks.length && i + k < prompt.length && child.blocks[k] === prompt[i + k]) k++;
        if (k < child.blocks.length) {
          // the prompt diverges inside this node: split it
          const top = { id: nextId++, blocks: child.blocks.slice(0, k), children: new Map() };
          split = { before: child.blocks.join(" "), k };
          child.blocks = child.blocks.slice(k);
          top.children.set(child.blocks[0], child);
          node.children.set(top.blocks[0], top);
          child = top;
        }
        path.push(child.id);
        hits += k; i += k; node = child;
      }
      const rest = prompt.slice(i);
      let created = null;
      if (rest.length) {
        created = { id: nextId++, blocks: rest, children: new Map() };
        node.children.set(rest[0], created);
      }
      saved += hits;
      $("#radix-saved").textContent = saved;
      let msg = `${pill(agent)} prompt (${prompt.length} blocks): <b style="color:#2d6a4f">reused ${hits}</b>, computed ${rest.length}.`;
      if (split) msg += ` <span style="color:#c58a17">Split</span> the node [${split.before}] after ${split.k} block${split.k > 1 ? "s" : ""}, because the prompt diverged inside it.`;
      if (!hits) msg += " Nothing was cached yet, so the whole prompt was computed.";
      $("#radix-msg").innerHTML = msg;
      render({ hit: new Set(path), created: created && created.id, split: split ? path[path.length - 1] : null });
    }

    function render(mark) {
      const svgEl = $("#radix-svg");
      svgEl.innerHTML = "";
      // layout: leaves left to right, parents centered over children
      const levels = [];
      let leafX = 0;
      function layout(node, depth) {
        node.depth = depth;
        const kids = [...node.children.values()];
        if (!kids.length) { node.x = leafX++; }
        else { kids.forEach((k) => layout(k, depth + 1)); node.x = kids.reduce((a, k) => a + k.x, 0) / kids.length; }
        (levels[depth] = levels[depth] || []).push(node);
      }
      layout(root, 0);
      const colW = 200, rowH = 92;
      const width = Math.max(900, leafX * colW + 60);
      const height = Math.max(220, levels.length * rowH + 30);
      svgEl.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const off = (width - (leafX - 1) * colW) / 2;
      const X = (n) => off + n.x * colW;
      const Y = (n) => 30 + n.depth * rowH;

      const nodes = levels.flat();
      for (const n of nodes) {
        for (const c of n.children.values()) {
          svgEl.append(s("path", { class: "tedge", d: `M${X(n)},${Y(n) + 16} C${X(n)},${Y(n) + 50} ${X(c)},${Y(c) - 40} ${X(c)},${Y(c) - 16}` }));
        }
      }
      for (const n of nodes) {
        if (n === root) {
          svgEl.append(s("circle", { cx: X(n), cy: Y(n), r: 9, fill: "#efe8d8", stroke: "#9a917f" }));
          svgEl.append(s("text", { x: X(n) + 16, y: Y(n) + 4, fill: "#9a917f", "font-size": 12 }, "root"));
          continue;
        }
        const chip = 22, gap = 4;
        const w = n.blocks.length * (chip + gap) + 14;
        const cls = "tnode" + (mark.split === n.id ? " flash" : mark.created === n.id ? " new" : mark.hit && mark.hit.has(n.id) ? " hit" : "");
        const g = s("g", { class: cls });
        g.append(s("rect", { class: "tnode-rect", x: X(n) - w / 2, y: Y(n) - 16, width: w, height: 32, rx: 9 }));
        n.blocks.forEach((b, k) => {
          const bx = X(n) - w / 2 + 7 + k * (chip + gap);
          g.append(s("rect", { x: bx, y: Y(n) - 11, width: chip, height: 22, rx: 5, fill: colorOf(b) }));
          g.append(s("text", { x: bx + chip / 2, y: Y(n) + 4, "text-anchor": "middle", "font-size": 10, "font-weight": 800, fill: "#fbf7ee", "font-family": "JetBrains Mono" }, b));
        });
        svgEl.append(g);
      }
      if (!root.children.size) {
        svgEl.append(s("text", { x: width / 2, y: 120, "text-anchor": "middle", fill: "#9a917f", "font-size": 14 }, "(empty)"));
      }
    }

    const box = $("#radix-buttons");
    for (const agent of Object.keys(PROMPTS)) {
      box.append(h("button", { class: "send-btn", style: `--c:${AGENT_COLOR[agent]}`, onclick: () => send(agent) },
        h("i", {}), "Send " + agent + " prompt"));
    }
    box.append(h("button", { class: "send-btn", style: "--c:#9a917f", onclick: reset }, "⟲ Reset"));
    reset();
    // play a short demo the first time the tree scrolls into view: share, then split
    onFirstView($("#radix-svg"), () => {
      if (root.children.size) return;
      send("Planner");
      setTimeout(() => send("Coder"), 1800);
      setTimeout(() => send("Planner"), 3600);
    });
  }

  // ------------------------------------------------------------------ generic step player
  class Player {
    constructor(root, render) {
      this.root = root; this.render = render; this.i = 0; this.n = 1; this.timer = null; this.delay = 1400;
      this.playBtn = root.querySelector('[data-act="play"]');
      root.querySelector('[data-act="reset"]').onclick = () => { this.stop(); this.go(0); };
      root.querySelector('[data-act="prev"]').onclick = () => { this.stop(); this.go(this.i - 1); };
      root.querySelector('[data-act="next"]').onclick = () => { this.stop(); this.go(this.i + 1); };
      this.playBtn.onclick = () => (this.timer ? this.stop() : this.play());
      root.querySelector('[data-act="speed"]').onchange = (e) => {
        this.delay = +e.target.value;
        if (this.timer) { this.stop(); this.play(); }
      };
    }
    setLength(n) { this.n = n; this.go(0); }
    go(i) { this.i = Math.max(0, Math.min(this.n - 1, i)); this.render(this.i); }
    play() {
      if (this.i >= this.n - 1) this.go(0);
      this.playBtn.textContent = "❚❚";
      this.timer = setInterval(() => {
        if (this.i >= this.n - 1) { this.stop(); return; }
        this.go(this.i + 1);
      }, this.delay);
    }
    stop() { clearInterval(this.timer); this.timer = null; this.playBtn.textContent = "▶"; }
  }

  function segmented(container, options, onPick) {
    container.innerHTML = "";
    options.forEach(([value, label], k) => {
      container.append(h("button", {
        class: k === 0 ? "active" : "",
        onclick: (e) => {
          container.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
          e.target.classList.add("active");
          onPick(value);
        },
      }, label));
    });
  }

  function seqStrip(container, sequence, i, onPick) {
    container.innerHTML = "";
    sequence.forEach((agent, k) => {
      container.append(h("div", {
        class: "seq-item" + (k < i ? " done" : "") + (k === i ? " now" : ""),
        style: `--c:${AGENT_COLOR[agent]}`, title: `step ${k + 1}: ${agent}`, onclick: () => onPick(k),
      }, SHORT[agent] + (k + 1)));
    });
  }

  function memCells(cells, prev) {
    return cells.map((label, k) => h("div", {
      class: "cell" + (label ? " filled" : "") + (prev && prev[k] !== label && label ? " flash" : ""),
      style: label ? `--c:${AGENT_COLOR[label]}` : null, title: label ? `block ${k}: ${label}` : `block ${k}: free`,
    }, label ? SHORT[label] : ""));
  }

  // ------------------------------------------------------------------ chapter 4: live simulator
  function initSimulator() {
    let wf = "cyclic";
    const prev = {};
    const player = new Player($("#sim-player"), render);

    function narrate(i) {
      const data = D.eviction[wf];
      const seq = data.sequence;
      const lane = (p) => data.lanes.find((l) => l.policy === p).steps[i];
      const lru = lane("LRU"), akv = lane("AgentKV"), mk = lane("AgentKV-Markov"), bel = lane("Belady");
      const next = seq[i + 1];
      const out = [`<span class="step-title">STEP ${i + 1} OF ${seq.length} · ${seq[i].toUpperCase()} RUNS</span>`];
      const ev = (st) => st.events.find((e) => e.type === "evict");
      const lruEv = ev(lru), akvEv = ev(akv), mkEv = ev(mk);

      if (!lruEv && !akvEv && !mkEv) {
        if (i < 3) out.push(`<p>The GPU still has free blocks, so nothing is evicted. ${pill(seq[i])}'s prompt is computed and cached. Only the shared system prompt is reused.</p>`);
        else out.push(`<p>${pill(seq[i])} is already cached by the workflow-aware policies, so nobody has to evict anything this step.</p>`);
      } else {
        if (lruEv) {
          out.push(lruEv.victim === next
            ? `<p><b style="color:#cf4a2e">LRU</b> threw out ${pill(lruEv.victim)} because it was used longest ago. But ${pill(next)} runs <b>next</b>. That's thrashing.</p>`
            : `<p><b style="color:#cf4a2e">LRU</b> threw out ${pill(lruEv.victim)}, the least recently used.</p>`);
        } else if (lru.hit < lru.total) {
          out.push(`<p><b style="color:#cf4a2e">LRU</b> missed again.</p>`);
        }
        if (akvEv) {
          const d = akv.distances ? akv.distances[akvEv.victim] : null;
          out.push(`<p><b style="color:#2d6a4f">AgentKV</b> threw out ${pill(akvEv.victim)}, which won't run for ${d === null ? "a long time" : d + " steps"} (BFS on the workflow graph).</p>`);
        } else if (akv.hit === akv.total) {
          out.push(`<p><b style="color:#2d6a4f">AgentKV</b>: full cache hit, all ${akv.total} blocks reused.</p>`);
        }
        if (wf === "branching" && mkEv && (!akvEv || mkEv.victim !== akvEv.victim)) {
          out.push(`<p><b style="color:#1f8a8a">AgentKV-Markov</b> picked ${pill(mkEv.victim)} instead. It's still learning the probabilities from what it has seen.</p>`);
        }
      }
      const rate = (st) => Math.round((100 * st.cum_hit) / st.cum_total) + "%";
      out.push(`<p class="muted small">Hit rate so far: <b style="color:#cf4a2e">LRU ${rate(lru)}</b> · <b style="color:#2d6a4f">AgentKV ${rate(akv)}</b> · <b style="color:#1f8a8a">Markov ${rate(mk)}</b> · <b style="color:#1d1a14">Bélády ${rate(bel)}</b></p>`);
      return out.join("");
    }

    function render(i) {
      const data = D.eviction[wf];
      seqStrip($("#sim-seq"), data.sequence, i, (k) => { player.stop(); player.go(k); });
      const akv = data.lanes.find((l) => l.policy === "AgentKV").steps[i];
      const badges = {};
      if (akv.distances) for (const a of AGENTS) badges[a] = a === data.sequence[i] ? "now" : akv.distances[a] === null ? "∞" : akv.distances[a] + " step" + (akv.distances[a] === 1 ? "" : "s");
      drawRing($("#sim-ring"), { size: 300, current: data.sequence[i], edges: wf === "cyclic" ? CYCLE_EDGES : branchEdges(0.3), badges });
      $("#sim-ring-caption").textContent = "Badges: BFS steps until each agent runs again (what AgentKV uses)";
      $("#sim-narration").innerHTML = narrate(i);

      const lanesEl = $("#sim-lanes");
      lanesEl.innerHTML = "";
      for (const lane of data.lanes) {
        const st = lane.steps[i];
        const key = wf + lane.policy;
        const used = st.gpu.filter(Boolean).length;
        let badge, text;
        if (st.hit === st.total) { badge = h("span", { class: "badge hit" }, "HIT"); text = `all ${st.total} blocks were already cached`; }
        else if (!data.sequence.slice(0, i).includes(st.agent)) {
          badge = h("span", { class: "badge partial" }, "FIRST RUN");
          text = i === 0 ? "nothing cached yet, the whole prompt is computed" : `first time ${st.agent} runs: only the shared system prompt can hit`;
        }
        else {
          badge = h("span", { class: "badge miss" }, "MISS");
          text = `only ${st.hit} of ${st.total} blocks hit · ${st.total - st.hit} computed`;
          if (st.recomputed) text += ` (${st.recomputed} were cached before and evicted: wasted work)`;
        }
        const decisions = st.events.filter((e) => e.type === "evict");
        const decisionEl = h("div", { class: "decision" });
        if (!decisions.length) decisionEl.append(h("div", { class: "decision-title" }, "No eviction needed this step."));
        for (const e of decisions) {
          decisionEl.append(h("div", { class: "decision-title" }, `Needed room, so it ranked the evictable leaves:`));
          for (const c of e.candidates) {
            decisionEl.append(h("div", { class: "cand" + (c.chosen ? " chosen" : "") },
              h("span", { class: "dot", style: `--c:${AGENT_COLOR[c.label]}` }),
              h("b", {}, c.label), h("span", { class: "why" }, c.why)));
          }
        }
        const pct = (100 * st.cum_hit) / st.cum_total;
        lanesEl.append(h("div", { class: "lane", style: `--pc:${POLICY_COLOR[lane.policy]}` },
          h("div", { class: "lane-head" },
            h("span", { class: "lane-name" }, pretty(lane.policy)),
            h("span", { class: "lane-rule" }, POLICY_RULE[lane.policy])),
          h("div", { class: "mem" }, memCells(st.gpu, prev[key])),
          h("div", { class: "mem-label" }, h("span", {}, "GPU · 13 blocks · each square = 16 tokens"), h("span", {}, `${used}/13 used`)),
          h("div", { class: "result" }, badge, h("span", {}, text)),
          decisionEl,
          h("div", { class: "lane-foot" },
            h("div", { class: "meter" }, h("div", { style: `width:${pct}%` })),
            h("div", { class: "lane-num", html: `hit rate <b>${pct.toFixed(0)}%</b> · recomputed <b>${st.cum_recomputed}</b>` }))));
        prev[key] = st.gpu;
      }
    }

    segmented($("#sim-workflow"), [["cyclic", "4-agent loop"], ["branching", "Branching loop (30% back to Coder)"]], (v) => {
      wf = v; player.stop(); player.setLength(D.eviction[wf].sequence.length);
    });
    player.setLength(D.eviction[wf].sequence.length);
    onFirstView($("#sim-player"), () => player.play());
  }

  // ------------------------------------------------------------------ chapter 5: BFS vs Markov
  function bfs(edges, start) {
    const dist = {}; AGENTS.forEach((a) => (dist[a] = Infinity)); dist[start] = 0;
    const q = [start];
    while (q.length) {
      const a = q.shift();
      for (const e of edges) if (e.from === a && dist[e.to] === Infinity) { dist[e.to] = dist[a] + 1; q.push(e.to); }
    }
    return dist;
  }
  function hittingTime(probs, start, target) {
    if (start === target) return 0;
    const can = new Set([target]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const a of AGENTS) {
        if (can.has(a)) continue;
        for (const b of Object.keys(probs[a] || {})) {
          if (probs[a][b] > 0 && can.has(b)) { can.add(a); changed = true; break; }
        }
      }
    }
    if (!can.has(start)) return Infinity;
    const others = AGENTS.filter((a) => can.has(a) && a !== target);
    const idx = {}; others.forEach((a, k) => (idx[a] = k));
    const n = others.length;
    const A = others.map((_, r) => others.map((__, c) => (r === c ? 1 : 0)));
    const b = others.map(() => 1);
    for (const a of others) for (const [dst, p] of Object.entries(probs[a] || {})) if (dst in idx) A[idx[a]][idx[dst]] -= p;
    // Gaussian elimination with partial pivoting
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      [A[col], A[piv]] = [A[piv], A[col]]; [b[col], b[piv]] = [b[piv], b[col]];
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = A[r][col] / A[col][col];
        for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
        b[r] -= f * b[col];
      }
    }
    return b[idx[start]] / A[idx[start]][idx[start]];
  }

  function initMarkov() {
    const slider = $("#markov-p");
    function update() {
      const p = +slider.value / 100;
      $("#o-p").textContent = Math.round(p * 100) + "%";
      const edges = branchEdges(p);
      const probs = { Planner: { Retriever: 1 }, Retriever: { Coder: 1 }, Coder: { Reviewer: 1 }, Reviewer: { Planner: 1 - p } };
      if (p > 0) probs.Reviewer.Coder = p;
      const B = bfs(edges, "Reviewer");
      const M = {}; for (const a of AGENTS) M[a] = hittingTime(probs, "Reviewer", a);
      const badges = {};
      for (const a of AGENTS) badges[a] = a === "Reviewer" ? "running" : `BFS ${B[a]} · E ${M[a].toFixed(2)}`;
      drawRing($("#markov-graph"), { size: 320, current: "Reviewer", edges, badges });

      const max = Math.max(...["Planner", "Coder", "Retriever"].map((a) => Math.max(B[a], M[a])));
      const bars = $("#markov-bars");
      bars.innerHTML = "";
      for (const a of ["Planner", "Coder", "Retriever"]) {
        bars.append(h("div", { class: "mbar" },
          h("div", { class: "mbar-name", html: pill(a) }),
          h("div", { class: "mbar-tracks" },
            h("div", { class: "track" }, h("span", {}, "BFS"),
              h("div", { class: "bar" }, h("div", { style: `width:${(100 * B[a]) / max}%;background:#2d6a4f` })),
              h("span", { class: "num" }, B[a] + " hop" + (B[a] === 1 ? "" : "s"))),
            h("div", { class: "track" }, h("span", {}, "Markov"),
              h("div", { class: "bar" }, h("div", { style: `width:${(100 * M[a]) / max}%;background:#1f8a8a` })),
              h("span", { class: "num" }, M[a].toFixed(2))))));
      }
      $("#markov-formula").innerHTML = p > 0
        ? `h(Planner) = 1 + <span class="hl">${p.toFixed(2)}</span> × (1 + h)  ⇒  h = (1 + ${p.toFixed(2)}) / (1 − ${p.toFixed(2)}) = <span class="hl">${M.Planner.toFixed(2)}</span> steps`
        : `p = 0: Reviewer always goes to Planner, so it's a plain loop and BFS = Markov.`;
      let verdict;
      if (p === 0) verdict = "With no branch, both methods agree exactly: Planner 1, Retriever 2, Coder 3.";
      else {
        const keep = M.Planner <= M.Coder ? "Planner" : "Coder";
        const drop = keep === "Planner" ? "Coder" : "Planner";
        verdict = `<b>BFS</b> sees Planner and Coder tied at 1 hop, so it can't tell which to keep. ` +
          `<b>Markov</b> knows ${pill(keep)} is expected sooner (${M[keep].toFixed(2)} vs ${M[drop].toFixed(2)} steps), so it keeps ${pill(keep)} and evicts ${pill(drop)}.` +
          (p > 0.5 ? " Above 50%, the loop back to Coder is the likely path, so Markov flips its choice." : "");
      }
      $("#markov-verdict").innerHTML = verdict;
    }
    slider.addEventListener("input", update);
    update();
  }

  // ------------------------------------------------------------------ chapter 6: tiers + prefetch
  function initTiers() {
    let wf = "cyclic";
    const prev = {};
    const player = new Player($("#tier-player"), render);
    $("#tier-note").innerHTML = `Same tiny setup as the simulator: GPU 13 blocks, CPU ${D.setup.cpu_blocks} blocks. ` +
      `Cost model: 1 block takes ${D.setup.recompute_ms_per_block} ms to recompute vs ${D.setup.transfer_ms_per_block} ms to copy over PCIe. ` +
      `Prefetched copies happen while the previous step runs, so they don't count as waiting.`;

    function evChip(e) {
      if (e.type === "ondemand") return h("span", { class: "ev ondemand" }, `⏳ ${e.label} copied back while the request waited (+${e.ms} ms)`);
      if (e.type === "prefetch") return h("span", { class: "ev prefetch" }, `⚡ ${e.label} prefetched before its turn`);
      if (e.type === "offload") return h("span", { class: "ev offload" }, e.phase === "prefetch" ? `↘ ${e.label} moved to CPU to make room for a prefetch` : `↘ ${e.label} offloaded to CPU`);
      return h("span", { class: "ev" }, `✕ ${e.label} deleted`);
    }

    function render(i) {
      const data = D.tiered[wf];
      seqStrip($("#tier-seq"), data.sequence, i, (k) => { player.stop(); player.go(k); });
      const box = $("#tier-lanes");
      box.innerHTML = "";
      for (const lane of data.lanes) {
        const st = lane.steps[i];
        const key = wf + lane.policy;
        const events = st.events.length ? st.events.map(evChip) : [h("span", { class: "ev none" }, "No data moved this step")];
        const status = st.ondemand
          ? h("span", { class: "badge miss" }, "WAITED")
          : st.reused === 5 ? h("span", { class: "badge hit" }, "ON GPU")
          : h("span", { class: "badge partial" }, "FIRST RUN");
        const acc = st.cum_prefetched ? `${st.cum_prefetch_used} of ${st.cum_prefetched}` : "-";
        box.append(h("div", { class: "tier-lane", style: `--pc:${POLICY_COLOR[lane.policy]}` },
          h("div", { class: "lane-head" },
            h("span", { class: "lane-name" }, lane.policy),
            h("span", { class: "lane-rule" }, POLICY_RULE[lane.policy])),
          h("div", { class: "tier-grid" },
            h("div", {},
              h("div", { class: "mem" }, memCells(st.gpu, prev[key + "g"])),
              h("div", { class: "mem-label" }, h("span", {}, "GPU (fast, small)"), h("span", {}, `${st.gpu.filter(Boolean).length}/13`))),
            h("div", {},
              h("div", { class: "mem cpu" }, memCells(st.cpu, prev[key + "c"])),
              h("div", { class: "mem-label" }, h("span", {}, "CPU RAM (big, a PCIe copy away)"), h("span", {}, `${st.cpu.filter(Boolean).length}/${st.cpu.length}`)))),
          h("div", { class: "result" }, status, h("span", { class: "muted small" }, `${st.agent}'s turn`)),
          h("div", { class: "events" }, events),
          h("div", { class: "stall", html:
            `<span>waited on copies: <b>${st.cum_ondemand}</b> blocks</span>` +
            `<span>prefetched blocks used so far: <b>${acc}</b></span>` +
            `<span>modeled stall so far: <b>${st.stall_ms.toFixed(2)} ms</b></span>` })));
        prev[key + "g"] = st.gpu; prev[key + "c"] = st.cpu;
      }
    }
    segmented($("#tier-workflow"), [["cyclic", "4-agent loop"], ["branching", "Branching loop"]], (v) => {
      wf = v; player.stop(); player.setLength(D.tiered[wf].sequence.length);
    });
    player.setLength(D.tiered[wf].sequence.length);
    onFirstView($("#tier-player"), () => player.play());
  }

  // ------------------------------------------------------------------ chapter 7: charts
  function lineChart(container, cfg) {
    container.innerHTML = "";
    const W = 900, H = 360, L = 54, Rm = 20, T = 18, B = 46;
    const pw = W - L - Rm, ph = H - T - B;
    const x = (k) => L + (pw * k) / (cfg.xs.length - 1);
    const y = (v) => T + ph * (1 - v / cfg.yMax);
    const svgEl = s("svg", { viewBox: `0 0 ${W} ${H}` });
    const axis = s("g", { class: "axis" });
    for (let v = 0; v <= cfg.yMax; v += 20) {
      axis.append(s("line", { class: "gridline", x1: L, x2: L + pw, y1: y(v), y2: y(v) }));
      axis.append(s("text", { x: L - 10, y: y(v) + 4, "text-anchor": "end" }, v + "%"));
    }
    cfg.xs.forEach((xv, k) => axis.append(s("text", { x: x(k), y: T + ph + 20, "text-anchor": "middle" }, xv)));
    axis.append(s("text", { x: L + pw / 2, y: H - 6, "text-anchor": "middle" }, cfg.xLabel));
    svgEl.append(axis);
    for (const se of cfg.series) {
      const d = se.values.map((v, k) => `${k ? "L" : "M"}${x(k)},${y(v)}`).join(" ");
      const path = s("path", { d, fill: "none", stroke: se.color, "stroke-width": se.width || 2.4, "stroke-dasharray": se.dash || null, "stroke-linejoin": "round" });
      const len = 2000;
      path.style.strokeDasharray = se.dash || (REDUCED ? "none" : len);
      if (!se.dash && !REDUCED) { path.style.strokeDashoffset = len; path.style.transition = "stroke-dashoffset 1.4s ease"; requestAnimationFrame(() => requestAnimationFrame(() => { path.style.strokeDashoffset = 0; })); }
      svgEl.append(path);
      se.values.forEach((v, k) => svgEl.append(s("circle", { cx: x(k), cy: y(v), r: 3.5, fill: se.color })));
    }
    const guide = s("line", { x1: 0, x2: 0, y1: T, y2: T + ph, stroke: "#9a917f", "stroke-dasharray": "4 4", opacity: 0 });
    svgEl.append(guide);
    const overlay = s("rect", { x: L, y: T, width: pw, height: ph, fill: "transparent" });
    overlay.addEventListener("mousemove", (evt) => {
      const rect = svgEl.getBoundingClientRect();
      const px = ((evt.clientX - rect.left) / rect.width) * W;
      const k = Math.max(0, Math.min(cfg.xs.length - 1, Math.round(((px - L) / pw) * (cfg.xs.length - 1))));
      guide.setAttribute("x1", x(k)); guide.setAttribute("x2", x(k)); guide.setAttribute("opacity", 1);
      const rows = cfg.series.map((se) => `<div class="t-row"><span style="color:${se.color}">${se.name}</span><b>${se.values[k].toFixed(1)}%</b></div>`).join("");
      showTip(evt, `<div class="t-title">${cfg.xs[k]} blocks · ${cfg.xs[k] * 2} MiB</div>${rows}`);
    });
    overlay.addEventListener("mouseleave", () => { guide.setAttribute("opacity", 0); hideTip(); });
    svgEl.append(overlay);
    container.append(svgEl);
    container.append(h("div", { class: "chart-legend" }, cfg.series.map((se) =>
      h("span", { style: `--c:${se.color}` }, h("i", {}), se.name))));
  }

  function initCharts() {
    const policies = ["LRU", "LRU-TailFirst", "AgentKV", "AgentKV-Markov", "Belady"];
    function sweep(wf) {
      const rows = R.sweep.filter((r) => r.workflow === wf);
      lineChart($("#sweep-chart"), {
        xs: rows.map((r) => r.capacity_blocks), yMax: 100, xLabel: "GPU cache size (blocks of 2 MiB)",
        series: policies.map((p) => ({
          name: pretty(p), color: POLICY_COLOR[p], values: rows.map((r) => r[p + "_hit_rate"]),
          dash: p === "Belady" ? "6 5" : null, width: p === "AgentKV" ? 3.6 : 2.2,
        })),
      });
      $("#sweep-note").textContent = wf === "cyclic"
        ? "On the fixed loop AgentKV sits exactly on the Bélády line. 240 and 300 match because 3 agents need 16 + 3 × 96 = 304 blocks. At 420, the tail-first hint alone is enough, so LRU-TailFirst catches up."
        : "With a 30% branch back to Coder, AgentKV stays 2–4 points below Bélády, which knows which branch will actually happen.";
    }
    segmented($("#sweep-tabs"), [["cyclic", "4-agent loop"], ["branching", "Branching loop"]], sweep);
    onFirstView($("#sweep-chart"), () => sweep("cyclic"));

    // seeds dot plot
    (function seedChart() {
      const per = R.headline.seeds_experiment.per_seed;
      const W = 520, H = 300, L = 46, T = 14, B = 54, pw = W - L - 12, ph = H - T - B;
      const lo = 15, hi = 55;
      const y = (v) => T + ph * (1 - (v - lo) / (hi - lo));
      const svgEl = s("svg", { viewBox: `0 0 ${W} ${H}` });
      const axis = s("g", { class: "axis" });
      for (let v = 20; v <= hi; v += 10) {
        axis.append(s("line", { class: "gridline", x1: L, x2: L + pw, y1: y(v), y2: y(v) }));
        axis.append(s("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, v + "%"));
      }
      svgEl.append(axis);
      const colW = pw / policies.length;
      policies.forEach((p, k) => {
        const cx = L + colW * (k + 0.5);
        const vals = per[p];
        const m = vals.reduce((a, v) => a + v, 0) / vals.length;
        svgEl.append(s("line", { x1: cx - colW * 0.32, x2: cx + colW * 0.32, y1: y(m), y2: y(m), stroke: POLICY_COLOR[p], "stroke-width": 3, "stroke-linecap": "round" }));
        vals.forEach((v, sIdx) => {
          const dot = s("circle", { cx: cx + (sIdx - 4.5) * (colW * 0.06), cy: y(v), r: 5, fill: POLICY_COLOR[p], opacity: 0.85 });
          dot.addEventListener("mousemove", (evt) => showTip(evt, `<div class="t-title">${pretty(p)} · seed ${sIdx}</div><div class="t-row"><span>hit rate</span><b>${v.toFixed(1)}%</b></div>`));
          dot.addEventListener("mouseleave", hideTip);
          svgEl.append(dot);
        });
        svgEl.append(s("text", { x: cx, y: T + ph + 20, "text-anchor": "middle", fill: "#3d382e", "font-size": 11 }, pretty(p).replace("AgentKV-", "")));
        svgEl.append(s("text", { x: cx, y: T + ph + 36, "text-anchor": "middle", fill: POLICY_COLOR[p], "font-size": 11, "font-weight": 700 }, m.toFixed(1) + "%"));
      });
      $("#seed-chart").append(svgEl);
    })();

    // workflows grouped bars
    (function wfChart() {
      const show = ["LRU", "AgentKV", "AgentKV-Markov", "Belady"];
      const rows = R.workflows;
      const W = 520, H = 300, L = 40, T = 14, B = 46, pw = W - L - 8, ph = H - T - B;
      const y = (v) => T + ph * (1 - v / 100);
      const svgEl = s("svg", { viewBox: `0 0 ${W} ${H}` });
      const axis = s("g", { class: "axis" });
      for (let v = 0; v <= 100; v += 25) {
        axis.append(s("line", { class: "gridline", x1: L, x2: L + pw, y1: y(v), y2: y(v) }));
        axis.append(s("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, v + "%"));
      }
      svgEl.append(axis);
      const gW = pw / rows.length, bw = (gW * 0.76) / show.length;
      rows.forEach((r, k) => {
        const gx = L + gW * k + gW * 0.12;
        show.forEach((p, j) => {
          const v = r[p + "_hit_rate"];
          const rect = s("rect", { x: gx + j * bw, y: y(v), width: bw - 2, height: ph - (y(v) - T), rx: 3, fill: POLICY_COLOR[p] });
          rect.addEventListener("mousemove", (evt) => showTip(evt, `<div class="t-title">${r.workflow} · ${r.capacity_blocks} blocks</div>` +
            show.map((q) => `<div class="t-row"><span style="color:${POLICY_COLOR[q]}">${pretty(q)}</span><b>${r[q + "_hit_rate"].toFixed(1)}%</b></div>`).join("")));
          rect.addEventListener("mouseleave", hideTip);
          svgEl.append(rect);
        });
        svgEl.append(s("text", { x: L + gW * (k + 0.5), y: T + ph + 20, "text-anchor": "middle", fill: "#3d382e", "font-size": 11.5 }, r.workflow));
        const tie = Math.abs(r.AgentKV_hit_rate - r.LRU_hit_rate) < 0.5;
        svgEl.append(s("text", { x: L + gW * (k + 0.5), y: T + ph + 36, "text-anchor": "middle", fill: tie ? "#c58a17" : "#2d6a4f", "font-size": 10.5, "font-weight": 700 }, tie ? "tie" : "AgentKV wins"));
      });
      $("#wf-chart").append(svgEl);
      $("#wf-chart").append(h("div", { class: "chart-legend" }, show.map((p) => h("span", { style: `--c:${POLICY_COLOR[p]}` }, h("i", {}), pretty(p)))));
    })();

    // stall bars
    const STALL_COLORS = ["#cf4a2e", "#b07d2b", "#c58a17", "#2d6a4f", "#1f8a8a"];
    function stall(key) {
      const t = R.headline.tiered[key];
      const labels = Object.keys(t);
      const max = Math.max(...labels.map((l) => t[l].stall_ms_per_request));
      const box = $("#stall-chart");
      box.innerHTML = "";
      labels.forEach((l, k) => {
        const v = t[l].stall_ms_per_request;
        box.append(h("div", { class: "track", style: "grid-template-columns: 260px 1fr 150px; margin: 10px 0; font-size: 0.86rem" },
          h("span", { style: "color:#1d1a14" }, l),
          h("div", { class: "bar", style: "height:18px" }, h("div", { style: `width:${Math.max(0.6, (100 * v) / max)}%;background:${STALL_COLORS[k]}` })),
          h("span", { class: "num", html: `${v.toFixed(2)} ms · <span style="color:#6a6355">GPU hit ${t[l].gpu_hit_rate.toFixed(0)}%</span>` })));
      });
    }
    segmented($("#stall-tabs"), [["cyclic_300", "Loop · 300"], ["branching_300", "Branching · 300"], ["branching_360", "Branching · 360"], ["fanout_360", "Fan-out · 360"]], stall);
    stall("cyclic_300");

    // sensitivity table
    const cols = ["LRU", "AgentKV", "AgentKV-Markov", "Belady"];
    const table = h("table", { class: "nice" },
      h("thead", {}, h("tr", {}, h("th", {}, "Setting"), h("th", {}, "Workflow"), cols.map((c) => h("th", {}, pretty(c))), h("th", {}, "Gain over LRU"))),
      h("tbody", {}, R.sensitivity.map((r) => {
        const gain = r.AgentKV_hit_rate - r.LRU_hit_rate;
        return h("tr", {}, h("td", {}, r.setting), h("td", {}, r.workflow),
          cols.map((c) => h("td", {}, r[c + "_hit_rate"].toFixed(1) + "%")),
          h("td", { class: "gain" + (gain < 3 ? " small-gain" : "") }, "+" + gain.toFixed(1)));
      })));
    $("#sens-table").append(h("div", { class: "table-wrap" }, table));
  }

  // ------------------------------------------------------------------ chapter 8: limits + facts
  function initLimits() {
    const wf = (name) => R.workflows.find((r) => r.workflow === name);
    const rnd = wf("random"), react = wf("react");
    const tail = R.sensitivity.find((r) => r.setting.startsWith("tail 1024"));
    const tiny = R.sweep.find((r) => r.workflow === "cyclic" && r.capacity_blocks === 120);
    const huge = R.sweep.find((r) => r.workflow === "cyclic" && r.capacity_blocks === 480);
    const cards = [
      ["tie", "Random agent order", "The graph says anyone could be next, so it carries no information. AgentKV falls back to LRU behaviour, and doesn't do worse.",
        `LRU ${fmt(rnd.LRU_hit_rate)}% · AgentKV ${fmt(rnd.AgentKV_hit_rate)}% · Bélády ${fmt(rnd.Belady_hit_rate)}%`],
      ["tie", "ReAct hub (Planner ↔ tools)", "Planner runs every other step, so LRU already keeps it hot. There's nothing to fix.",
        `LRU ${fmt(react.LRU_hit_rate)}% · AgentKV ${fmt(react.AgentKV_hit_rate)}%`],
      ["small", "Mostly new text per prompt", "When 1,024–2,048 fresh tokens dominate each prompt, there's little reusable prefix to protect.",
        `gain only +${fmt(tail.AgentKV_hit_rate - tail.LRU_hit_rate)} points`],
      ["small", "Tiny cache", "Fewer than two agents fit, so no policy can save anything. Everyone sits at the system-prompt floor.",
        `all policies ${fmt(tiny.LRU_hit_rate)}% at ${tiny.capacity_blocks} blocks`],
      ["small", "Huge cache", "Everything fits, nothing is evicted, and every policy ties at the maximum.",
        `all policies ~${fmt(huge.AgentKV_hit_rate)}% at ${huge.capacity_blocks} blocks`],
      ["surprise", "AgentKV beat the oracle", "Bélády's rule is only provably optimal for same-size items. Our tree nodes range from 4 to 96 blocks, and on fan-out it slipped.",
        "fan-out, 300 blocks: AgentKV 37.0% vs Bélády 28.5% (separate 10-seed run)"],
    ];
    const grid = $("#limit-grid");
    for (const [tag, title, text, num] of cards) {
      grid.append(h("div", { class: "limit" },
        h("span", { class: "tag " + tag }, tag === "surprise" ? "surprise" : tag === "tie" ? "tie" : "small effect"),
        h("h4", {}, title), h("p", {}, text), h("span", { class: "num" }, num)));
    }
    const facts = [["80", "pytest tests, including differential tests against a brute-force cache"],
      ["1,700", "simulated runs in ~28 s on a laptop"], ["0", "ML dependencies: pure Python + NumPy"],
      ["7", "experiments, every number an exact count"]];
    const fbox = $("#facts");
    for (const [n, t] of facts) fbox.append(h("div", { class: "fact" }, h("b", {}, n), h("span", {}, t)));
  }

  // ------------------------------------------------------------------ go
  initHero();
  initProblem();
  initKV();
  initRadix();
  initSimulator();
  initMarkov();
  initTiers();
  initCharts();
  initLimits();
})();
