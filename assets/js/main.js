/* Agima project page: interactive figures shared by both site versions.
   All numbers are copied from the paper (Tables 1-3 and Section 5). */
(function () {
  "use strict";

  /* ---------- data ---------- */
  var ENVS = ["Pacman", "Maze2D", "Sokoban", "PushT"];
  // Table 1: success on the hard splits (PushT = success on the far split). null = not evaluated.
  var K_RESULTS = {
    "1": [
      ["Gemini 3.1 Pro", "closed", [0, 38, 3, 0]],
      ["GPT-5.6 Sol", "closed", [0, 48, 32, 0]],
      ["Agima-A7B", "ours", [84, 62, 47, 42]]
    ],
    "3": [
      ["Gemini 3.1 Pro", "closed", [0, 21, 4, 0]],
      ["GPT-5.6 Sol", "closed", [0, 39, 26, 0]],
      ["Qwen2.5-VL-7B", "sft", [0, 40, 4, 4]],
      ["Qwen3-VL-8B", "sft", [0, 85, 10, 6]],
      ["Agima-A7B (No Imagine)", "noimg", [25, 54, 10, 12]],
      ["Agima-A7B", "ours", [38, 59, 53, 24]]
    ],
    "5": [
      ["Gemini 3.1 Pro", "closed", [0, 20, 6, 0]],
      ["GPT-5.6 Sol", "closed", [1, 34, 20, 1]],
      ["Qwen2.5-VL-7B", "sft", [0, 28, 0, 2]],
      ["Qwen3-VL-8B", "sft", [0, 30, 3, 1]],
      ["Agima-A7B (No Imagine)", "noimg", [1, 23, 2, 1]],
      ["Agima-A7B", "ours", [37, 40, 40, 18]]
    ],
    "10": [
      ["Gemini 3.1 Pro", "closed", [0, 15, 4, 0]],
      ["GPT-5.6 Sol", "closed", [1, 34, 23, 0]],
      ["Qwen2.5-VL-7B", "sft", [0, 0, 0, 2]],
      ["Qwen3-VL-8B", "sft", [0, 0, 0, 0]],
      ["Agima-A7B (No Imagine)", "noimg", [0, 2, 0, 1]],
      ["Agima-A7B", "ours", [7, 21, 15, 9]]
    ],
    "inf": [
      ["Gemini 3.1 Pro", "closed", [0, 9, 13, 0]],
      ["GPT-5.6 Sol", "closed", [2, 36, 21, 0]],
      ["Qwen2.5-VL-7B", "sft", [0, 0, 0, 2]],
      ["Qwen3-VL-8B", "sft", [0, 0, 0, 3]],
      ["Agima-A7B (No Imagine)", "noimg", [0, 0, 0, 3]],
      ["Agima-A7B", "ours", [6, 3, 19, 3]]
    ]
  };
  var K_NOTE = {
    "1": "K = 1: a real observation after every action (reactive reference).",
    "3": "K = 3: the agent plans 3 actions, then sees the real board again.",
    "5": "K = 5: the agent plans 5 actions from its own imagined frames, then sees the real board again.",
    "10": "K = 10: long stretches with no real feedback.",
    "inf": "K = ∞: one real observation at the start, then fully open-loop."
  };
  // Table 2: imagined frames replaced at inference (K = 5, hard splits).
  var ABLATION = {
    conditions: ["No Imagine", "Own predictions", "Blank", "Stale", "Oracle"],
    notes: ["action-only model", "the model's own imagined frames", "uniform gray image",
            "latest real observation", "ground-truth frames for the proposed actions"],
    Sokoban: [2, 40, 1, 0, 43],
    Pacman: [1, 37, 0, 0, 81]
  };
  // Table 3 (a) Sokoban and (b) Pacman route-order traps, K = infinity.
  var REASONING = {
    sokoban: {
      cols: ["Deadlock 2-box", "Deadlock 3-box", "Deadlock 4-box", "Fork 2-box", "Fork 3-box", "Fork 4-box"],
      rows: [
        ["Gemini 3.1 Pro", "closed", [74, 42, 19, 84, 34, 7]],
        ["GPT-5.6 Sol", "closed", [44, 23, 17, 60, 20, 12]],
        ["Qwen2.5-VL-7B", "sft", [1, 2, 0, 12, 2, 0]],
        ["Qwen3-VL-8B", "sft", [6, 1, 1, 19, 2, 0]],
        ["Agima-A7B (No Imagine)", "noimg", [12, 5, 1, 15, 2, 1]],
        ["Agima-A7B", "ours-lin", [22, 39, 27, 87, 34, 22]],
        ["Agima-A7B (Branching)", "ours", [30, 39, 32, 84, 39, 32]],
        ["Agima-A7B (Forward)", "ours", [60, 48, 24, 93, 46, 22]]
      ]
    },
    pacman: {
      cols: ["G8 / F2", "G8 / F4", "G8 / F6", "G8 / F8", "G10 / F8"],
      rows: [
        ["Gemini 3.1 Pro", "closed", [1, 1, 0, 1, 0]],
        ["GPT-5.6 Sol", "closed", [16, 19, 6, 8, 0]],
        ["Qwen2.5-VL-7B", "sft", [0, 0, 0, 0, 0]],
        ["Qwen3-VL-8B", "sft", [0, 0, 0, 1, 0]],
        ["Agima-A7B (No Imagine)", "noimg", [3, 0, 2, 0, 0]],
        ["Agima-A7B", "ours-lin", [7, 22, 16, 16, 5]],
        ["Agima-A7B (Forward)", "ours", [40, 54, 41, 37, 12]]
      ]
    }
  };
  // Rollout explorer: pages rendered from the appendix trajectory figures.
  var ROLLOUTS = {
    maze2d: {label: "Maze2D", k: {"1": [["maze2d_k1_1", "0-17"]], "3": [["maze2d_k3_2", "0-15"], ["maze2d_k3_3", "15-27"]], "5": [["maze2d_k5_4", "0-23"]], "10": [["maze2d_k10_5", "0-21"]], "inf": [["maze2d_kinf_6", "0-13"]]}},
    sokoban: {label: "Sokoban", k: {"1": [["sokoban_k1_1", "0-21"]], "3": [["sokoban_k3_2", "0-21"]], "5": [["sokoban_k5_3", "0-19"]], "10": [["sokoban_k10_4", "0-19"]], "inf": [["sokoban_kinf_5", "0-17"]]}},
    pacman: {label: "Pacman", k: {"1": [["pacman_k1_1", "0-33"]], "3": [["pacman_k3_2", "0-20"], ["pacman_k3_3", "20-33"]], "5": [["pacman_k5_4", "0-20"], ["pacman_k5_5", "20-33"]], "10": [["pacman_k10_6", "0-20"], ["pacman_k10_7", "20-35"]], "inf": [["pacman_kinf_8", "0-15"], ["pacman_kinf_9", "15-27"]]}},
    pusht: {label: "PushT", k: {"1": [["pusht_k1_1", "0-19"]], "3": [["pusht_k3_2", "0-18"]], "5": [["pusht_k5_3", "0-15"], ["pusht_k5_4", "15-27"]], "10": [["pusht_k10_5", "0-20"]], "inf": [["pusht_kinf_6", "0-15"], ["pusht_kinf_7", "15-28"]]}}
  };
  var KLABEL = {"1": "1", "3": "3", "5": "5", "10": "10", "inf": "∞"};
  var FIG = "assets/images/";

  /* ---------- helpers ---------- */
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "text") n.textContent = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else n.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) n.appendChild(c); });
    return n;
  }
  function segmented(container, name, options, current, onChange) {
    var group = el("div", {class: "seg", role: "tablist", "aria-label": name});
    options.forEach(function (o) {
      var b = el("button", {type: "button", class: "seg-btn", role: "tab", "data-value": o[0],
                             "aria-selected": String(o[0] === current), id: name.replace(/\W+/g, "-") + "-" + o[0], text: o[1]});
      b.addEventListener("click", function () {
        group.querySelectorAll(".seg-btn").forEach(function (x) { x.setAttribute("aria-selected", String(x === b)); });
        onChange(o[0]);
      });
      group.appendChild(b);
    });
    container.appendChild(group);
    return group;
  }
  var tip = null;
  function tooltip(target, text) {
    target.addEventListener("mouseenter", function (e) {
      if (!tip) { tip = el("div", {class: "chart-tip", role: "status"}); document.body.appendChild(tip); }
      tip.textContent = text; tip.hidden = false; place(e);
    });
    target.addEventListener("mousemove", place);
    target.addEventListener("mouseleave", function () { if (tip) tip.hidden = true; });
    function place(e) { if (!tip) return; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px"; }
  }

  /* ---------- 1. K results: horizontal bars per environment ---------- */
  function renderKResults(root) {
    var state = {k: "5"};
    var controls = el("div", {class: "chart-controls"});
    root.appendChild(controls);
    controls.appendChild(el("h3", {text: "Success across K"}));
    segmented(controls, "Actions before feedback K", [["1", "K = 1"], ["3", "K = 3"], ["5", "K = 5"], ["10", "K = 10"], ["inf", "K = ∞"]], state.k, function (v) { state.k = v; draw(); });
    var toggle = el("button", {type: "button", class: "link-btn", id: "k-results-table-toggle", text: "Show as table"});
    controls.appendChild(toggle);
    var note = el("p", {class: "chart-note"});
    var grid = el("div", {class: "bar-grid"});
    var table = el("div", {class: "table-wrap", hidden: ""});
    root.appendChild(note); root.appendChild(grid); root.appendChild(table);
    root.appendChild(legend());
    toggle.addEventListener("click", function () {
      var show = table.hidden; table.hidden = !show; grid.hidden = show;
      toggle.textContent = show ? "Show as chart" : "Show as table";
    });
    function draw() {
      var rows = K_RESULTS[state.k];
      note.textContent = K_NOTE[state.k];
      grid.innerHTML = "";
      ENVS.forEach(function (env, ei) {
        var panel = el("div", {class: "bar-panel"}, [el("h4", {text: env})]);
        rows.forEach(function (r) {
          var v = r[2][ei];
          var bar = el("div", {class: "bar bar-" + r[1], style: "width:" + Math.max(v, 0.6) + "%"});
          var row = el("div", {class: "bar-row"}, [
            el("span", {class: "bar-name", text: r[0].replace("Agima-A7B", "Agima")}),
            el("span", {class: "bar-track"}, [bar]),
            el("span", {class: "bar-val", text: v + "%"})
          ]);
          tooltip(row, r[0] + " · " + env + " · K = " + KLABEL[state.k] + ": " + v + "% success");
          panel.appendChild(row);
        });
        grid.appendChild(panel);
      });
      table.innerHTML = "";
      var t = el("table", {class: "data-table"});
      var head = el("tr", {}, [el("th", {text: "Method (K = " + KLABEL[state.k] + ")"})].concat(ENVS.map(function (e) { return el("th", {text: e}); })));
      t.appendChild(el("thead", {}, [head]));
      var tb = el("tbody");
      rows.forEach(function (r) {
        tb.appendChild(el("tr", {class: r[1] === "ours" ? "is-ours" : ""}, [el("td", {text: r[0]})].concat(r[2].map(function (v) { return el("td", {text: v + "%"}); }))));
      });
      t.appendChild(tb); table.appendChild(t);
    }
    draw();
  }
  function legend() {
    return el("div", {class: "chart-legend"}, [
      el("span", {}, [el("i", {class: "sw sw-ours"}), document.createTextNode("Agima")]),
      el("span", {}, [el("i", {class: "sw sw-noimg"}), document.createTextNode("Agima (No Imagine)")]),
      el("span", {}, [el("i", {class: "sw sw-other"}), document.createTextNode("Other baselines")])
    ]);
  }

  /* ---------- 2. imagination ablation ---------- */
  function renderAblation(root) {
    var grid = el("div", {class: "bar-grid bar-grid-2"});
    ["Sokoban", "Pacman"].forEach(function (env) {
      var panel = el("div", {class: "bar-panel"}, [el("h4", {text: env + " (K = 5)"})]);
      ABLATION.conditions.forEach(function (c, i) {
        var v = ABLATION[env][i];
        var cls = c === "Own predictions" ? "ours" : c === "Oracle" ? "oracle" : c === "No Imagine" ? "noimg" : "other";
        var row = el("div", {class: "bar-row"}, [
          el("span", {class: "bar-name", text: c}),
          el("span", {class: "bar-track"}, [el("div", {class: "bar bar-" + cls, style: "width:" + Math.max(v, 0.6) + "%"})]),
          el("span", {class: "bar-val", text: v + "%"})
        ]);
        tooltip(row, env + " · " + c + " (" + ABLATION.notes[i] + "): " + v + "%");
        panel.appendChild(row);
      });
      grid.appendChild(panel);
    });
    root.appendChild(grid);
  }

  /* ---------- 3. rollout explorer ---------- */
  function renderExplorer(root) {
    var state = {env: "sokoban", k: "5"};
    var controls = el("div", {class: "chart-controls explorer-controls"});
    root.appendChild(controls);
    segmented(controls, "Environment", Object.keys(ROLLOUTS).map(function (e) { return [e, ROLLOUTS[e].label]; }), state.env, function (v) { state.env = v; draw(); });
    segmented(controls, "Explorer K", [["1", "K = 1"], ["3", "K = 3"], ["5", "K = 5"], ["10", "K = 10"], ["inf", "K = ∞"]], state.k, function (v) { state.k = v; draw(); });
    var caption = el("p", {class: "chart-note"});
    var frame = el("div", {class: "explorer-frame"});
    root.appendChild(caption); root.appendChild(frame);
    function draw() {
      var pages = ROLLOUTS[state.env].k[state.k];
      var first = pages[0][1].split("-")[0], last = pages[pages.length - 1][1].split("-")[1];
      caption.textContent = ROLLOUTS[state.env].label + ", K = " + KLABEL[state.k] + ": steps " + first + "–" + last +
        (state.k === "1" ? ". Real observation after every action." : state.k === "inf" ? ". One real observation, then every frame is imagined." : ". A real observation every " + state.k + " actions; frames in between are imagined.");
      frame.innerHTML = "";
      pages.forEach(function (p) {
        frame.appendChild(el("img", {src: FIG + "rollouts/" + p[0] + ".webp", loading: "lazy", decoding: "async",
          alt: ROLLOUTS[state.env].label + " trajectory at K = " + KLABEL[state.k] + ", steps " + p[1]}));
      });
    }
    draw();
  }

  /* ---------- 4. reasoning tables ---------- */
  function renderReasoning(root) {
    var state = {task: "sokoban"};
    var controls = el("div", {class: "chart-controls"});
    root.appendChild(controls);
    controls.appendChild(el("h3", {text: "Success on planning traps"}));
    segmented(controls, "Reasoning task", [["sokoban", "Sokoban"], ["pacman", "Pacman"]], state.task, function (v) { state.task = v; draw(); });
    var wrap = el("div", {class: "table-wrap"});
    root.appendChild(wrap);
    function draw() {
      var d = REASONING[state.task];
      var best = d.cols.map(function (_, ci) { return Math.max.apply(null, d.rows.map(function (r) { return r[2][ci]; })); });
      var t = el("table", {class: "data-table"});
      t.appendChild(el("thead", {}, [el("tr", {}, [el("th", {text: "Method"})].concat(d.cols.map(function (c) { return el("th", {text: c}); })))]));
      var tb = el("tbody");
      var lin = d.rows.filter(function (r) { return r[1] === "ours-lin"; })[0];
      d.rows.forEach(function (r) {
        tb.appendChild(el("tr", {class: r[1] === "ours" ? "is-ours" : r[1] === "ours-lin" ? "is-lin" : r[1] === "closed" ? "is-closed" : ""},
          [el("td", {text: r[0]})].concat(r[2].map(function (v, ci) {
            var delta = r[1] === "ours" ? v - lin[2][ci] : null;
            var cell = el("td", {class: v === best[ci] ? "is-best" : ""}, [document.createTextNode(v + "%")]);
            if (delta !== null) cell.appendChild(el("span", {class: "delta " + (delta > 0 ? "up" : delta < 0 ? "down" : "flat"), text: (delta > 0 ? "+" : "") + delta}));
            return cell;
          }))));
      });
      t.appendChild(tb);
      wrap.innerHTML = ""; wrap.appendChild(t);
      wrap.appendChild(el("p", {class: "table-foot", text: state.task === "sokoban"
        ? "All models trained on two-box Sokoban and run at K = ∞. Bold = best in column; small numbers = change vs. linear-rollout Agima (points)."
        : "Route-order traps: collecting the nearer food first leaves no safe way to finish. G = grid size, F = food items. K = ∞."}));
    }
    draw();
  }


  /* ---------- 5. what K means: one block of the action-chunk figure per K ---------- */
  function renderKChunks(root) {
    var info = {
      "1": "K = 1: after every action the real frame comes back. No imagined frames.",
      "3": "K = 3: imagine three moves, commit them, then receive the real frame.",
      "5": "K = 5: imagine five moves before each real frame.",
      "10": "K = 10: imagine ten moves before each real frame.",
      "inf": "K = ∞: one real frame at the start; the whole episode is imagined and committed at once."
    };
    var sizes = {"1": [2400, 326], "3": [2400, 586], "5": [2400, 586], "10": [2400, 586], "inf": [2400, 444]};
    var controls = el("div", {class: "chart-controls"});
    root.appendChild(controls);
    var note = el("p", {class: "chart-note"});
    var legend = el("img", {class: "k-legend", src: FIG + "kchunks/legend.webp", width: "2400", height: "76",
      alt: "Legend: o ground-truth frame, ô imagined frame, a committed action, â imagined action."});
    var img = el("img", {class: "k-block", width: "2400", height: "586", alt: ""});
    root.appendChild(note); root.appendChild(el("div", {class: "k-frame"}, [legend, img]));
    function show(k) {
      note.textContent = info[k];
      img.src = FIG + "kchunks/k" + k + ".webp";
      img.width = sizes[k][0]; img.height = sizes[k][1];
      img.alt = "The same Sokoban episode planned with K = " + KLABEL[k] + ".";
    }
    segmented(controls, "Action chunk size K", [["1", "K = 1"], ["3", "K = 3"], ["5", "K = 5"], ["10", "K = 10"], ["inf", "K = ∞"]], "5", show);
    show("5");
  }

  /* ---------- tabs, copy, back-to-top, contents nav ---------- */
  function initTabs() {
    document.querySelectorAll("[data-tabs]").forEach(function (box) {
      var btns = box.querySelectorAll("[data-tab]");
      btns.forEach(function (b) {
        b.addEventListener("click", function () {
          btns.forEach(function (x) { x.setAttribute("aria-selected", String(x === b)); });
          box.querySelectorAll("[data-panel]").forEach(function (p) { p.hidden = p.getAttribute("data-panel") !== b.getAttribute("data-tab"); });
        });
      });
    });
  }
  function initCopy() {
    document.querySelectorAll("[data-copy]").forEach(function (b) {
      b.addEventListener("click", function () {
        var src = document.getElementById(b.getAttribute("data-copy"));
        var done = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1600); };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(src.textContent).then(done, function () { select(src); });
        else select(src);
      });
    });
    function select(n) { var r = document.createRange(); r.selectNodeContents(n); var s = getSelection(); s.removeAllRanges(); s.addRange(r); }
  }
  function initContents() {
    var nav = document.querySelector(".contents");
    var startEl = document.querySelector(".prose .lead");
    if (!nav || !startEl) return;
    var items = Array.prototype.slice.call(nav.querySelectorAll('a[href^="#"]')).map(function (a) {
      return {link: a, target: document.querySelector(a.getAttribute("href"))};
    }).filter(function (x) { return x.target; });
    var frame;
    function update() {
      frame = null;
      var marker = window.scrollY + innerHeight * 0.4;
      var startScroll = startEl.getBoundingClientRect().top + window.scrollY - innerHeight * 0.55;
      var endScroll = Math.max(startScroll + 1, document.documentElement.scrollHeight - innerHeight);
      var progress = Math.min(1, Math.max(0, (window.scrollY - startScroll) / (endScroll - startScroll)));
      nav.classList.toggle("is-visible", window.scrollY >= startScroll);
      nav.style.setProperty("--scroll-progress", progress);
      var active = items[0];
      items.forEach(function (it) { if (it.target.getBoundingClientRect().top + window.scrollY <= marker) active = it; });
      items.forEach(function (it) { if (it === active) it.link.setAttribute("aria-current", "location"); else it.link.removeAttribute("aria-current"); });
    }
    function request() { if (!frame) frame = requestAnimationFrame(update); }
    window.addEventListener("scroll", request, {passive: true});
    window.addEventListener("resize", request);
    update();
  }

  /* ---------- Figure 1: step through linear rollout / branching / forward thinking ---------- */
  function initImagineAnimation() {
    var diagram = document.querySelector(".team-diagram");
    var hl = document.querySelector(".team-highlight-primary");
    var cards = Array.prototype.slice.call(document.querySelectorAll("[data-format-step]"));
    if (!diagram || !hl || !cards.length) return;
    // Regions of fig1_overview (percent of the image): panel (b), then the two halves of panel (c).
    var steps = [
      {name: "linear", box: {left: "50.4%", top: "0.3%", width: "49.3%", height: "24.2%"}},
      {name: "branch", box: {left: "0.6%", top: "27.8%", width: "52.6%", height: "53.2%"}},
      {name: "forward", box: {left: "53.1%", top: "27.8%", width: "46.6%", height: "52.4%"}}
    ];
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var idx = 0, timer = null;
    function show(i) {
      idx = i;
      Object.keys(steps[i].box).forEach(function (k) { hl.style[k] = steps[i].box[k]; });
      cards.forEach(function (c) { c.classList.toggle("is-active", c.getAttribute("data-format-step") === steps[i].name); });
    }
    function schedule() { clearTimeout(timer); if (!reduce) timer = setTimeout(function () { show((idx + 1) % steps.length); schedule(); }, 3000); }
    cards.forEach(function (c, i) {
      c.setAttribute("tabindex", "0");
      var go = function () { diagram.classList.add("is-active"); show(i); schedule(); };
      c.addEventListener("click", go);
      c.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
    show(0);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        if (entries[0].isIntersecting) { diagram.classList.add("is-active"); show(idx); schedule(); }
        else { clearTimeout(timer); }
      }, {threshold: 0.35}).observe(diagram);
    } else { diagram.classList.add("is-active"); }
  }

  function boot() {
    var m;
    if ((m = document.getElementById("k-results"))) renderKResults(m);
    if ((m = document.getElementById("ablation"))) renderAblation(m);
    if ((m = document.getElementById("rollout-explorer"))) renderExplorer(m);
    if ((m = document.getElementById("reasoning-results"))) renderReasoning(m);
    if ((m = document.getElementById("k-chunks"))) renderKChunks(m);
    initTabs(); initCopy(); initContents(); initImagineAnimation();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
