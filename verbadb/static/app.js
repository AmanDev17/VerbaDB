"use strict";

/* Everything is rendered with DOM nodes and textContent — model, database and user text is never parsed as HTML. */

const $ = (id) => document.getElementById(id);
const state = {
  sessionId: newSessionId(), dbId: null, databases: [], busy: false, controller: null, health: null,
  details: null, undoButtons: new Map(), refreshTimer: null,
};

const BUILDER_EXAMPLES = [
  "Create a database for a small bookstore with authors, books, customers and orders, with sample data",
  "Make a database for a pet clinic: owners, pets, vets and appointments",
  "Build a student and course enrollment database with grades",
  "Create a simple expense tracker with categories and some sample transactions",
];
const SANDBOX_EXAMPLES = [
  "What tables are in this database, and how are they related?",
  "Add 5 more sample rows to the largest table",
  "Add a new column to the largest table and fill it with sensible values",
  "Delete the 3 oldest rows from the largest table and show me what changed",
];
const ANALYTICS_EXAMPLES = [
  "Top 5 customers by revenue last month",
  "Monthly revenue for the last 3 months",
  "Which product categories sell the most?",
  "How many orders were cancelled or refunded last month?",
];
const GENERIC_EXAMPLES = [
  "What tables are in this database, and what do they contain?",
  "Show me a few example rows from the biggest table",
  "Which table has the most rows?",
];

function newSessionId() { return crypto.randomUUID().replace(/-/g, ""); }

function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === false || value == null) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

async function api(method, url, body) {
  const resp = await fetch(url, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      if (typeof data.detail === "string") detail = data.detail;
      else if (Array.isArray(data.detail)) detail = "Please check what you entered.";
    } catch (_) { /* keep the default */ }
    throw new Error(detail);
  }
  return resp.status === 204 ? null : resp.json();
}

function remember(key, value) {
  try { if (value == null) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch (_) { /* private mode */ }
}
function recall(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}

/* ---------- colour theme: follows the OS until the switch is used, then remembers the choice ---------- */

const THEME_KEY = "verbadb.theme";
const systemDark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

function explicitTheme() {
  const value = document.documentElement.getAttribute("data-theme");   // set early by theme.js
  return value === "dark" || value === "light" ? value : null;
}

function effectiveTheme() {
  return explicitTheme() || (systemDark && systemDark.matches ? "dark" : "light");
}

function renderTheme() {
  $("theme-toggle").setAttribute("aria-checked", String(effectiveTheme() === "dark"));
  $("theme-auto").hidden = explicitTheme() === null;                    // "Auto" appears only while a choice overrides the system
  const meta = document.querySelector('meta[name="theme-color"]');      // browser chrome follows the page
  if (meta) meta.setAttribute("content", getComputedStyle(document.documentElement).getPropertyValue("--surface").trim() || "#ffffff");
}

function setTheme(mode) {                                               // "light" | "dark" | null (= follow the system)
  const root = document.documentElement;
  if (mode) root.setAttribute("data-theme", mode); else root.removeAttribute("data-theme");
  remember(THEME_KEY, mode);
  renderTheme();
}

/* ---------- side panel (drawer on small screens) ---------- */

function setMenu(open) {
  $("side").classList.toggle("open", open);
  $("menu-toggle").setAttribute("aria-expanded", String(open));
}

function timeAgo(iso) {
  if (!iso) return "";
  const then = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
  const seconds = Math.max(0, (Date.now() - then.getTime()) / 1000);
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  if (seconds < 172800) return "yesterday";
  return then.toLocaleDateString();
}

/* ---------- tiny, safe markdown: paragraphs, lists, tables, headings, **bold**, *italic*, `code` ---------- */

function inline(text) {
  const out = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s](?:[^*]*[^*\s])?\*)/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) out.push(text.slice(last, match.index));
    const token = match[0];
    if (token.startsWith("`")) out.push(h("code", { text: token.slice(1, -1) }));
    else if (token.startsWith("**")) out.push(h("strong", { text: token.slice(2, -2) }));
    else out.push(h("em", { text: token.slice(1, -1) }));
    last = match.index + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function renderMarkdown(text) {
  const root = h("div");
  let list = null;
  let listType = null;
  let table = null;
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trimEnd();
    if (!line.trim()) { list = null; listType = null; table = null; continue; }

    if (/^\s*\|.*\|\s*$/.test(line)) {
      if (/^[\s|:-]+$/.test(line) && line.includes("-")) continue;
      const row = cells(line);
      list = null;
      if (!table) {
        table = { head: h("thead", {}, h("tr", {}, row.map((c) => h("th", { scope: "col" }, inline(c))))), body: h("tbody") };
        root.append(h("div", { class: "table-wrap md-table", tabindex: "0", role: "group", "aria-label": "Table in the answer" },
          h("table", {}, table.head, table.body)));
      } else {
        table.body.append(h("tr", {}, row.map((c) => h("td", {}, inline(c)))));
      }
      continue;
    }
    table = null;

    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (heading) {
      list = null; listType = null;
      root.append(h("p", { class: "md-h" }, inline(heading[1].trim())));
    } else if (bullet || numbered) {
      const type = bullet ? "ul" : "ol";
      if (!list || listType !== type) { list = h(type); listType = type; root.append(list); }
      list.append(h("li", {}, inline((bullet || numbered)[1])));
    } else {
      list = null; listType = null;
      root.append(h("p", {}, inline(line.trim())));
    }
  }
  return root;
}

/* ---------- results table ---------- */

function isNumber(v) { return typeof v === "number" && Number.isFinite(v); }

function formatCell(v) {
  if (v === null || v === undefined) return h("td", { class: "null", text: "—" });
  if (isNumber(v)) return h("td", { class: "num", text: v.toLocaleString(undefined, { maximumFractionDigits: 4 }) });
  if (typeof v === "object") return h("td", { text: JSON.stringify(v), title: JSON.stringify(v) });
  return h("td", { text: String(v), title: String(v) });
}

function toCSV(columns, rows) {
  const cell = (v) => {
    if (v === null || v === undefined) return "";
    let s = typeof v === "object" ? JSON.stringify(v) : String(v);
    if (typeof v === "string" && /^[=+\-@\t\r]/.test(s)) s = "'" + s;     // neutralise spreadsheet formulas
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  return [columns.map(cell).join(","), ...rows.map((r) => columns.map((c) => cell(r[c])).join(","))].join("\r\n");
}

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderResult(ev) {
  const numeric = ev.columns.map((c) => ev.rows.some((r) => isNumber(r[c])) && ev.rows.every((r) => r[c] == null || isNumber(r[c])));
  const head = h("div", { class: "head" },
    h("span", { class: "title", text: `${ev.row_count.toLocaleString()} row${ev.row_count === 1 ? "" : "s"}` }),
    ev.truncated ? h("span", { class: "tag", text: "stopped at the row limit" }) : null,
    h("span", { class: "muted small", text: `${ev.elapsed_ms} ms` }),
    h("span", { class: "spacer" }),
    ev.rows.length ? h("button", { class: "btn", type: "button", text: "Download CSV" }) : null,
  );
  const csvButton = head.querySelector("button");
  if (csvButton) csvButton.addEventListener("click", () => download("result.csv", toCSV(ev.columns, ev.rows)));

  const query = h("details", { class: "query" },
    h("summary", { text: ev.language === "mongodb" ? `Show the pipeline (${ev.collection})` : "Show the SQL that ran" }),
    h("pre", { class: "code", text: ev.query }),
  );

  let body;
  if (!ev.rows.length) {
    body = h("div", { class: "empty-result", text: "The query ran but returned no rows." });
  } else {
    const thead = h("thead", {}, h("tr", {}, ev.columns.map((c, i) => h("th", { scope: "col", class: numeric[i] ? "num" : "", text: c }))));
    const tbody = h("tbody", {}, ev.rows.map((r) => h("tr", {}, ev.columns.map((c, i) => {
      const td = formatCell(r[c]);
      if (numeric[i]) td.classList.add("num");
      return td;
    }))));
    body = h("div", { class: "table-wrap", tabindex: "0", role: "group", "aria-label": "Query result table" },
      h("table", {}, h("caption", { class: "sr-only", text: `Query result: ${ev.row_count} row${ev.row_count === 1 ? "" : "s"}` }), thead, tbody));
  }
  return h("div", { class: "card result", role: "group", "aria-label": "Query result" }, head, query, body);
}

/* ---------- simulated change (before / after) ---------- */

function plain(v) { return v === null || v === undefined ? "NULL" : typeof v === "object" ? JSON.stringify(v) : String(v); }

function diffCell(v, extra) {
  const td = formatCell(v);
  if (extra) td.classList.add(extra);
  return td;
}

function renderTableChange(t) {
  const counts = h("span", { class: "counts" });
  const parts = [];
  if (t.inserted_total) parts.push(h("span", { class: "p", text: `+${t.inserted_total}` }));
  if (t.deleted_total) parts.push(h("span", { class: "m", text: `−${t.deleted_total}` }));
  if (t.updated_total) parts.push(h("span", { class: "e", text: `~${t.updated_total}` }));
  parts.forEach((p, i) => { if (i) counts.append(" "); counts.append(p); });
  counts.append(`${parts.length ? " · " : ""}${t.before_count.toLocaleString()} → ${t.after_count.toLocaleString()} rows`);

  const block = h("div", { class: "tbl-change" },
    h("h3", {}, h("span", { class: `kind ${t.change}`, text: t.change }), t.name, counts));
  if (t.columns_added.length) block.append(h("p", { class: "cols-note", text: `Columns added: ${t.columns_added.join(", ")}` }));
  if (t.columns_removed.length) block.append(h("p", { class: "cols-note", text: `Columns removed: ${t.columns_removed.join(", ")}` }));

  const rows = [];
  const mark = (symbol) => h("td", { class: "mark", text: symbol, "aria-label": symbol === "+" ? "added" : symbol === "−" ? "removed" : "edited" });
  for (const row of t.inserted) rows.push(h("tr", { class: "ins" }, mark("+"), t.columns.map((c) => diffCell(row[c]))));
  for (const row of t.deleted) rows.push(h("tr", { class: "del" }, mark("−"), t.columns.map((c) => diffCell(row[c]))));
  for (const u of t.updated) {
    rows.push(h("tr", { class: "upd" }, mark("~"), t.columns.map((c) => {
      if (!u.changed.includes(c)) return diffCell(u.after[c]);
      return h("td", { class: "chg", title: `${plain(u.before[c])} → ${plain(u.after[c])}` },
        h("s", { text: plain(u.before[c]) }), h("strong", { text: plain(u.after[c]) }));
    })));
  }
  if (rows.length) {
    const thead = h("thead", {}, h("tr", {}, h("th", { scope: "col" }, h("span", { class: "sr-only", text: "Change" })),
      t.columns.map((c) => h("th", { scope: "col", text: c }))));
    block.append(h("div", { class: "diff", tabindex: "0", role: "group", "aria-label": `Changed rows of ${t.name}` },
      h("table", {}, h("caption", { class: "sr-only", text: `Changes to ${t.name}: added, removed and edited rows` }), thead, h("tbody", {}, rows))));
  }
  const notes = [];
  if (t.inserted_total > t.inserted.length) notes.push(`${t.inserted_total - t.inserted.length} more added`);
  if (t.deleted_total > t.deleted.length) notes.push(`${t.deleted_total - t.deleted.length} more removed`);
  if (t.updated_total > t.updated.length) notes.push(`${t.updated_total - t.updated.length} more edited`);
  if (notes.length) block.append(h("p", { class: "diff-note", text: `…and ${notes.join(", ")} (not shown)` }));
  if (!t.row_diff) block.append(h("p", { class: "diff-note", text: "This table is too large to compare row by row; only the row count is shown." }));
  return block;
}

function renderChange(ev, turn) {
  const undo = ev.can_undo ? h("button", { class: "btn small", type: "button", text: "Undo this change" }) : null;
  const head = h("div", { class: "head" },
    h("span", { class: "title", text: "Simulated change" }),
    h("span", { class: "summary", text: ev.summary }),
    h("span", { class: "spacer" }),
    undo);
  const sql = h("details", { class: "sql" },
    h("summary", { text: `Show the ${ev.statements.length} statement${ev.statements.length === 1 ? "" : "s"} that ran` }),
    h("pre", { class: "code", text: ev.statements.join(";\n\n") + ";" }));
  const card = h("div", { class: "card change", role: "group", "aria-label": "Simulated change" }, head, sql,
    ev.tables.length ? ev.tables.map(renderTableChange) : h("div", { class: "empty-result", text: "Nothing visible changed." }));

  if (undo) {
    // Only the newest change of a database can be undone from its card: older ones lose their button.
    const previous = state.undoButtons.get(ev.database_id);
    if (previous && previous.isConnected && !previous.disabled) {
      previous.disabled = true;
      previous.title = "A newer change exists. Undo that one first (or use Undo in the database list).";
    }
    state.undoButtons.set(ev.database_id, undo);
    undo.addEventListener("click", async () => {
      undo.disabled = true;
      try {
        const res = await api("POST", `/api/databases/${encodeURIComponent(ev.database_id)}/undo`);
        undo.textContent = "Undone ✓";
        card.classList.add("undone");
        head.append(h("span", { class: "muted small", text: res.undone ? `Restored the state before: ${res.undone}` : "Restored." }));
        refreshSoon();
      } catch (err) {
        undo.disabled = false;
        head.append(h("span", { class: "form-error", text: err.message }));
      }
    });
  }
  return card;
}

/* ---------- one question / answer ---------- */

const ERROR_TITLES = {
  quota: "Gemini quota reached", config: "Setup needed", blocked: "Gemini declined to answer",
  model: "Model not available", connection: "Database problem", limit: "Stopped before finishing",
  busy: "Still working", input: "Check your message", unavailable: "Gemini is unavailable", internal: "Something went wrong",
};

function describeStep(tool, args) {
  switch (tool) {
    case "list_tables": return { title: "Listing the tables", code: null };
    case "describe_tables": return { title: `Inspecting ${(args.tables || []).join(", ") || "tables"}`, code: null };
    case "sample_rows": return { title: `Looking at example rows from ${args.table}`, code: null };
    case "run_sql": return { title: "Writing a query", code: typeof args.sql === "string" ? args.sql.trim() : args.sql };
    case "run_aggregation": {
      let code = args.pipeline_json;
      try { code = JSON.stringify(JSON.parse(args.pipeline_json), null, 2); } catch (_) { /* show as sent */ }
      return { title: `Running an aggregation on ${args.collection}`, code };
    }
    case "create_database": return { title: `Creating the database “${args.name}”`, code: null };
    case "modify_database": {
      const list = Array.isArray(args.statements) ? args.statements : [];
      return { title: `Changing the database (${list.length} statement${list.length === 1 ? "" : "s"})`, code: list.map((s) => String(s).trim()).join(";\n") };
    }
    default: return { title: tool, code: null };
  }
}

function newTurn(question) {
  const progressText = h("span", { text: "Starting…" });
  const progress = h("div", { class: "progress", role: "status" }, h("span", { class: "spinner", "aria-hidden": "true" }), progressText);
  const stepList = h("ol", { class: "step-list" });
  const steps = h("details", { class: "card steps", open: true, hidden: true }, h("summary", { text: "What I did" }), stepList);
  const slot = h("div");                                   // result / change cards go here
  const answer = h("div", { class: "card answer", hidden: true });
  const error = h("div", { class: "card error", hidden: true });
  const meta = h("p", { class: "meta muted small", hidden: true });
  const root = h("article", { class: "turn", "aria-busy": "true" },
    h("div", { class: "q" }, h("span", { class: "sr-only", text: "You said:" }), h("p", { text: question })),
    h("div", { class: "a" }, h("span", { class: "sr-only", text: "VerbaDB:" }), progress, steps, slot, answer, error, meta));
  return { root, progress, progressText, steps, stepList, slot, answer, error, meta, pending: new Map(), resultCard: null };
}

function applyEvent(turn, ev) {
  switch (ev.type) {
    case "start":
      turn.progressText.textContent = ev.database ? "Looking at the schema…" : "Thinking…";
      break;
    case "status":
      turn.progressText.textContent = ev.message;
      break;
    case "text":
      turn.steps.hidden = false;
      turn.stepList.append(h("li", {}, h("span", { class: "mark", text: "•" }), h("span", { class: "note", text: ev.text })));
      break;
    case "step": {
      const { title, code } = describeStep(ev.tool, ev.args || {});
      turn.steps.hidden = false;
      const mark = h("span", { class: "mark", text: "…" });
      const detail = h("div", { class: "detail" });
      const li = h("li", {}, mark, h("div", {}, h("div", { text: title }), code ? h("pre", { class: "code", text: code }) : null, detail));
      turn.stepList.append(li);
      turn.pending.set(ev.id, { mark, detail });
      turn.progressText.textContent = title + "…";
      break;
    }
    case "observation": {
      const row = turn.pending.get(ev.id);
      if (row) {
        row.mark.textContent = ev.ok ? "✓" : "✕";
        row.mark.classList.add(ev.ok ? "ok" : "bad");
        row.detail.textContent = ev.summary;
      }
      turn.progressText.textContent = "Thinking…";
      break;
    }
    case "result":
      if (turn.resultCard) turn.resultCard.remove();       // a later query replaces an earlier one
      turn.resultCard = renderResult(ev);
      turn.slot.append(turn.resultCard);
      break;
    case "change":
      turn.slot.append(renderChange(ev, turn));
      refreshSoon();
      break;
    case "database":
      selectDatabase(ev.database.id, { keepChat: true, quiet: true });
      refreshSoon(true);
      break;
    case "final":
      turn.answer.replaceChildren(renderMarkdown(ev.text));
      turn.answer.hidden = false;
      turn.meta.textContent = `${ev.requests} Gemini request${ev.requests === 1 ? "" : "s"} · ${ev.total_tokens.toLocaleString()} tokens`;
      turn.meta.hidden = false;
      break;
    case "error": {
      let message = ev.message;
      if (ev.kind === "quota" && !ev.daily && ev.retry_after) message += ` Try again in about ${Math.ceil(ev.retry_after)} seconds.`;
      turn.error.replaceChildren(h("div", { class: "title", text: ERROR_TITLES[ev.kind] || "Error" }), h("div", { text: message }));
      turn.error.hidden = false;
      break;
    }
    default:
      break;
  }
}

function finishTurn(turn, failed) {
  turn.root.setAttribute("aria-busy", "false");
  turn.progress.remove();
  if (turn.stepList.children.length && !failed) turn.steps.open = false;
}

/* ---------- streaming ---------- */

async function streamAsk(question, onEvent, signal) {
  const resp = await fetch("/api/ask", {
    method: "POST", signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, database_id: state.dbId, question }),
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try { const body = await resp.json(); detail = typeof body.detail === "string" ? body.detail : "The request was rejected."; } catch (_) { /* keep */ }
    throw new Error(detail);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data:")) onEvent(JSON.parse(line.slice(5).trim()));
      }
    }
  }
}

function setBusy(busy) {
  state.busy = busy;
  $("send").textContent = busy ? "Stop" : "Send";
  $("send").classList.toggle("danger", busy);
  $("send").classList.toggle("primary", !busy);
  $("question").disabled = busy;
}

function nearBottom(feed) { return feed.scrollHeight - feed.scrollTop - feed.clientHeight < 140; }

async function ask(question) {
  question = question.trim();
  if (!question || state.busy) return;
  const feed = $("feed");
  $("empty")?.remove();
  const turn = newTurn(question);
  feed.append(turn.root);
  feed.scrollTop = feed.scrollHeight;

  setBusy(true);
  state.controller = new AbortController();
  let failed = false;
  try {
    await streamAsk(question, (ev) => {
      if (ev.type === "error") failed = true;
      const stick = nearBottom(feed);
      applyEvent(turn, ev);
      if (stick) feed.scrollTop = feed.scrollHeight;
    }, state.controller.signal);
  } catch (err) {
    failed = true;
    applyEvent(turn, err.name === "AbortError"
      ? { type: "error", kind: "input", message: "Stopped." }
      : { type: "error", kind: "internal", message: err.message || "The connection to the server was lost." });
  } finally {
    finishTurn(turn, failed);
    setBusy(false);
    state.controller = null;
    loadHistory();
    $("question").focus();
  }
}

/* ---------- empty states ---------- */

function chipsFor(db, schemaNames) {
  if (!db) return BUILDER_EXAMPLES;
  if (db.writable) return SANDBOX_EXAMPLES;
  if (schemaNames == null) return [];             // schema still loading: show no chips yet rather than swapping them later
  const names = schemaNames.map((n) => n.toLowerCase());
  return names.includes("orders") && names.includes("customers") ? ANALYTICS_EXAMPLES : GENERIC_EXAMPLES;
}

function buildEmpty() {
  const db = currentDb();
  const chips = h("div", { class: "chips", id: "chips" }, chipsFor(db, state.schemaNames).map((text) => {
    const chip = h("button", { class: "chip", type: "button", text });
    chip.addEventListener("click", () => ask(text));
    return chip;
  }));
  const [title, sub] = !db
    ? ["Describe a database and I'll build it",
       "Tell me what you want to store, for example a bookstore, a clinic or a school. I create a sandbox database inside this app, fill it with sample data, and show you exactly what I changed. You can undo any change."]
    : db.writable
      ? [`Working in “${db.name}”`, "Ask questions, or ask me to change it — add tables, insert, update or delete rows. Every change is shown before/after, and you can undo it."]
      : [`Asking “${db.name}”`, `Ask in plain English. The agent looks at the schema, writes a read-only query, runs it, and explains the result. Nothing here can be changed.`];
  return h("div", { id: "empty", class: "empty" }, h("h2", { text: title }), h("p", { class: "muted", text: sub }), chips);
}

function showEmpty() { $("feed").replaceChildren(buildEmpty()); }

/* ---------- databases ---------- */

function currentDb() { return state.databases.find((d) => d.id === state.dbId) || null; }

function kindLabel(db) {
  return db.kind === "sandbox" ? "Sandbox" : db.kind === "shared" ? "Shared" : "Yours";
}

function renderActive() {
  const db = currentDb();
  const box = $("active");
  document.title = db ? `${db.name} · VerbaDB` : "VerbaDB: talk to your databases";
  if (!db) {
    box.replaceChildren(h("span", { class: "muted", text: "No database yet — describe one below" }));
    $("question").placeholder = "Describe the database you want, e.g. “a library with books, members and loans”…";
    return;
  }
  const e = state.details && state.details.id === db.id ? state.details.enforcement : null;
  // If the database's own read-only session could not be verified, say so: only the query guard is protecting it.
  const guardOnly = !db.writable && e && e.session_read_only !== "verified";
  const badge = h("span", {
    class: `badge ${db.writable ? "rw" : "ro"}`,
    text: db.writable ? "Sandbox · read/write" : guardOnly ? "Read-only · guard only" : "Read-only",
  });
  if (e) badge.title = [`Query guard: ${e.guard ? "on" : "off"}`, `Read-only session: ${e.session_read_only}`,
    `Statement timeout: ${e.statement_timeout}`, ...(e.notes || [])].join("\n");
  box.replaceChildren(h("span", { class: "name", text: db.name, title: db.name }), badge);
  $("question").placeholder = db.writable ? "Ask a question or describe a change…" : "Ask a question about your data…";
}

function renderDatabases() {
  const list = $("db-list");
  const items = [];

  const none = h("li", { class: "db-item", "aria-current": state.dbId === null ? "true" : null },
    h("button", { class: "pick", type: "button" },
      h("span", { class: "name", text: "New database from a description" }),
      h("span", { class: "meta" }, h("span", { class: "muted small", text: "Chat without a database; I'll create one for you" }))));
  none.querySelector("button").addEventListener("click", () => selectDatabase(null));
  items.push(none);

  for (const db of state.databases) {
    const active = db.id === state.dbId;
    const pick = h("button", { class: "pick", type: "button" },          // its visible text (name, kind, engine) is its name
      h("span", { class: "name", text: db.name }),
      h("span", { class: "meta" },
        h("span", { class: `badge ${db.writable ? "rw" : "ro"}`, text: db.writable ? "Sandbox · read/write" : `${kindLabel(db)} · read-only` }),
        h("span", { class: "muted small", text: db.backend })));
    pick.addEventListener("click", () => selectDatabase(db.id));
    const li = h("li", { class: "db-item", "aria-current": active ? "true" : null }, pick);

    if (active) {
      const actions = h("div", { class: "db-actions" });
      if (db.kind === "sandbox") {
        const depth = db.undo_depth || 0;
        const undo = h("button", { class: "btn small", type: "button", text: depth ? `Undo (${depth})` : "Undo" });
        undo.disabled = !depth;
        undo.addEventListener("click", () => undoLast(db));
        const reset = h("button", { class: "btn small", type: "button", text: "Reset" });
        reset.addEventListener("click", () => resetDb(db));
        actions.append(undo, reset);
      }
      if (db.kind !== "shared") {
        const del = h("button", { class: "btn small danger", type: "button", text: db.kind === "sandbox" ? "Delete" : "Remove" });
        del.addEventListener("click", () => deleteDb(db));
        actions.append(del);
      }
      if (db.url) actions.append(h("span", { class: "muted small", text: db.url }));
      if (actions.children.length) li.append(actions);
    }
    items.push(li);
  }
  list.replaceChildren(...items);
}

let databasesRequest = 0;       // responses can arrive out of order: only the newest request may update the page

async function loadDatabases() {
  const mine = ++databasesRequest;
  let fetched;
  try {
    fetched = await api("GET", "/api/databases");
  } catch (err) {
    showSetup(`Could not load your databases: ${err.message}`);
    return;
  }
  if (mine !== databasesRequest) return;
  state.databases = fetched;
  if (state.dbId && !currentDb()) { state.dbId = null; remember("verbadb.db", null); }
  renderDatabases();
  renderActive();
}

async function loadDetails() {
  state.details = null;
  if (!state.dbId) { renderActive(); return; }
  try {
    const details = await api("GET", `/api/databases/${encodeURIComponent(state.dbId)}`);
    if (details.id === state.dbId) state.details = details;
  } catch (_) { /* the badge simply has no tooltip */ }
  renderActive();
}

async function selectDatabase(id, opts = {}) {
  const changed = id !== state.dbId;
  state.dbId = id;
  remember("verbadb.db", id);
  if (changed && !opts.keepChat) {
    if (state.controller) state.controller.abort();
    fetch(`/api/sessions/${state.sessionId}/reset`, { method: "POST" }).catch(() => {});
    state.sessionId = newSessionId();
  }
  state.schemaNames = null;                                  // unknown until the schema has loaded
  renderDatabases();
  renderActive();
  if (changed && !opts.keepChat) showEmpty();                // start the new chat right away, not after the loads
  await Promise.all([loadSchema(), loadDetails(), loadDatabases().then(() => null)]);
  // The schema is known now: refresh the example chips, but only if nobody has started chatting meanwhile.
  if (changed && !opts.keepChat && state.dbId === id && $("empty")) showEmpty();
  if (document.getElementById("tab-history").getAttribute("aria-selected") === "true" && $("history-this").checked) loadHistory();
  if (!opts.quiet) setMenu(false);
}

function refreshSoon(immediate) {
  clearTimeout(state.refreshTimer);
  const run = async () => {
    await loadDatabases();
    if (state.dbId) { await Promise.all([loadSchema(), loadDetails()]); }
  };
  if (immediate) run();
  else state.refreshTimer = setTimeout(run, 150);
}

async function undoLast(db) {
  try {
    const res = await api("POST", `/api/databases/${encodeURIComponent(db.id)}/undo`);
    noteInFeed(`Undid the last change${res.undone ? `: ${res.undone}` : ""}.`);
  } catch (err) {
    noteInFeed(err.message, true);
  }
  refreshSoon(true);
}

async function resetDb(db) {
  if (!confirm(`Remove all tables and data from “${db.name}”? You can still undo this afterwards.`)) return;
  try {
    await api("POST", `/api/databases/${encodeURIComponent(db.id)}/reset`);
    noteInFeed(`“${db.name}” was emptied. Use Undo to bring it back.`);
  } catch (err) {
    noteInFeed(err.message, true);
  }
  refreshSoon(true);
}

async function deleteDb(db) {
  const what = db.kind === "sandbox" ? "Delete this sandbox database and its undo history?" : "Remove this connection? The database itself is not touched.";
  if (!confirm(`${what}\n\n${db.name}`)) return;
  try {
    await api("DELETE", `/api/databases/${encodeURIComponent(db.id)}`);
  } catch (err) {
    noteInFeed(err.message, true);
    return;
  }
  await selectDatabase(null);
}

function noteInFeed(text, bad) {
  $("empty")?.remove();
  const feed = $("feed");
  feed.append(h("p", { class: `card ${bad ? "error" : "answer"}`, text }));
  feed.scrollTop = feed.scrollHeight;
}

/* ---------- schema ---------- */

async function loadSchema() {
  const list = $("schema-list");
  if (!state.dbId) {
    list.textContent = "Select a database to see its tables.";
    list.className = "schema-list muted small";
    return;
  }
  try {
    const data = await api("GET", `/api/schema/${encodeURIComponent(state.dbId)}`);
    if (data.id !== state.dbId) return;
    state.schemaNames = data.tables.map((t) => t.name);
    list.className = "schema-list";
    if (!data.tables.length) { list.textContent = "No tables yet."; list.className = "schema-list muted small"; return; }
    list.replaceChildren(...data.tables.map((t) => h("details", {},
      h("summary", {}, t.name, h("span", { class: "count", text: `${t.columns.length} cols` })),
      h("ul", {}, t.columns.map((c) => h("li", {},
        h("span", {}, c.name, c.primary_key ? h("span", { class: "pk", text: "PK" }) : null),
        h("span", { class: "type", text: c.type || "" })))))));
  } catch (err) {
    if (state.schemaNames == null) state.schemaNames = [];   // unknown after all: stop waiting for it
    list.className = "schema-list muted small";
    list.textContent = err.message;
  }
}

/* ---------- history ---------- */

let historyTimer = null;
let historyRequest = 0;         // a slow, older response must never redraw over a newer one (e.g. a prompt just deleted)

async function loadHistory() {
  const mine = ++historyRequest;
  const params = new URLSearchParams();
  const q = $("history-search").value.trim();
  if (q) params.set("q", q);
  if ($("history-this").checked && state.dbId) params.set("database_id", state.dbId);
  let rows = [];
  try { rows = await api("GET", `/api/history?${params}`); } catch (_) { /* leave the list as it was */ return; }
  if (mine !== historyRequest) return;
  const list = $("history-list");
  if (!rows.length) {
    list.replaceChildren(h("li", { class: "muted small", text: q ? "No prompts match." : "Your prompts will appear here." }));
    return;
  }
  list.replaceChildren(...rows.map((row) => {
    const pick = h("button", { class: "pick", type: "button", title: row.summary || row.text },
      h("span", { class: "text", text: row.text }),
      h("span", { class: "meta" },
        h("span", { class: `st-${row.status}`, text: row.status === "ok" ? "✓ answered" : row.status === "error" ? "✕ failed" : "■ stopped" }),
        row.database_name ? h("span", { text: row.database_name }) : null,
        h("span", { text: timeAgo(row.created_at) })));
    pick.addEventListener("click", () => reuse(row, false));
    const again = h("button", { class: "btn small", type: "button", text: "Ask again" });
    again.addEventListener("click", () => reuse(row, true));
    const del = h("button", { class: "btn small", type: "button", text: "Delete", "aria-label": `Delete prompt: ${row.text}` });
    del.addEventListener("click", async () => {
      try { await api("DELETE", `/api/history/${row.id}`); } catch (_) { /* already gone */ }
      loadHistory();
    });
    return h("li", { class: "hist-item" }, pick, h("div", { class: "hist-actions" }, again, del));
  }));
}

async function reuse(row, send) {
  if (state.busy) return;
  if (row.database_id !== state.dbId && (row.database_id === null || state.databases.some((d) => d.id === row.database_id))) {
    await selectDatabase(row.database_id, { quiet: true });
  }
  setMenu(false);
  if (send) { ask(row.text); return; }
  const box = $("question");
  box.value = row.text;
  box.dispatchEvent(new Event("input"));
  box.focus();
}

/* ---------- tabs and dialogs ---------- */

const TABS = ["databases", "history", "schema"];

function showTab(name, moveFocus) {
  for (const tab of TABS) {
    const selected = tab === name;
    const button = $(`tab-${tab}`);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;                    // roving tabindex: one Tab stop for the whole tab list
    $(`panel-${tab}`).hidden = !selected;
  }
  if (moveFocus) $(`tab-${name}`).focus();
  if (name === "history") loadHistory();
  if (name === "schema") loadSchema();
}

/* WAI-ARIA tabs pattern: arrow keys, Home and End move between the tabs */
function onTabKey(event) {
  const index = TABS.indexOf(event.currentTarget.id.replace("tab-", ""));
  let next = null;
  if (event.key === "ArrowRight") next = (index + 1) % TABS.length;
  else if (event.key === "ArrowLeft") next = (index + TABS.length - 1) % TABS.length;
  else if (event.key === "Home") next = 0;
  else if (event.key === "End") next = TABS.length - 1;
  if (next === null) return;
  event.preventDefault();
  showTab(TABS[next], true);
}

/* a form error is shown, announced (role="alert") and tied to the field it is about */
function fieldError(input, errorBox, message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  input.setAttribute("aria-invalid", "true");
  input.focus();
}

function clearFieldError(input, errorBox) {
  errorBox.hidden = true;
  input.removeAttribute("aria-invalid");
}

function openNewDialog() {
  clearFieldError($("new-name"), $("new-error"));
  $("new-name").value = "My database";
  document.querySelector('input[name="start"][value="empty"]').checked = true;
  $("dlg-new").showModal();
  $("new-name").select();
}

function openConnectDialog() {
  clearFieldError($("connect-url"), $("connect-error"));
  $("connect-name").value = "";
  $("connect-url").value = "";
  $("dlg-connect").showModal();
  $("connect-name").focus();
}

function showSetup(text) {
  $("setup").hidden = false;
  $("setup").textContent = text;
}

function renderPrivacy(health) {
  const box = $("privacy");
  const rows = health.preview_rows;
  const redact = (health.redact_columns || []).join(", ");
  const shared = rows > 0
    ? `the schema and up to ${rows} result rows per query` + (redact ? ` (values in columns matching ${redact} are masked)` : "")
    : "the schema and aggregate counts (no row values)";
  const detail = ` Shared with Google: ${shared}. Use non-sensitive data, or a paid-tier key for anything private.`;
  box.title = "Free-tier Gemini: your data is sent to Google and may be used to improve its products." + detail;
  box.replaceChildren(
    h("strong", { text: "Free-tier Gemini: " }),
    "your messages, schema and query results are sent to Google and may be used to improve its products.",
    h("span", { class: "detail", text: detail }),
  );
}

function renderHealth(health) {
  state.health = health;
  $("model").textContent = health.model || "";
  renderPrivacy(health);
  let problem = null;
  if (!health.llm_ready) problem = health.message || "GEMINI_API_KEY is not set.";
  else if (health.model_ok === false) problem = health.model_detail;
  $("setup").hidden = !problem;
  if (problem) $("setup").textContent = problem;
  $("send").disabled = !health.llm_ready;
  $("side-connect").hidden = !health.allow_user_connections;
}

function resetChat() {
  if (state.controller) state.controller.abort();
  fetch(`/api/sessions/${state.sessionId}/reset`, { method: "POST" }).catch(() => {});
  state.sessionId = newSessionId();
  showEmpty();
}

async function init() {
  renderTheme();
  $("theme-toggle").addEventListener("click", () => setTheme(effectiveTheme() === "dark" ? "light" : "dark"));
  $("theme-auto").addEventListener("click", () => { setTheme(null); $("theme-toggle").focus(); });
  if (systemDark) systemDark.addEventListener("change", renderTheme);   // while following the system, track it live

  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.busy) { state.controller?.abort(); return; }
    const box = $("question");
    const text = box.value;
    box.value = "";
    box.style.height = "";
    ask(text);
  });
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  $("question").addEventListener("input", (e) => {
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  });
  $("new-chat").addEventListener("click", resetChat);
  $("menu-toggle").addEventListener("click", () => {
    const open = !$("side").classList.contains("open");
    setMenu(open);
    if (open) $(`tab-${TABS.find((t) => $(`tab-${t}`).getAttribute("aria-selected") === "true")}`).focus();
  });
  document.addEventListener("keydown", (e) => {                         // Escape closes the drawer and returns focus to its button
    if (e.key === "Escape" && $("side").classList.contains("open") && !document.querySelector("dialog[open]")) {
      setMenu(false);
      $("menu-toggle").focus();
    }
  });
  for (const tab of TABS) {
    $(`tab-${tab}`).addEventListener("click", () => showTab(tab));
    $(`tab-${tab}`).addEventListener("keydown", onTabKey);
  }
  $("history-search").addEventListener("input", () => { clearTimeout(historyTimer); historyTimer = setTimeout(loadHistory, 200); });
  $("history-this").addEventListener("change", loadHistory);
  $("history-clear").addEventListener("click", async () => {
    if (!confirm("Delete all your saved prompts?")) return;
    try { await api("DELETE", "/api/history"); } catch (_) { /* nothing to clear */ }
    loadHistory();
  });

  for (const id of ["new-db-btn", "side-new-db"]) $(id).addEventListener("click", openNewDialog);
  $("side-connect").addEventListener("click", openConnectDialog);
  $("new-cancel").addEventListener("click", () => $("dlg-new").close());
  $("connect-cancel").addEventListener("click", () => $("dlg-connect").close());

  $("new-name").addEventListener("input", () => clearFieldError($("new-name"), $("new-error")));
  $("connect-url").addEventListener("input", () => clearFieldError($("connect-url"), $("connect-error")));
  $("connect-name").addEventListener("input", () => clearFieldError($("connect-url"), $("connect-error")));

  $("form-new").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("new-create").disabled = true;
    clearFieldError($("new-name"), $("new-error"));
    try {
      const sample = document.querySelector('input[name="start"]:checked').value === "sample";
      const info = await api("POST", "/api/databases", { kind: "sandbox", name: $("new-name").value, sample });
      $("dlg-new").close();
      await loadDatabases();
      await selectDatabase(info.id);
    } catch (err) {
      fieldError($("new-name"), $("new-error"), err.message);
    } finally {
      $("new-create").disabled = false;
    }
  });

  $("form-connect").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("connect-save").disabled = true;
    $("connect-save").textContent = "Connecting…";
    clearFieldError($("connect-url"), $("connect-error"));
    try {
      const info = await api("POST", "/api/databases", { kind: "connection", name: $("connect-name").value, url: $("connect-url").value });
      $("connect-url").value = "";                            // never keep the password in the page
      $("dlg-connect").close();
      await loadDatabases();
      await selectDatabase(info.id);
    } catch (err) {
      fieldError($("connect-url"), $("connect-error"), err.message);
    } finally {
      $("connect-save").disabled = false;
      $("connect-save").textContent = "Test & add";
    }
  });

  showEmpty();
  try {
    renderHealth(await api("GET", "/api/health"));
  } catch (err) {
    showSetup(`Cannot reach the server: ${err.message}`);
    return;
  }

  await loadDatabases();
  const saved = recall("verbadb.db");
  const first = state.databases.find((d) => d.id === saved) || state.databases.find((d) => d.kind === "sandbox") || state.databases[0];
  await selectDatabase(first ? first.id : null, { quiet: true });
  loadHistory();
  // Deliberately no focus() here: moving focus on page load would make keyboard and screen-reader users skip the
  // header and the skip link. Focus moves only in response to something the person did.
}

init();
