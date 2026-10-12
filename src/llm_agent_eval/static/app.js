"use strict";
const $ = (id) => document.getElementById(id);
let token = "",
  indexes = [],
  active = null,
  lastResult = null,
  polling = false,
  searching = false,
  denseEnabled = false;
const names = {
  bm25: "BM25 + path prior",
  dense: "CodeBERT · experimental",
  hybrid: "Hybrid RRF · experimental",
};
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function notice(message = "") {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
  if (response.status === 204) return null;
  const body = await response.json();
  if (!response.ok) {
    if (response.status === 401) $("access-dialog").showModal();
    throw Error(
      Array.isArray(body.detail)
        ? body.detail.map((e) => e.msg).join("; ")
        : body.detail || `Request failed (${response.status})`,
    );
  }
  return body;
}
function clearResults() {
  lastResult = null;
  $("results").replaceChildren();
  $("search-summary").hidden = true;
  $("results-note").hidden = true;
  $("export-button").disabled = true;
  $("empty-state").hidden = false;
}
function selectIndex(row) {
  if (active?.id !== row.id) clearResults();
  active = row;
  const currentMethod = $("method").value;
  const methods = row.methods.filter((m) => m === "bm25" || denseEnabled);
  $("method").replaceChildren(
    ...methods.map((method) => {
      const opt = node("option", names[method]);
      opt.value = method;
      return opt;
    }),
  );
  if (methods.includes(currentMethod)) $("method").value = currentMethod;
  $("active-index").className = `active-index ${row.status}`;
  const info = node("div");
  info.append(
    node(
      "strong",
      row.repository === "demo"
        ? "Sample shop · demonstration"
        : row.repository,
    ),
  );
  info.append(
    node(
      "small",
      row.status === "ready"
        ? `${row.files} files · ${row.chunks} chunks · ${row.commit.slice(0, 12)}`
        : row.status === "failed"
          ? row.error
          : `${row.status === "queued" ? "Queued" : "Indexing source"}… you can leave this page open.`,
    ),
  );
  $("active-index").replaceChildren(node("span", "", "status-dot"), info);
  $("search-button").disabled = row.status !== "ready" || searching;
  renderIndexes();
}
function renderIndexes() {
  $("index-list").replaceChildren();
  if (!indexes.length)
    $("index-list").append(
      node("p", "Your repositories will appear here.", "muted small"),
    );
  for (const row of indexes) {
    const card = node(
      "div",
      undefined,
      `index-card ${active?.id === row.id ? "selected" : ""}`,
    );
    const select = node("button", undefined, "index-select");
    select.append(
      node(
        "strong",
        row.repository === "demo" ? "Sample shop" : row.repository,
      ),
      node(
        "small",
        `${row.status} ${row.commit ? "· " + row.commit.slice(0, 7) : ""}`,
      ),
    );
    select.addEventListener("click", () => selectIndex(row));
    card.append(select);
    if (!["queued", "indexing"].includes(row.status)) {
      const remove = node("button", "×", "delete-index");
      remove.setAttribute("aria-label", `Delete index for ${row.repository}`);
      remove.addEventListener("click", async () => {
        if (!confirm(`Delete the saved index for ${row.repository}?`)) return;
        try {
          await api(`/api/indexes/${row.id}`, { method: "DELETE" });
          if (active?.id === row.id) {
            active = null;
            clearResults();
            $("search-button").disabled = true;
            $("active-index").textContent =
              "Select or create a repository index.";
          }
          await refresh();
        } catch (e) {
          notice(e.message);
        }
      });
      card.append(remove);
    }
    $("index-list").append(card);
  }
}
async function refresh() {
  indexes = await api("/api/indexes");
  if (active) {
    const row = indexes.find((r) => r.id === active.id);
    if (row) selectIndex(row);
  } else if (indexes.length)
    selectIndex(indexes.find((r) => r.status === "ready") || indexes[0]);
  renderIndexes();
}
$("index-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  notice();
  $("index-button").disabled = true;
  $("search-button").disabled = true;
  try {
    const row = await api("/api/indexes", {
      method: "POST",
      body: JSON.stringify({
        repository: $("repository").value,
        revision: $("revision").value,
        dense: $("dense-index").checked,
      }),
    });
    await refresh();
    selectIndex(row);
  } catch (e) {
    notice(e.message);
  } finally {
    $("index-button").disabled = false;
    $("search-button").disabled = active?.status !== "ready" || searching;
  }
});
$("demo-button").addEventListener("click", async () => {
  notice();
  $("demo-button").disabled = true;
  $("search-button").disabled = true;
  $("query").value =
    "Cancelling an order does not restore inventory. Stock stays reserved after the order is cancelled.";
  try {
    const row = await api("/api/demo", { method: "POST" });
    await refresh();
    selectIndex(indexes.find((r) => r.id === row.id) || row);
  } catch (e) {
    notice(e.message);
  } finally {
    $("demo-button").disabled = false;
    $("search-button").disabled = active?.status !== "ready" || searching;
  }
});
function codeBlock(snippet) {
  const pre = node("pre");
  const terms = new Set(snippet.matched_terms);
  snippet.text.split("\n").forEach((line, i) => {
    const row = node("span", undefined, "code-line");
    row.append(node("span", String(snippet.start_line + i), "line-number"));
    for (const part of line.split(/([A-Za-z_][A-Za-z0-9_]*|\d+)/)) {
      row.append(
        terms.has(part.toLowerCase())
          ? node("mark", part)
          : document.createTextNode(part),
      );
    }
    pre.append(row);
  });
  return pre;
}
function renderResults(result) {
  lastResult = result;
  $("empty-state").hidden = true;
  $("results").replaceChildren();
  $("search-summary").replaceChildren();
  for (const label of [
    `${result.files.length} files`,
    `${result.retrieval_ms} ms retrieval`,
    `${result.selected_characters.toLocaleString()} / ${result.context_budget_chars.toLocaleString()} chars`,
    names[result.method],
  ])
    $("search-summary").append(node("span", label, "metric-pill"));
  $("search-summary").hidden = false;
  if (!result.files.length) {
    const empty = node("div", undefined, "panel");
    empty.append(
      node("h3", "No code selected"),
      node(
        "p",
        "Try specific identifiers or error messages, or increase the context budget.",
        "muted",
      ),
    );
    $("results").append(empty);
  }
  result.files.forEach((file, i) => {
    const card = node("details", undefined, "result-card");
    card.open = i === 0;
    const summary = node("summary");
    const title = node("div");
    title.append(
      node("span", file.path, "result-name"),
      node(
        "small",
        `Ranking score ${file.score.toFixed(3)} · ${file.snippets.length} source excerpt${file.snippets.length === 1 ? "" : "s"}`,
        "result-score",
      ),
    );
    summary.append(node("span", String(i + 1).padStart(2, "0"), "rank"), title);
    card.append(summary);
    for (const snippet of file.snippets) {
      const section = node("div", undefined, "snippet");
      const heading = node("div", undefined, "snippet-heading");
      heading.append(
        node("span", `Lines ${snippet.start_line}–${snippet.end_line}`),
      );
      if (snippet.source_url) {
        const link = node("a", "View at commit ↗");
        link.href = snippet.source_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        heading.append(link);
      } else heading.append(node("span", "Bundled demo source"));
      const evidence = node("div", "Matching terms: ", "evidence");
      evidence.append(
        node(
          "strong",
          snippet.matched_terms.join(" · ") ||
            "No exact term matches; inspect the semantic result.",
        ),
      );
      section.append(heading, codeBlock(snippet), evidence);
      card.append(section);
    }
    $("results").append(card);
  });
  $("results-note").textContent =
    `${result.explanation} Budget counts all selected chunks, including files beyond the displayed top five; overlapping lines can repeat.`;
  $("results-note").hidden = false;
  $("export-button").disabled = false;
}
$("search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!active || active.status !== "ready") return;
  notice();
  searching = true;
  $("search-button").disabled = true;
  $("search-button").textContent = "Searching source…";
  const indexId = active.id;
  try {
    const result = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({
        index_id: indexId,
        query: $("query").value,
        method: $("method").value,
        context_budget_chars: Number($("budget").value),
      }),
    });
    if (active?.id === indexId) renderResults(result);
  } catch (e) {
    notice(e.message);
  } finally {
    searching = false;
    $("search-button").disabled = active?.status !== "ready";
    $("search-button").textContent = "Find relevant code →";
  }
});
$("export-button").addEventListener("click", () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(lastResult, null, 2)], {
      type: "application/json",
    }),
  );
  const a = node("a");
  a.href = url;
  a.download = `repolens-${lastResult.commit.slice(0, 12)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function evaluation() {
  try {
    const data = await api("/api/evaluation"),
      b = data.baseline,
      a = b.aggregate;
    const container = $("evaluation-content");
    container.replaceChildren();
    const metrics = node("div", undefined, "metrics-grid");
    for (const [label, value, note] of [
      [
        "File Recall@5",
        a.mean_recall_at_k.toFixed(3),
        "Relevant files found in top five",
      ],
      [
        "Mean reciprocal rank",
        a.mean_reciprocal_rank.toFixed(3),
        "Position of the first relevant file",
      ],
      [
        "nDCG@5",
        a.mean_ndcg_at_k.toFixed(3),
        "Ranking quality across relevant files",
      ],
      [
        "Completed tasks",
        `${b.n_successful} / ${b.n_requested}`,
        "BM25 + filename/path prior",
      ],
    ]) {
      const card = node("div", undefined, "metric-card");
      card.append(
        node("span", label),
        node("strong", value),
        node("small", note),
      );
      metrics.append(card);
    }
    container.append(metrics);
    const columns = node("div", undefined, "eval-columns"),
      protocol = node("div", undefined, "panel"),
      caveats = node("div", undefined, "panel");
    protocol.append(
      node("p", "EXPERIMENT PROTOCOL", "eyebrow"),
      node("h2", data.label),
      node(
        "p",
        "Each issue is searched against its repository before the fix. Reference patches provide scoring labels only. All candidates are packed under the same 40,000-character budget.",
      ),
    );
    const source = node("a", "Inspect the raw result artifact ↗");
    source.href = data.source;
    source.target = "_blank";
    source.rel = "noopener";
    protocol.append(source);
    caveats.append(
      node("p", "WHAT THIS DOES AND DOESN'T SHOW", "eyebrow"),
      node("h2", "Keep the claims proportional"),
    );
    const list = node("ul");
    data.limitations.forEach((text) => list.append(node("li", text)));
    caveats.append(list);
    columns.append(protocol, caveats);
    container.append(columns);
    const tablePanel = node("div", undefined, "panel table-wrap");
    tablePanel.append(node("h2", "Every task, including the misses"));
    const table = node("table"),
      head = node("thead"),
      hr = node("tr");
    ["Instance", "Recall@5", "Reciprocal rank", "nDCG@5"].forEach((t) =>
      hr.append(node("th", t)),
    );
    head.append(hr);
    table.append(head);
    const body = node("tbody");
    b.tasks.forEach((task) => {
      const tr = node("tr");
      tr.append(node("td", task.task_id));
      const recall = node("td");
      const track = node(
        "span",
        undefined,
        `score-track ${task.recall_at_k === 1 ? "full" : task.recall_at_k === 0 ? "zero" : "half"}`,
      );
      track.append(node("span", "", "score-fill"));
      recall.append(
        track,
        document.createTextNode(task.recall_at_k.toFixed(2)),
      );
      tr.append(
        recall,
        node("td", task.reciprocal_rank.toFixed(3)),
        node("td", task.ndcg_at_k.toFixed(3)),
      );
      body.append(tr);
    });
    table.append(body);
    tablePanel.append(table);
    container.append(tablePanel);
  } catch (e) {
    notice(e.message);
  }
}
function tab(lab) {
  $("search-view").hidden = lab;
  $("evaluation-view").hidden = !lab;
  $("search-tab").classList.toggle("active", !lab);
  $("eval-tab").classList.toggle("active", lab);
  $("search-tab").setAttribute("aria-current", lab ? "false" : "page");
  $("eval-tab").setAttribute("aria-current", lab ? "page" : "false");
  $("page-name").textContent = lab ? "Evaluation lab" : "Investigate";
  notice();
  if (lab) evaluation();
}
$("search-tab").addEventListener("click", () => tab(false));
$("eval-tab").addEventListener("click", () => tab(true));
$("refresh").addEventListener("click", () =>
  refresh().catch((e) => notice(e.message)),
);
$("access-button").addEventListener("click", () =>
  $("access-dialog").showModal(),
);
$("close-access").addEventListener("click", () => $("access-dialog").close());
$("access-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  token = $("token").value;
  $("token").value = "";
  $("access-dialog").close();
  await connect();
});
async function connect() {
  try {
    const config = await api("/api/config");
    denseEnabled = config.dense_enabled;
    $("dense-option").hidden = !denseEnabled;
    $("connection").textContent = "Server connected";
    notice();
    await refresh();
  } catch (e) {
    $("connection").textContent = "Connection required";
    notice(e.message);
  }
}
setInterval(async () => {
  if (
    polling ||
    !indexes.some((r) => ["queued", "indexing"].includes(r.status))
  )
    return;
  polling = true;
  try {
    await refresh();
  } catch (e) {
    notice(e.message);
  } finally {
    polling = false;
  }
}, 1500);
connect();
