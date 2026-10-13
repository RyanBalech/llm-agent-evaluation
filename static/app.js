"use strict";
const $ = (id) => document.getElementById(id);
let token = "",
  indexes = [],
  active = null,
  lastResult = null,
  polling = false,
  searching = false,
  denseEnabled = false,
  publicDemo = false,
  explanationsEnabled = false;
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
  if (window.repoLensPages) return window.repoLensPages(path, options);
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
  $("explanation").hidden = true;
  $("explain-button").hidden = true;
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
    if (!publicDemo && !["queued", "indexing"].includes(row.status)) {
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
  $("explanation").hidden = true;
  $("explain-button").hidden = !explanationsEnabled || !result.files.length;
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
    for (const study of data.studies || []) {
      const panel = node("div", undefined, "panel table-wrap");
      panel.append(
        node("h2", study.name),
        node(
          "p",
          `${study.n_tasks} identical tasks across ${study.n_repositories} repositories. All three methods use the same source snapshots and context budget.`,
        ),
      );
      const comparison = node("table"),
        heading = node("tr");
      [
        "Method",
        "Recall@5",
        "MRR",
        "nDCG@5",
        "Recall difference vs BM25 (95% paired interval)",
      ].forEach((label) => heading.append(node("th", label)));
      const header = node("thead");
      header.append(heading);
      comparison.append(header);
      const rows = node("tbody");
      for (const method of ["bm25", "dense", "hybrid"]) {
        const metrics = study.aggregates[method],
          row = node("tr");
        const difference = study.paired_vs_bm25[method]?.metrics.recall_at_k;
        const label = { bm25: "BM25", dense: "CodeBERT", hybrid: "Hybrid RRF" }[
          method
        ];
        [
          label,
          metrics.mean_recall_at_k.toFixed(3),
          metrics.mean_reciprocal_rank.toFixed(3),
          metrics.mean_ndcg_at_k.toFixed(3),
          difference
            ? `${difference.mean_difference.toFixed(3)} [${difference.ci_low.toFixed(3)}, ${difference.ci_high.toFixed(3)}]`
            : "Reference",
        ].forEach((value) => row.append(node("td", value)));
        rows.append(row);
      }
      comparison.append(rows);
      panel.append(comparison);
      const notes = node("ul");
      study.interpretation.forEach((text) => notes.append(node("li", text)));
      panel.append(notes);
      const download = node(
        "button",
        "Download complete study JSON",
        "button secondary",
      );
      download.addEventListener("click", () => {
        const url = URL.createObjectURL(
          new Blob([JSON.stringify(study, null, 2)], {
            type: "application/json",
          }),
        );
        const link = node("a");
        link.href = url;
        link.download = `repolens-study-${study.n_tasks}.json`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
      panel.append(download);
      container.append(panel);
    }
    const tablePanel = node("div", undefined, "panel table-wrap");
    tablePanel.append(node("h2", "Every task, including the misses"));
    const table = node("table"),
      head = node("thead"),
      hr = node("tr");
    table.id = "baseline-tasks";
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
    publicDemo = config.public_demo;
    explanationsEnabled = config.explanations_enabled;
    $("index-form").hidden = publicDemo;
    $("public-demo-note").hidden = !publicDemo;
    $("dense-option").hidden = !denseEnabled;
    $("connection").textContent = config.browser_mode
      ? "Browser ready"
      : "Server connected";
    if (config.browser_mode) {
      $("access-button").hidden = true;
      $("public-demo-note").textContent =
        "Search prepared repository snapshots locally in your browser. For arbitrary GitHub indexing and dense retrieval, run the Python app. Optional explanations download a local model (~2 GB) and require WebGPU.";
      $("explain-button").textContent = "Explain locally · download model";
    }
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

$("explain-button").addEventListener("click", async () => {
  if (!lastResult) return;
  const captured = lastResult;
  const button = $("explain-button");
  const box = $("explanation");
  button.disabled = true;
  box.hidden = false;
  box.dataset.status = "generating";
  box.replaceChildren(
    node(
      "p",
      "Generating a cited explanation locally… The first request also loads the model.",
    ),
  );
  try {
    let job = await api("/api/explanations", {
      method: "POST",
      body: JSON.stringify({
        index_id: captured.index_id,
        query: captured.query,
        method: captured.method,
        context_budget_chars: captured.context_budget_chars,
        top_k: captured.configuration.top_k,
      }),
    });
    const deadline = Date.now() + (window.repoLensPages ? 900000 : 300000);
    while (["queued", "generating"].includes(job.status)) {
      if (Date.now() > deadline)
        throw Error(
          "Explanation is taking longer than expected. Try again later.",
        );
      await new Promise((resolve) => setTimeout(resolve, 2000));
      if (lastResult !== captured) return;
      job = await api(`/api/explanations/${job.id}`);
      if (job.message) box.replaceChildren(node("p", job.message));
    }
    if (lastResult !== captured) return;
    if (job.status !== "ready") throw Error(job.error);
    const result = job.result;
    captured.generated_explanation = result;
    box.dataset.status = "ready";
    box.replaceChildren(node("p", "GENERATED ANALYSIS", "eyebrow"));
    for (const finding of result.findings) {
      box.append(node("p", finding.statement));
      for (const citation of finding.evidence) {
        box.append(node("blockquote", citation.quote));
        const source = node(
          citation.source_url ? "a" : "span",
          `${citation.path}:${citation.start_line}–${citation.end_line}`,
          "small",
        );
        if (citation.source_url) {
          source.href = citation.source_url;
          source.target = "_blank";
          source.rel = "noopener noreferrer";
        }
        box.append(source);
      }
    }
    box.append(
      node("h2", "Suggested check"),
      node("p", result.suggested_check),
      node("p", result.notice, "fine-print"),
    );
  } catch (e) {
    if (lastResult === captured) {
      box.dataset.status = "failed";
      box.replaceChildren(node("p", e.message));
    }
  } finally {
    button.disabled = false;
  }
});
