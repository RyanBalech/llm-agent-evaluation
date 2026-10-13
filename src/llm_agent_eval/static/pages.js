/* Static hosting transport. Search and generation stay in this browser. */
"use strict";
(() => {
  let worker,
    next = 0,
    rowsPromise,
    evaluationPromise;
  const pending = new Map(),
    jobs = new Map();
  const read = async (path) => {
    const response = await fetch(path);
    if (!response.ok)
      throw Error(`Could not load ${path} (${response.status})`);
    return response.json();
  };
  const rows = () => (rowsPromise ||= read("./indexes.json"));
  function call(action, payload) {
    if (!worker) {
      worker = new Worker(
        new URL(
          "./pages-worker.js",
          document.currentScript?.src ||
            new URL("./static/pages.js", location.href),
        ),
        { type: "module" },
      );
      worker.onmessage = ({ data }) => {
        if (data.progress) {
          const job = jobs.get(data.job_id);
          if (job) job.message = data.progress;
          return;
        }
        const handler = pending.get(data.id);
        if (!handler) return;
        pending.delete(data.id);
        data.error
          ? handler.reject(Error(data.error))
          : handler.resolve(data.result);
      };
      worker.onerror = (event) => {
        pending.forEach((p) =>
          p.reject(Error(event.message || "Browser worker failed")),
        );
        pending.clear();
        worker.terminate();
        worker = null;
      };
    }
    return new Promise((resolve, reject) => {
      const id = ++next;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, action, payload });
    });
  }
  window.repoLensPages = async (path, options = {}) => {
    if (path === "/api/config")
      return {
        public_demo: true,
        browser_mode: true,
        dense_enabled: false,
        explanations_enabled: !!navigator.gpu,
      };
    if (path === "/api/indexes") return await rows();
    if (path === "/api/demo")
      return (await rows()).find((r) => r.repository === "demo");
    if (path.startsWith("/api/indexes/"))
      return (await rows()).find((r) => r.id === path.split("/").pop());
    if (path === "/api/evaluation")
      return (evaluationPromise ||= read("./evaluation.json"));
    if (path === "/api/search") return call("search", JSON.parse(options.body));
    if (path === "/api/explanations") {
      if ([...jobs.values()].some((j) => j.status === "generating"))
        throw Error("An explanation is already running.");
      const job = {
        id: crypto.randomUUID(),
        status: "generating",
        message:
          "Loading the local browser model. The first download is approximately 2 GB…",
      };
      jobs.set(job.id, job);
      call("explain", { ...JSON.parse(options.body), job_id: job.id }).then(
        (result) => Object.assign(job, { status: "ready", result }),
        (error) =>
          Object.assign(job, { status: "failed", error: error.message }),
      );
      return { ...job };
    }
    if (path.startsWith("/api/explanations/"))
      return { ...jobs.get(path.split("/").pop()) };
    throw Error(
      "This operation requires the Python server. The public demo searches prepared snapshots.",
    );
  };
})();
