const state = {
  config: null,
  runs: [],
};

const elements = {
  subtitle: document.querySelector("#subtitle"),
  providerBadge: document.querySelector("#provider-badge"),
  modelBadge: document.querySelector("#model-badge"),
  memoryBadge: document.querySelector("#memory-badge"),
  traceBadge: document.querySelector("#trace-badge"),
  task: document.querySelector("#task-input"),
  provider: document.querySelector("#provider-select"),
  model: document.querySelector("#model-input"),
  iterations: document.querySelector("#iterations-input"),
  mcp: document.querySelector("#mcp-input"),
  runButton: document.querySelector("#run-button"),
  runError: document.querySelector("#run-error"),
  metrics: document.querySelector("#metrics"),
  conversation: document.querySelector("#conversation"),
  traceBody: document.querySelector("#trace-body"),
  memoryQuery: document.querySelector("#memory-query"),
  memorySearchButton: document.querySelector("#memory-search-button"),
  memoryRefreshButton: document.querySelector("#memory-refresh-button"),
  memoryList: document.querySelector("#memory-list"),
  historyList: document.querySelector("#history-list"),
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(payload?.error || `HTTP ${response.status}`);
  }
  return payload;
}

function escapeText(value) {
  return String(value ?? "");
}

function showRunError(message) {
  elements.runError.textContent = message;
  elements.runError.classList.remove("hidden");
}

function clearRunError() {
  elements.runError.textContent = "";
  elements.runError.classList.add("hidden");
}

function setBusy(busy) {
  elements.runButton.disabled = busy;
  elements.runButton.textContent = busy ? "运行中..." : "运行任务";
}

function metric(label, value) {
  return `<div class="metric"><strong>${escapeText(value)}</strong><span>${escapeText(label)}</span></div>`;
}

function renderRun(run) {
  const statusClass = run.status === "success" ? "ok" : "fail";
  elements.metrics.innerHTML = [
    metric("状态", run.status),
    metric("迭代", run.iterations),
    metric("Tokens", run.usage?.total ?? 0),
    metric("压缩", run.compressions?.length ?? 0),
  ].join("");

  elements.conversation.classList.remove("empty");
  elements.conversation.innerHTML = `
    <div class="message user">
      <div class="message-label">用户任务</div>
      <div>${escapeText(run.task).replace(/</g, "&lt;")}</div>
    </div>
    <div class="message assistant">
      <div class="message-label">MiniAgent 输出</div>
      <div>${escapeText(run.answer || "无最终输出").replace(/</g, "&lt;")}</div>
    </div>
    <div class="message">
      <div class="message-label">终止原因</div>
      <div class="${statusClass}">${escapeText(run.termination_reason || "-")}</div>
    </div>
  `;
  renderSteps(run.steps || []);
  if (state.config?.traceEnabled) loadHistory().catch(() => {});
}

function renderSteps(steps) {
  if (!steps.length) {
    elements.traceBody.innerHTML = '<tr><td colspan="9" class="empty-cell">暂无轨迹</td></tr>';
    return;
  }
  elements.traceBody.innerHTML = steps.map((step, index) => {
    const args = step.tool_args ? JSON.stringify(step.tool_args, null, 2) : "-";
    const result = step.ok === null ? "-" : (step.ok ? "OK" : "ERROR");
    const resultClass = step.ok === false ? "fail" : (step.ok ? "ok" : "");
    const detail = step.error || step.note || "-";
    return `
      <tr>
        <td>${index + 1}</td>
        <td>${escapeText(step.iteration)}</td>
        <td><span class="state state-${escapeText(step.state)}">${escapeText(step.state)}</span></td>
        <td>${escapeText(step.tool_name || "-")}</td>
        <td><code>${escapeText(args).replace(/</g, "&lt;")}</code></td>
        <td class="${resultClass}">${result}</td>
        <td>${escapeText(step.duration_ms || 0)}ms</td>
        <td>${escapeText(step.tokens || 0)}</td>
        <td>${escapeText(detail).replace(/</g, "&lt;")}</td>
      </tr>
    `;
  }).join("");
}

async function loadConfig() {
  state.config = await requestJson("/api/config");
  const config = state.config;
  elements.subtitle.textContent = `本地可视化 ReAct Agent · ${config.provider} · ${config.model}`;
  elements.providerBadge.textContent = `provider: ${config.provider}`;
  elements.modelBadge.textContent = `model: ${config.model}`;
  elements.memoryBadge.textContent = `memory: ${config.memoryEnabled ? "ON" : "OFF"}`;
  elements.traceBadge.textContent = `trace: ${config.traceEnabled ? "ON" : "OFF"}`;
}

async function runTask() {
  const task = elements.task.value.trim();
  if (!task) {
    showRunError("请输入任务描述。");
    return;
  }
  clearRunError();
  setBusy(true);
  const payload = { task };
  if (elements.provider.value) payload.provider = elements.provider.value;
  if (elements.model.value.trim()) payload.model = elements.model.value.trim();
  if (elements.iterations.value) payload.maxIterations = Number(elements.iterations.value);
  payload.useBuiltinMcp = elements.mcp.checked;

  try {
    const data = await requestJson("/api/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderRun(data.run);
    if (state.config?.memoryEnabled) loadMemories().catch(() => {});
  } catch (error) {
    showRunError(error.message || String(error));
  } finally {
    setBusy(false);
  }
}

async function loadMemories() {
  const data = await requestJson("/api/memories");
  if (!data.memories.length) {
    elements.memoryList.innerHTML = '<li class="empty-cell">暂无记忆</li>';
    return;
  }
  elements.memoryList.innerHTML = data.memories.map((item) => `
    <li>
      <div class="memory-meta"><span>#${item.id}</span><span>${escapeText(item.createdAt || "")}</span></div>
      <div class="memory-content">${escapeText(item.content).replace(/</g, "&lt;")}</div>
      <button class="delete-button" data-id="${item.id}">删除</button>
    </li>
  `).join("");
}

async function searchMemories() {
  const query = elements.memoryQuery.value.trim();
  if (!query) {
    loadMemories().catch(() => {});
    return;
  }
  const data = await requestJson("/api/memories/search", {
    method: "POST",
    body: JSON.stringify({ query, topK: 10 }),
  });
  if (!data.matches.length) {
    elements.memoryList.innerHTML = '<li class="empty-cell">没有匹配记忆</li>';
    return;
  }
  elements.memoryList.innerHTML = data.matches.map((item) => `
    <li>
      <div class="memory-meta"><span>#${item.id}</span><span>distance: ${item.distance.toFixed(4)}</span></div>
      <div class="memory-content">${escapeText(item.content).replace(/</g, "&lt;")}</div>
      <button class="delete-button" data-id="${item.id}">删除</button>
    </li>
  `).join("");
}

async function deleteMemory(id) {
  await requestJson(`/api/memories/${id}`, { method: "DELETE" });
  await loadMemories();
}

async function loadHistory() {
  const data = await requestJson("/api/runs");
  state.runs = data.runs;
  if (!data.runs.length) {
    elements.historyList.innerHTML = '<li class="empty-cell">暂无历史</li>';
    return;
  }
  elements.historyList.innerHTML = data.runs.map((run) => `
    <li class="history-item" data-run-id="${escapeText(run.runId)}">
      <div class="memory-content">${escapeText(run.task).replace(/</g, "&lt;")}</div>
      <div class="history-meta">
        <span class="history-status status-${run.status === "success" ? "success" : "error"}">${escapeText(run.status)}</span>
        <span>${escapeText(run.iterations)} iter</span>
        <span>${escapeText(run.totalTokens)} tokens</span>
      </div>
    </li>
  `).join("");
}

async function loadHistoricalRun(runId) {
  const data = await requestJson(`/api/runs/${encodeURIComponent(runId)}`);
  renderSteps(data.steps || []);
  elements.metrics.innerHTML = [
    metric("历史状态", data.run.status),
    metric("迭代", data.run.iterations),
    metric("Tokens", data.run.total_tokens),
    metric("解析重试", data.run.parse_retries),
  ].join("");
  elements.conversation.classList.remove("empty");
  elements.conversation.innerHTML = `
    <div class="message user">
      <div class="message-label">历史任务</div>
      <div>${escapeText(data.run.task).replace(/</g, "&lt;")}</div>
    </div>
    <div class="message assistant">
      <div class="message-label">当时输出</div>
      <div>${escapeText(data.run.answer || "无最终输出").replace(/</g, "&lt;")}</div>
    </div>
  `;
}

function bindEvents() {
  elements.runButton.addEventListener("click", runTask);
  elements.memorySearchButton.addEventListener("click", () => searchMemories().catch((error) => alert(error.message)));
  elements.memoryRefreshButton.addEventListener("click", () => loadMemories().catch((error) => alert(error.message)));
  elements.memoryList.addEventListener("click", async (event) => {
    const button = event.target.closest(".delete-button");
    if (!button) return;
    try {
      await deleteMemory(Number(button.dataset.id));
    } catch (error) {
      alert(error.message);
    }
  });
  elements.historyList.addEventListener("click", async (event) => {
    const item = event.target.closest(".history-item");
    if (!item) return;
    try {
      await loadHistoricalRun(item.dataset.runId);
    } catch (error) {
      alert(error.message);
    }
  });
  document.querySelectorAll(".quick-actions button").forEach((button) => {
    button.addEventListener("click", () => {
      elements.task.value = button.dataset.example || "";
      elements.mcp.checked = button.dataset.example?.includes("MCP") || false;
    });
  });
}

async function initialize() {
  bindEvents();
  try {
    await loadConfig();
    if (state.config.memoryEnabled) await loadMemories();
    if (state.config.traceEnabled) await loadHistory();
  } catch (error) {
    showRunError(`初始化失败：${error.message}`);
  }
}

initialize();
