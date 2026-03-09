const state = {
  providers: [],
  settings: null,
  selectedMode: "hard",
};

const chatLog = document.querySelector("#chatLog");
const statusBar = document.querySelector("#statusBar");
const providerSelect = document.querySelector("#providerSelect");
const providerKind = document.querySelector("#providerKind");
const baseUrlInput = document.querySelector("#baseUrl");
const modelPlan = document.querySelector("#modelPlan");
const modelExecute = document.querySelector("#modelExecute");
const modelCritique = document.querySelector("#modelCritique");
const modelSuggestions = document.querySelector("#modelSuggestions");
const promptInput = document.querySelector("#promptInput");
const sendButton = document.querySelector("#sendButton");

function renderMath(element) {
  if (!window.renderMathInElement) {
    return;
  }

  window.renderMathInElement(element, {
    throwOnError: false,
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
  });
}

function renderFinalAnswer(element, content, html) {
  if (html) {
    element.innerHTML = html;
    renderMath(element);
    return;
  }

  element.textContent = content || "No answer returned.";
}

function renderStepContent(step) {
  renderFinalAnswer(step.content, step.buffer, step.html);
}

function flushStepRender(step) {
  step.renderScheduled = false;
  if (!step.isExpanded) {
    return;
  }

  renderStepContent(step);
}

function scheduleStepRender(step) {
  if (step.renderScheduled || !step.isExpanded) {
    return;
  }

  step.renderScheduled = true;
  window.requestAnimationFrame(() => flushStepRender(step));
}

function flushFinalAnswerRender(container) {
  container.renderScheduled = false;
  renderFinalAnswer(
    container.finalAnswer,
    container.pendingAnswerContent,
    container.pendingAnswerHtml,
  );
}

function scheduleFinalAnswerRender(container) {
  if (container.renderScheduled) {
    return;
  }

  container.renderScheduled = true;
  window.requestAnimationFrame(() => flushFinalAnswerRender(container));
}

function setStatus(message) {
  statusBar.textContent = message;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderProviders() {
  providerSelect.innerHTML = "";

  state.providers.forEach((provider) => {
    const option = document.createElement("option");
    option.value = provider.name;
    option.textContent = provider.available
      ? `${provider.name} (${provider.models.length} models)`
      : `${provider.name} (offline)`;
    option.dataset.baseUrl = provider.base_url;
    option.dataset.kind = provider.kind;
    providerSelect.appendChild(option);
  });

  if (state.settings?.provider_name) {
    providerSelect.value = state.settings.provider_name;
  }
}

function modelOptionsForSelectedProvider() {
  const selected = state.providers.find(
    (provider) => provider.name === providerSelect.value,
  );
  return selected?.models ?? [];
}

function populateModelInput(inputElement, selectedValue) {
  const models = modelOptionsForSelectedProvider();
  if (models.length) {
    inputElement.placeholder = models[0];
  }
  inputElement.value = selectedValue || models[0] || "";
}

function populateModelSuggestions() {
  const models = modelOptionsForSelectedProvider();
  modelSuggestions.innerHTML = "";

  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    modelSuggestions.appendChild(option);
  });
}

function renderSettings() {
  if (!state.settings) {
    return;
  }

  renderProviders();
  providerKind.value = state.settings.provider_kind;
  baseUrlInput.value = state.settings.base_url;

  populateModelSuggestions();
  populateModelInput(modelPlan, state.settings.model_map.plan);
  populateModelInput(modelExecute, state.settings.model_map.execute);
  populateModelInput(modelCritique, state.settings.model_map.critique);
}

async function loadSettings(endpoint = "/api/settings") {
  const response = await fetch(endpoint, {
    method: endpoint === "/api/settings" ? "GET" : "POST",
  });
  const payload = await response.json();
  state.providers = payload.providers;
  state.settings = payload.settings;
  renderSettings();

  const activeProvider = state.providers.find(
    (provider) => provider.name === state.settings.provider_name,
  );
  if (activeProvider?.available) {
    setStatus(
      `Connected to ${activeProvider.name} at ${activeProvider.base_url}.`,
    );
  } else if (state.providers.length) {
    setStatus(
      "No detected provider is online. You can still point the app at a local endpoint manually.",
    );
  } else {
    setStatus("No providers detected yet.");
  }
}

function syncProviderFields() {
  const selected = state.providers.find(
    (provider) => provider.name === providerSelect.value,
  );
  if (!selected) {
    return;
  }

  providerKind.value = selected.kind;
  baseUrlInput.value = selected.base_url;
  populateModelSuggestions();
  populateModelInput(modelPlan, state.settings?.model_map.plan);
  populateModelInput(modelExecute, state.settings?.model_map.execute);
  populateModelInput(modelCritique, state.settings?.model_map.critique);
}

async function saveSettings() {
  const payload = {
    provider_name: providerSelect.value,
    provider_kind: providerKind.value,
    base_url: baseUrlInput.value,
    model_map: {
      plan: modelPlan.value,
      execute: modelExecute.value,
      critique: modelCritique.value,
    },
  };

  const response = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const updated = await response.json();
  state.providers = updated.providers;
  state.settings = updated.settings;
  setStatus(`Saved settings for ${payload.provider_name || payload.base_url}.`);
  renderSettings();
}

function appendUserMessage(text) {
  const template = document.querySelector("#userMessageTemplate");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector('[data-role="userBody"]').textContent = text;
  chatLog.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

function createAssistantMessage() {
  const template = document.querySelector("#assistantMessageTemplate");
  const node = template.content.firstElementChild.cloneNode(true);
  chatLog.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return {
    root: node,
    finalAnswer: node.querySelector('[data-role="finalAnswer"]'),
    timeline: node.querySelector('[data-role="timeline"]'),
    steps: new Map(),
    pendingAnswerContent: "",
    pendingAnswerHtml: "",
    renderScheduled: false,
  };
}

function createStepCard(container, event) {
  const template = document.querySelector("#stepTemplate");
  const stepNode = template.content.firstElementChild.cloneNode(true);
  stepNode.dataset.step = event.step;
  stepNode.querySelector('[data-role="stepLabel"]').textContent = event.label;
  stepNode.querySelector('[data-role="stepModel"]').textContent = event.model;

  const content = stepNode.querySelector('[data-role="stepContent"]');
  const toggle = stepNode.querySelector('[data-role="stepToggle"]');
  toggle.addEventListener("click", () => {
    content.classList.toggle("collapsed");
    step.isExpanded = !content.classList.contains("collapsed");
    if (step.isExpanded) {
      renderStepContent(step);
    }
  });

  const step = {
    root: stepNode,
    content,
    state: stepNode.querySelector('[data-role="stepState"]'),
    buffer: "",
    html: "",
    isExpanded: false,
    renderScheduled: false,
  };

  container.timeline.appendChild(stepNode);
  return step;
}

function updateStep(container, event) {
  let step = container.steps.get(event.step);
  if (!step && event.type === "step-start") {
    step = createStepCard(container, event);
    container.steps.set(event.step, step);
  }

  if (!step) {
    return;
  }

  if (event.type === "step-delta") {
    step.buffer = event.content ?? `${step.buffer}${event.delta}`;
    step.html = event.html ?? "";
    scheduleStepRender(step);
  }

  if (event.type === "step-complete") {
    step.buffer = event.content;
    step.html = event.html ?? "";
    if (step.renderScheduled) {
      flushStepRender(step);
    } else if (step.isExpanded) {
      renderStepContent(step);
    }
    step.state.textContent = "Done";
  }
}

async function streamChat(message) {
  appendUserMessage(message);
  const assistant = createAssistantMessage();

  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, mode: state.selectedMode }),
  });

  if (!response.ok || !response.body) {
    assistant.finalAnswer.textContent = "The local pipeline failed to start.";
    return;
  }

  assistant.finalAnswer.innerHTML = '<span class="hint">Thinking...</span>';
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalAnswer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    lines.forEach((line) => {
      if (!line.trim()) {
        return;
      }

      const event = JSON.parse(line);

      if (event.type.startsWith("step-")) {
        updateStep(assistant, event);
      }

      if (event.type === "answer-start") {
        assistant.pendingAnswerContent = "";
        assistant.pendingAnswerHtml = "";
        assistant.finalAnswer.textContent = "";
      }

      if (event.type === "answer-delta") {
        finalAnswer = event.content ?? `${finalAnswer}${event.delta}`;
        assistant.pendingAnswerContent = finalAnswer;
        assistant.pendingAnswerHtml = event.html ?? "";
        scheduleFinalAnswerRender(assistant);
      }

      if (event.type === "answer-complete") {
        finalAnswer = event.content;
        assistant.pendingAnswerContent = finalAnswer;
        assistant.pendingAnswerHtml = event.html ?? "";
        if (assistant.renderScheduled) {
          flushFinalAnswerRender(assistant);
        } else {
          renderFinalAnswer(assistant.finalAnswer, finalAnswer, event.html);
        }
      }

      if (event.type === "error") {
        assistant.finalAnswer.textContent = event.message;
      }
    });
  }

  assistant.root.querySelector(".message-meta").textContent =
    state.selectedMode.toUpperCase();
  assistant.root.scrollIntoView({ behavior: "smooth", block: "end" });
}

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll(".mode-button")
      .forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    state.selectedMode = button.dataset.mode;
    setStatus(`Reasoning mode set to ${state.selectedMode}.`);
  });
});

const appShell = document.querySelector("#appShell");
const toggleSidebar = document.querySelector("#toggleSidebar");
if (toggleSidebar && appShell) {
  toggleSidebar.addEventListener("click", () => {
    appShell.classList.toggle("sidebar-collapsed");
  });
}

providerSelect.addEventListener("change", syncProviderFields);
document.querySelector("#saveSettings").addEventListener("click", saveSettings);
document
  .querySelector("#refreshProviders")
  .addEventListener("click", async () => {
    setStatus("Refreshing local providers...");
    await loadSettings("/api/providers/refresh");
  });

document
  .querySelector("#composer")
  .addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = promptInput.value.trim();
    if (!message) {
      return;
    }

    sendButton.disabled = true;
    setStatus(`Running ${state.selectedMode} reasoning pipeline...`);
    try {
      await streamChat(message);
      promptInput.value = "";
      setStatus("Pipeline complete.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Streaming failed.");
    } finally {
      sendButton.disabled = false;
    }
  });

loadSettings().catch((error) => {
  setStatus(
    error instanceof Error ? error.message : "Failed to load settings.",
  );
});
