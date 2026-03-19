const state = {
  providers: [],
  settings: null,
  selectedMode: "off",
  activeTab: "standard",
  activeConversationId: null
};

// Storage Manager
const STORAGE_KEYS = {
  SETTINGS: 'multimind_settings',
  CONVERSATIONS: 'multimind_conversations',
  ACTIVE_CONVERSATION: 'multimind_active_conversation',
  SIDEBAR_OPEN: 'multimind_sidebar_open'
};

function generateId() {
  return crypto.randomUUID();
}

function getSettings() {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.SETTINGS);
    return data ? JSON.parse(data) : null;
  } catch (e) {
    console.warn("Failed to load settings from localStorage", e);
    return null;
  }
}

function saveLocalSettings(settingsPayload) {
  try {
    localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settingsPayload));
  } catch (e) {
    if (e.name === 'QuotaExceededError') {
      setStatus("Storage full. Settings not saved locally.");
    } else {
      console.warn("Failed to save settings to localStorage", e);
    }
  }
}

function getConversations() {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.CONVERSATIONS);
    const convs = data ? JSON.parse(data) : [];
    return convs.sort((a, b) => b.updated_at - a.updated_at);
  } catch (e) {
    console.warn("Failed to load conversations from localStorage", e);
    localStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify([]));
    return [];
  }
}

function getConversation(id) {
  return getConversations().find(c => c.id === id) || null;
}

function saveConversation(conv) {
  try {
    let conversations = getConversations();
    const index = conversations.findIndex(c => c.id === conv.id);
    if (index > -1) {
      conversations[index] = conv;
    } else {
      conversations.push(conv);
    }
    localStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));
  } catch (e) {
    if (e.name === 'QuotaExceededError') {
      setStatus("Storage full. Delete some conversations to save new ones.");
    } else {
      console.warn("Failed to save conversation to localStorage", e);
    }
  }
}

function deleteConversation(id) {
  try {
    let conversations = getConversations();
    conversations = conversations.filter(c => c.id !== id);
    localStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));

    if (state.activeConversationId === id) {
      startNewChat();
    }
    renderConversationList();
  } catch (e) {
    console.warn("Failed to delete conversation", e);
  }
}

function startNewChat() {
  const id = generateId();
  state.activeConversationId = id;
  localStorage.setItem(STORAGE_KEYS.ACTIVE_CONVERSATION, id);
  chatLog.innerHTML = '';

  const newConv = {
    id,
    title: "New Chat",
    created_at: Date.now(),
    updated_at: Date.now(),
    mode: state.selectedMode,
    messages: []
  };

  saveConversation(newConv);
  if (typeof renderConversationList === 'function') {
    renderConversationList();
  }
}

function loadChat(id) {
  const conv = getConversation(id);
  if (!conv) {
    startNewChat();
    return;
  }

  state.activeConversationId = id;
  localStorage.setItem(STORAGE_KEYS.ACTIVE_CONVERSATION, id);
  chatLog.innerHTML = '';

  // Update UI mode
  if (conv.mode) {
    state.selectedMode = conv.mode;

    // Update tabs
    document.querySelectorAll(".tab-button").forEach(btn => btn.classList.remove("active"));
    let tabName = "standard";
    if (conv.mode === "council") tabName = "council";
    if (conv.mode === "org") tabName = "org";

    const tabBtn = document.querySelector(`.tab-button[data-tab="${tabName}"]`);
    if(tabBtn) tabBtn.classList.add("active");

    // Toggle UI panels based on mode (reusing logic from tab click handler)
    if (tabName === "council") {
      standardSettings.style.display = "none";
      councilSettings.style.display = "block";
      orgSettings.style.display = "none";
      modeToggle.style.display = "none";
      heroTitle.textContent = "Consult the Agent Council";
    } else if (tabName === "org") {
      standardSettings.style.display = "none";
      councilSettings.style.display = "none";
      orgSettings.style.display = "block";
      modeToggle.style.display = "none";
      heroTitle.textContent = "Delegate to the Organisation";
    } else {
      standardSettings.style.display = "block";
      councilSettings.style.display = "none";
      orgSettings.style.display = "none";
      modeToggle.style.display = "flex";
      heroTitle.textContent = "Make your local models reason";

      // update mode toggle buttons
      modeToggle.querySelectorAll(".mode-button").forEach(node => node.classList.remove("active"));
      const modeBtn = modeToggle.querySelector(`.mode-button[data-mode="${conv.mode}"]`);
      if(modeBtn) modeBtn.classList.add("active");
    }
  }

  // Replay messages
  conv.messages.forEach(msg => {
    if (msg.role === "user") {
      appendUserMessage(msg.content);
    } else {
      // Simplify replay for assistants - just show final text
      // We don't save the full timeline/org chart in localStorage
      const template = document.querySelector("#assistantMessageTemplate");
      const node = template.content.firstElementChild.cloneNode(true);
      const finalAnswer = node.querySelector('[data-role="finalAnswer"]');
      const html = DOMPurify.sanitize(marked.parse(msg.content));

      node.querySelector('.message-meta').textContent = "SAVED";

      const timeline = node.querySelector('[data-role="timeline"]');
      if (timeline) {
          timeline.remove(); // Remove timeline for saved chats
      }

      // Also handle org chart removal if it's an org template
      const orgChart = node.querySelector('[data-role="orgChart"]');
      if (orgChart) {
          orgChart.remove();
      }

      renderFinalAnswer(finalAnswer, html);
      chatLog.appendChild(node);
    }
  });

  if (chatLog.lastElementChild) {
    chatLog.lastElementChild.scrollIntoView();
  }

  // renderConversationList might not exist yet, so check if it does
  if (typeof renderConversationList === 'function') {
      renderConversationList();
  }
}

function saveMessage(role, content) {
  if (!state.activeConversationId) {
    startNewChat();
  }

  const conv = getConversation(state.activeConversationId);
  if (!conv) return;

  // Set title on first user message
  if (role === "user" && conv.messages.length === 0) {
    conv.title = content.substring(0, 50) + (content.length > 50 ? "..." : "");
  }

  // Update mode to current state
  conv.mode = state.selectedMode;

  conv.messages.push({
    role,
    content,
    timestamp: Date.now()
  });

  conv.updated_at = Date.now();
  saveConversation(conv);

  if (typeof renderConversationList === 'function') {
      renderConversationList();
  }
}

function getSidebarState() {
  return localStorage.getItem(STORAGE_KEYS.SIDEBAR_OPEN) !== 'false';
}

function saveSidebarState(isOpen) {
  localStorage.setItem(STORAGE_KEYS.SIDEBAR_OPEN, String(isOpen));
}


const chatLog = document.querySelector("#chatLog");
const statusBar = document.querySelector("#statusBar");
const providerSelect = document.querySelector("#providerSelect");
const providerKind = document.querySelector("#providerKind");
const baseUrlInput = document.querySelector("#baseUrl");
const ollamaThinkInput = document.querySelector("#ollamaThink");
const ollamaThinkLabel = document.querySelector("#ollamaThinkLabel");
const modelPlan = document.querySelector("#modelPlan");
const modelExecute = document.querySelector("#modelExecute");
const modelCritique = document.querySelector("#modelCritique");
const addCouncilMemberBtn = document.querySelector("#addCouncilMember");
const judgeModelInput = document.querySelector("#judgeModel");
const modelSuggestions = document.querySelector("#modelSuggestions");
const promptInput = document.querySelector("#promptInput");
const sendButton = document.querySelector("#sendButton");
const mainTabs = document.querySelector("#mainTabs");
const standardSettings = document.querySelector("#standardSettings");
const councilSettings = document.querySelector("#councilSettings");
const modeToggle = document.querySelector("#modeToggle");
const heroTitle = document.querySelector("#heroTitle");
const orgModelInput = document.querySelector("#orgModel");
const orgSettings = document.querySelector("#orgSettings");

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

function enhanceCodeBlocks(element) {
  element.querySelectorAll("pre").forEach((pre) => {
    if (pre.parentElement?.classList.contains("code-block-wrapper")) {
      return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "code-block-wrapper";

    const toolbar = document.createElement("div");
    toolbar.className = "code-block-toolbar";

    const codeEl = pre.querySelector("code");
    const lang = [...(codeEl?.classList ?? [])]
      .find((c) => c.startsWith("language-"))
      ?.replace("language-", "") ?? "";

    if (lang) {
      const langLabel = document.createElement("span");
      langLabel.className = "code-block-lang";
      langLabel.textContent = lang;
      toolbar.appendChild(langLabel);
    }

    const copyBtn = document.createElement("button");
    copyBtn.className = "code-copy-btn";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", () => {
      const text = (codeEl ?? pre).textContent ?? "";
      navigator.clipboard.writeText(text).then(() => {
        copyBtn.textContent = "Copied!";
        setTimeout(() => (copyBtn.textContent = "Copy"), 2000);
      });
    });

    toolbar.appendChild(copyBtn);
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(toolbar);
    wrapper.appendChild(pre);
  });
}

function renderFinalAnswer(element, html) {
  if (html) {
    element.innerHTML = html;
    renderMath(element);
    enhanceCodeBlocks(element);
    return;
  }

  element.textContent = "No answer returned.";
}

function renderStepContent(step) {
  const html = DOMPurify.sanitize(marked.parse(step.buffer));
  renderFinalAnswer(step.content, html);
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
  const html = DOMPurify.sanitize(marked.parse(container.pendingAnswerContent));
  renderFinalAnswer(container.finalAnswer, html);
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

function syncProviderSpecificSettings() {
  const isOllama = providerKind.value === "ollama";
  ollamaThinkLabel.style.display = isOllama ? "flex" : "none";
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

function populateCouncilModels(selectedModels) {
  const container = document.querySelector("#councilMembersContainer");
  const label = container.querySelector("label");
  container.innerHTML = "";
  container.appendChild(label);

  const modelsArray =
    selectedModels && selectedModels.length ? selectedModels : [""];
  modelsArray.forEach((model) => {
    addCouncilMemberInput(model);
  });
}

function addCouncilMemberInput(value = "") {
  const container = document.querySelector("#councilMembersContainer");

  const wrapper = document.createElement("div");
  wrapper.style.display = "flex";
  wrapper.style.gap = "0.5rem";
  wrapper.style.marginBottom = "0.5rem";

  const input = document.createElement("input");
  input.type = "text";
  input.setAttribute("list", "modelSuggestions");
  input.placeholder = "llama3.2:3b";
  input.value = value;
  input.className = "council-member-input";
  input.style.flex = "1";

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "ghost-button icon-button";
  removeBtn.innerHTML = "&times;";
  removeBtn.title = "Remove advisor";
  removeBtn.style.padding = "0.25rem 0.5rem";
  removeBtn.style.fontSize = "1.25rem";
  removeBtn.addEventListener("click", () => {
    wrapper.remove();
    saveSettings();
  });

  wrapper.appendChild(input);
  wrapper.appendChild(removeBtn);
  container.appendChild(wrapper);
}

function renderSettings() {
  if (!state.settings) {
    return;
  }

  renderProviders();
  providerKind.value = state.settings.provider_kind;
  baseUrlInput.value = state.settings.base_url;
  ollamaThinkInput.checked = Boolean(state.settings.ollama_think);
  syncProviderSpecificSettings();

  populateModelSuggestions();
  populateModelInput(modelPlan, state.settings.model_map.plan);
  populateModelInput(modelExecute, state.settings.model_map.execute);
  populateModelInput(modelCritique, state.settings.model_map.critique);

  populateCouncilModels(state.settings.council_models);
  populateModelInput(judgeModelInput, state.settings.judge_model);

  populateModelInput(orgModelInput, state.settings.org_model);
}

async function loadSettings(endpoint = "/api/settings") {
  try {
    let payload;
    let localSettings = null;

    if (endpoint === "/api/settings") {
        localSettings = getSettings();
    }

    if (localSettings) {
      // We have local settings, push them to server to sync
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(localSettings),
      });
      payload = await response.json();
    } else {
      // Fallback to server fetch or refresh
      const response = await fetch(endpoint, {
        method: endpoint === "/api/settings" ? "GET" : "POST",
      });
      payload = await response.json();

      // Cache fetched settings if GET
      if (endpoint === "/api/settings") {
          saveLocalSettings(payload.settings);
      }
    }

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
  } catch(error) {
     setStatus(error instanceof Error ? error.message : "Failed to load settings.");
     throw error;
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
  syncProviderSpecificSettings();
  populateModelSuggestions();
  populateModelInput(modelPlan, state.settings?.model_map.plan);
  populateModelInput(modelExecute, state.settings?.model_map.execute);
  populateModelInput(modelCritique, state.settings?.model_map.critique);
  populateCouncilModels(state.settings?.council_models);
  populateModelInput(judgeModelInput, state.settings?.judge_model);
  populateModelInput(orgModelInput, state.settings?.org_model);
}

async function saveSettings() {
  const selectedCouncilModels = Array.from(
    document.querySelectorAll(".council-member-input"),
  )
    .map((input) => input.value.trim())
    .filter((v) => v !== "");

  const payload = {
    provider_name: providerSelect.value,
    provider_kind: providerKind.value,
    base_url: baseUrlInput.value,
    ollama_think: ollamaThinkInput.checked,
    model_map: {
      plan: modelPlan.value,
      execute: modelExecute.value,
      critique: modelCritique.value,
    },
    council_models: selectedCouncilModels,
    judge_model: judgeModelInput.value,
    org_model: orgModelInput.value,
  };

  // Save locally immediately
  saveLocalSettings(payload);

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

// Below saveSettings, add event listeners for auto-save
function setupAutoSave() {
  const inputs = [
    providerSelect, providerKind, baseUrlInput, ollamaThinkInput,
    modelPlan, modelExecute, modelCritique, judgeModelInput, orgModelInput
  ];

  inputs.forEach(input => {
    if(input) {
        input.addEventListener('change', saveSettings);
        if(input.type === 'text') {
            input.addEventListener('blur', saveSettings);
        }
    }
  });

  // Event delegation for dynamically added council members
  const councilMembersContainer = document.querySelector("#councilMembersContainer");
  if (councilMembersContainer) {
    councilMembersContainer.addEventListener('change', (e) => {
      if (e.target.classList.contains('council-member-input')) {
        saveSettings();
      }
    });
    councilMembersContainer.addEventListener('blur', (e) => {
      if (e.target.classList.contains('council-member-input')) {
        saveSettings();
      }
    }, true); // useCapture for blur event
  }

  // Update remove 'saveSettings' button listener to just provide visual feedback since it auto-saves
  const saveBtn = document.querySelector("#saveSettings");
  if(saveBtn) {
      saveBtn.textContent = "Settings Auto-Save";
      saveBtn.disabled = true;
      saveBtn.style.opacity = "0.7";
      saveBtn.style.cursor = "default";
  }
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
    renderScheduled: false,
  };
}

function createOrgChartMessage() {
  const template = document.querySelector("#orgChartMessageTemplate");
  const node = template.content.firstElementChild.cloneNode(true);
  chatLog.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return {
    root: node,
    finalAnswer: node.querySelector('[data-role="finalAnswer"]'),
    orgChart: node.querySelector('[data-role="orgChart"]'),
    nodes: new Map(),
    pendingAnswerContent: "",
    renderScheduled: false,
  };
}

function createOrgNode(container, event) {
  const template = document.querySelector("#orgNodeTemplate");
  const nodeEl = template.content.firstElementChild.cloneNode(true);
  const parentNode = event.parent_id ? container.nodes.get(event.parent_id) : null;
  const depth = parentNode ? parentNode.depth + 1 : 0;

  nodeEl.dataset.nodeId = event.node_id;
  nodeEl.dataset.depth = String(depth);
  nodeEl.classList.add("org-node--active");

  nodeEl.querySelector('[data-role="orgNodeRole"]').textContent = event.role;
  nodeEl.querySelector('[data-role="orgNodeSlug"]').textContent =
    event.department ? `${event.department}` : event.slug;

  const badge = nodeEl.querySelector('[data-role="orgNodeBadge"]');
  badge.textContent = `${event.reports || 0} reports`;
  if (!event.reports) badge.style.display = "none";

  const output = nodeEl.querySelector('[data-role="orgNodeOutput"]');
  const toggle = nodeEl.querySelector('[data-role="orgNodeToggle"]');

  toggle.addEventListener("click", () => {
    output.classList.toggle("collapsed");
    orgNode.isExpanded = !output.classList.contains("collapsed");
    if (orgNode.isExpanded) {
      const html = DOMPurify.sanitize(marked.parse(orgNode.buffer));
      renderFinalAnswer(output, html);
    }
  });

  if (parentNode) {
    parentNode.childrenEl.appendChild(nodeEl);
  } else {
    container.orgChart.appendChild(nodeEl);
  }

  const orgNode = {
    root: nodeEl,
    output,
    status: nodeEl.querySelector('[data-role="orgNodeStatus"]'),
    badge,
    childrenEl: nodeEl.querySelector('[data-role="orgNodeChildren"]'),
    buffer: "",
    depth,
    isExpanded: false,
    renderScheduled: false,
  };

  container.nodes.set(event.node_id, orgNode);
  nodeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return orgNode;
}

function updateOrgNode(container, event) {
  const orgNode = container.nodes.get(event.node_id);
  if (!orgNode) return;

  if (event.type === "org-node-delta") {
    orgNode.buffer = event.content ?? `${orgNode.buffer}${event.delta}`;
    if (orgNode.isExpanded) {
      if (!orgNode.renderScheduled) {
        orgNode.renderScheduled = true;
        window.requestAnimationFrame(() => {
          orgNode.renderScheduled = false;
          const html = DOMPurify.sanitize(marked.parse(orgNode.buffer));
          renderFinalAnswer(orgNode.output, html);
        });
      }
    }
  }

  if (event.type === "org-node-complete") {
    orgNode.buffer = event.content;
    orgNode.status.textContent = "Done";
    orgNode.root.classList.remove("org-node--active");

    if (event.reports !== undefined && event.reports > 0) {
      orgNode.badge.textContent = `${event.reports} reports`;
      orgNode.badge.style.display = "";
    }

    if (orgNode.isExpanded) {
      const html = DOMPurify.sanitize(marked.parse(orgNode.buffer));
      renderFinalAnswer(orgNode.output, html);
    }
  }
}

function createStepCard(container, event) {
  const template = document.querySelector("#stepTemplate");
  const stepNode = template.content.firstElementChild.cloneNode(true);
  stepNode.dataset.step = event.step;
  stepNode.querySelector('[data-role="stepLabel"]').textContent = event.label;
  stepNode.querySelector('[data-role="stepModel"]').textContent = event.model;

  const content = stepNode.querySelector('[data-role="stepContent"]');
  const toggle = stepNode.querySelector('[data-role="stepToggle"]');
  const showContent = event.thought !== false;

  if (!showContent) {
    content.remove();
  } else {
    toggle.addEventListener("click", () => {
      content.classList.toggle("collapsed");
      step.isExpanded = !content.classList.contains("collapsed");
      if (step.isExpanded) {
        renderStepContent(step);
      }
    });
  }

  const step = {
    root: stepNode,
    content,
    state: stepNode.querySelector('[data-role="stepState"]'),
    buffer: "",
    isExpanded: false,
    renderScheduled: false,
    showContent,
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
    if (step.showContent) {
      scheduleStepRender(step);
    }
  }

  if (event.type === "step-complete") {
    step.buffer = event.content;
    if (step.showContent) {
      if (step.renderScheduled) {
        flushStepRender(step);
      } else if (step.isExpanded) {
        renderStepContent(step);
      }
    }
    step.state.textContent = "Done";
  }
}

async function streamChat(message) {
  if (state.selectedMode === "org") {
    await streamOrgChat(message);
  } else {
    await streamStandardChat(message);
  }
}

async function streamOrgChat(message) {
  appendUserMessage(message);
  saveMessage("user", message);
  const container = createOrgChartMessage();

  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, mode: "org" }),
  });

  if (!response.ok || !response.body) {
    container.finalAnswer.textContent = "The org pipeline failed to start.";
    return;
  }

  container.finalAnswer.innerHTML = '<span class="hint">Delegating...</span>';
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalAnswer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    lines.forEach((line) => {
      if (!line.trim()) return;
      const event = JSON.parse(line);

      if (event.type === "org-node-start") {
        createOrgNode(container, event);
      }

      if (event.type === "org-node-delta" || event.type === "org-node-complete") {
        updateOrgNode(container, event);
      }

      if (event.type === "answer-start") {
        container.pendingAnswerContent = "";
        container.finalAnswer.textContent = "";
      }

      if (event.type === "answer-delta") {
        finalAnswer = event.content ?? `${finalAnswer}${event.delta}`;
        container.pendingAnswerContent = finalAnswer;
        scheduleFinalAnswerRender(container);
      }

      if (event.type === "answer-complete") {
        finalAnswer = event.content;
        container.pendingAnswerContent = finalAnswer;
        if (container.renderScheduled) {
          flushFinalAnswerRender(container);
        } else {
          const html = DOMPurify.sanitize(marked.parse(finalAnswer));
          renderFinalAnswer(container.finalAnswer, html);
        }
      }

      if (event.type === "error") {
        container.finalAnswer.textContent = event.message;
      }
    });
  }

  container.root.querySelector(".message-meta").textContent = "ORG CHART";
  container.root.scrollIntoView({ behavior: "smooth", block: "end" });

  if (finalAnswer) {
    saveMessage("assistant", finalAnswer);
  }
}

async function streamStandardChat(message) {
  appendUserMessage(message);
  saveMessage("user", message);
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
        assistant.finalAnswer.textContent = "";
      }

      if (event.type === "answer-delta") {
        finalAnswer = event.content ?? `${finalAnswer}${event.delta}`;
        assistant.pendingAnswerContent = finalAnswer;
        scheduleFinalAnswerRender(assistant);
      }

      if (event.type === "answer-complete") {
        finalAnswer = event.content;
        assistant.pendingAnswerContent = finalAnswer;
        if (assistant.renderScheduled) {
          flushFinalAnswerRender(assistant);
        } else {
          const html = DOMPurify.sanitize(marked.parse(finalAnswer));
          renderFinalAnswer(assistant.finalAnswer, html);
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

  if (finalAnswer) {
    saveMessage("assistant", finalAnswer);
  }
}

if (modeToggle) {
  modeToggle.querySelectorAll(".mode-button").forEach((button) => {
    button.addEventListener("click", () => {
      modeToggle
        .querySelectorAll(".mode-button")
        .forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      state.selectedMode = button.dataset.mode;
      setStatus(`Reasoning mode set to ${state.selectedMode}.`);
    });
  });
}

if (mainTabs) {
  mainTabs.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON") {
      document
        .querySelectorAll(".tab-button")
        .forEach((btn) => btn.classList.remove("active"));
      e.target.classList.add("active");
      state.activeTab = e.target.dataset.tab;

      if (state.activeTab === "council") {
        standardSettings.style.display = "none";
        councilSettings.style.display = "block";
        orgSettings.style.display = "none";
        modeToggle.style.display = "none";
        heroTitle.textContent = "Consult the Agent Council";
        state.selectedMode = "council";
      } else if (state.activeTab === "org") {
        standardSettings.style.display = "none";
        councilSettings.style.display = "none";
        orgSettings.style.display = "block";
        modeToggle.style.display = "none";
        heroTitle.textContent = "Delegate to the Organisation";
        state.selectedMode = "org";
      } else {
        standardSettings.style.display = "block";
        councilSettings.style.display = "none";
        orgSettings.style.display = "none";
        modeToggle.style.display = "flex";
        heroTitle.textContent = "Make your local models reason";

        // Restore previous standard mode or default to off
        const activeModeBtn = modeToggle
          ? modeToggle.querySelector(".mode-button.active")
          : null;
        state.selectedMode = activeModeBtn
          ? activeModeBtn.dataset.mode
          : "off";
      }
    }
  });
}

const appShell = document.querySelector("#appShell");
const toggleSidebar = document.querySelector("#toggleSidebar");
if (toggleSidebar && appShell) {
  toggleSidebar.addEventListener("click", () => {
    appShell.classList.toggle("sidebar-collapsed");
  });
}

providerSelect.addEventListener("change", syncProviderFields);
providerKind.addEventListener("change", syncProviderSpecificSettings);

if (addCouncilMemberBtn) {
  addCouncilMemberBtn.addEventListener("click", () =>
    addCouncilMemberInput(""),
  );
}

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

function renderConversationList() {
  const list = document.querySelector("#conversationList");
  if (!list) return;

  list.innerHTML = '';
  const conversations = getConversations();

  conversations.forEach(conv => {
    const li = document.createElement("li");
    li.className = `conversation-item ${conv.id === state.activeConversationId ? 'active' : ''}`;

    const titleSpan = document.createElement("span");
    titleSpan.className = "conversation-title";
    titleSpan.textContent = conv.title || "Empty Chat";

    const countBadge = document.createElement("span");
    countBadge.className = "chat-count-badge";

    // Map mode to a single letter
    let modeLetter = "S"; // standard/off
    if (conv.mode === "council") modeLetter = "C";
    else if (conv.mode === "org") modeLetter = "O";
    else if (conv.mode === "medium") modeLetter = "M";
    else if (conv.mode === "hard") modeLetter = "H";
    else if (conv.mode === "thinking") modeLetter = "T"; // thinking (legacy or specific)

    countBadge.textContent = modeLetter;

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "ghost-button icon-button delete-chat-btn";
    deleteBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path>
      </svg>
    `;

    li.addEventListener("click", () => {
      if (conv.id !== state.activeConversationId) {
        loadChat(conv.id);
      }
    });

    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent triggering list item click
      deleteConversation(conv.id);
    });

    li.appendChild(titleSpan);
    li.appendChild(countBadge);
    li.appendChild(deleteBtn);
    list.appendChild(li);
  });
}

function initHistorySidebar() {
  const toggleBtn = document.querySelector("#toggleHistorySidebar");
  const newChatBtn = document.querySelector("#newChatButton");
  const appShell = document.querySelector("#appShell");

  if (toggleBtn && appShell) {
    // Set initial state
    if (!getSidebarState()) {
      appShell.classList.add("history-collapsed");
    }

    toggleBtn.addEventListener("click", () => {
      const isCollapsed = appShell.classList.toggle("history-collapsed");
      saveSidebarState(!isCollapsed);
    });
  }

  if (newChatBtn) {
    newChatBtn.addEventListener("click", startNewChat);
  }
}

async function initializeApp() {
  initHistorySidebar();
  setupAutoSave();

  // Load settings
  try {
    await loadSettings();
  } catch (error) {
    console.error("Settings load failed", error);
  }

  // Restore active conversation or start new
  const savedActiveId = localStorage.getItem(STORAGE_KEYS.ACTIVE_CONVERSATION);
  if (savedActiveId && getConversation(savedActiveId)) {
    loadChat(savedActiveId);
  } else {
    startNewChat();
  }
}

initializeApp();
