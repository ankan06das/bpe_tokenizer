const API = {
  async tokenize(text) {
    const res = await fetch("/tokenize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    return res.json();
  },
  async load(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/load", { method: "POST", body: form });
    return res.json();
  },
  async listModels() {
    const res = await fetch("/models");
    return res.json();
  },
  async loadPretrained(filename) {
    const res = await fetch(`/load-pretrained/${encodeURIComponent(filename)}`, {
      method: "POST"
    });
    return res.json();
  }
};

const TIKTOKEN_COLORS = [
  { light: "rgba(107,64,216,.3)", dark: "rgba(107,64,216,.5)" },
  { light: "rgba(104,204,202,.4)", dark: "rgba(104,204,202,.6)" },
  { light: "rgba(253,228,148,.6)", dark: "rgba(253,228,148,.2)" },
  { light: "rgba(255,115,115,.4)", dark: "rgba(255,115,115,.6)" },
  { light: "rgba(141,171,255,.4)", dark: "rgba(141,171,255,.6)" }
];

function updateModelStatus() {
  const textEl = document.getElementById("status-text");
  const indicatorEl = document.getElementById("status-indicator");
  if (window._modelTrained) {
    textEl.textContent = `Trained (vocab size: ${window._vocabSize})`;
    indicatorEl.className = "h-3 w-3 rounded-full bg-emerald-500";
  } else {
    textEl.textContent = "No model loaded";
    indicatorEl.className = "h-3 w-3 rounded-full bg-slate-300 dark:bg-slate-600 animate-pulse";
  }
}

async function loadModel() {
  const fileInput = document.getElementById("model-file");
  const status = document.getElementById("load-status");
  const btn = document.getElementById("load-btn");

  if (!fileInput.files.length) {
    status.textContent = "Please select a .pkl file.";
    status.className = "mt-2 text-xs text-red-500 font-medium";
    return;
  }

  status.textContent = "Loading model...";
  status.className = "mt-2 text-xs text-indigo-500 font-medium";
  btn.disabled = true;

  try {
    const result = await API.load(fileInput.files[0]);
    status.textContent = `Model loaded successfully (vocab size: ${result.vocab_size})`;
    status.className = "mt-2 text-xs text-emerald-500 font-medium";
    window._modelTrained = true;
    window._vocabSize = result.vocab_size;
    updateModelStatus();
    // Re-tokenize current text with the new model
    tokenize();
  } catch (err) {
    status.textContent = `Error: ${err.message || err}`;
    status.className = "mt-2 text-xs text-red-500 font-medium";
  } finally {
    btn.disabled = false;
  }
}

function switchModelTab(tab) {
  const tabPretrained = document.getElementById("tab-pretrained");
  const tabUpload = document.getElementById("tab-upload");
  const pretrainedContent = document.getElementById("pretrained-content");
  const uploadContent = document.getElementById("upload-content");
  const status = document.getElementById("load-status");

  if (status) {
    status.textContent = "";
  }

  if (tab === "pretrained") {
    tabPretrained.classList.add("bg-white", "dark:bg-slate-700", "shadow-sm");
    tabPretrained.classList.remove("text-slate-500", "dark:text-slate-400");
    tabPretrained.classList.add("text-slate-700", "dark:text-slate-300");

    tabUpload.classList.remove("bg-white", "dark:bg-slate-700", "shadow-sm");
    tabUpload.classList.add("text-slate-500", "dark:text-slate-400");
    tabUpload.classList.remove("text-slate-700", "dark:text-slate-300");

    pretrainedContent.classList.remove("hidden");
    uploadContent.classList.add("hidden");
  } else {
    tabUpload.classList.add("bg-white", "dark:bg-slate-700", "shadow-sm");
    tabUpload.classList.remove("text-slate-500", "dark:text-slate-400");
    tabUpload.classList.add("text-slate-700", "dark:text-slate-300");

    tabPretrained.classList.remove("bg-white", "dark:bg-slate-700", "shadow-sm");
    tabPretrained.classList.add("text-slate-500", "dark:text-slate-400");
    tabPretrained.classList.remove("text-slate-700", "dark:text-slate-300");

    uploadContent.classList.remove("hidden");
    pretrainedContent.classList.add("hidden");
  }
}

async function loadPretrainedModel() {
  const select = document.getElementById("model-select");
  const status = document.getElementById("load-status");
  const btn = document.getElementById("load-pretrained-btn");

  if (!select.value) {
    status.textContent = "Please select a model from the list.";
    status.className = "mt-2 text-xs text-red-500 font-medium";
    return;
  }

  status.textContent = "Loading model...";
  status.className = "mt-2 text-xs text-indigo-500 font-medium";
  btn.disabled = true;

  try {
    const result = await API.loadPretrained(select.value);
    status.textContent = `Model loaded successfully (vocab size: ${result.vocab_size})`;
    status.className = "mt-2 text-xs text-emerald-500 font-medium";
    window._modelTrained = true;
    window._vocabSize = result.vocab_size;
    updateModelStatus();
    tokenize();
  } catch (err) {
    status.textContent = `Error: ${err.message || err}`;
    status.className = "mt-2 text-xs text-red-500 font-medium";
  } finally {
    btn.disabled = false;
  }
}

async function fetchModels() {
  const select = document.getElementById("model-select");
  if (!select) return;

  try {
    const models = await API.listModels();
    select.innerHTML = "";
    if (models.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "No models available in /models";
      opt.disabled = true;
      opt.selected = true;
      select.appendChild(opt);
      return;
    }

    models.forEach((model) => {
      const opt = document.createElement("option");
      opt.value = model;
      // Make the label pretty: e.g. tokenizer_shakespeare_12712.pkl -> Shakespeare 12712
      let prettyName = model.replace(/\.pkl$/, "").replace(/^tokenizer_/, "");
      prettyName = prettyName.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
      opt.textContent = prettyName;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load pretrained models list:", err);
    select.innerHTML = `<option disabled selected>Error loading models</option>`;
  }
}

let currentTokens = [];

async function tokenize() {
  const textInput = document.getElementById("input-text");
  const text = textInput.value;
  if (!text) {
    renderTokens([]);
    return;
  }

  try {
    const tokens = await API.tokenize(text);
    currentTokens = tokens;
    renderTokens(tokens);
  } catch (err) {
    console.error(`Tokenize error: ${err.message}`);
  }
}

function renderTokens(tokens) {
  const viz = document.getElementById("token-list");
  const idsContainer = document.getElementById("visual-breakdown");
  const count = document.getElementById("token-count");
  const showWhitespace = document.getElementById("show-whitespace").checked;
  const countCard = document.getElementById("token-count-card");

  viz.innerHTML = "";
  idsContainer.innerHTML = "";
  count.textContent = tokens.length;

  if (tokens.length > 0) {
    countCard.classList.add("shadow-indigo-300/30", "dark:shadow-indigo-900/40");
  } else {
    countCard.classList.remove("shadow-indigo-300/30", "dark:shadow-indigo-900/40");
  }

  if (tokens.length === 0) {
    viz.className = "p-5 min-h-[140px] text-slate-400 dark:text-slate-500 text-sm flex flex-wrap gap-2 italic content-start custom-scrollbar overflow-y-auto";
    viz.textContent = "Tokens will appear here...";
    idsContainer.className = "p-5 min-h-[140px] text-slate-400 dark:text-slate-500 text-sm flex flex-wrap gap-2 italic content-start custom-scrollbar overflow-y-auto";
    idsContainer.textContent = "The IDs of the tokens will appear here...";
    return;
  }

  viz.className = "p-5 min-h-[140px] text-lg font-mono custom-scrollbar overflow-y-auto leading-relaxed whitespace-pre-wrap select-text";
  idsContainer.className = "p-5 min-h-[140px] text-lg font-mono custom-scrollbar overflow-y-auto leading-relaxed select-text";

  tokens.forEach((t, index) => {
    const colorTheme = TIKTOKEN_COLORS[index % TIKTOKEN_COLORS.length];
    const bgColorLight = colorTheme.light;
    const textColorLight = "inherit";
    const bgColorDark = colorTheme.dark;
    const textColorDark = "inherit";

    let displayToken = t.token;
    if (showWhitespace) {
      displayToken = displayToken.replace(/ /g, '·').replace(/\n/g, '↵\n');
    }

    // Token Chip
    const span = document.createElement("span");
    span.className = "token-chip px-0.5 rounded transition-all cursor-pointer font-mono";
    span.style.setProperty('--bg-light', bgColorLight);
    span.style.setProperty('--text-light', textColorLight);
    span.style.setProperty('--bg-dark', bgColorDark);
    span.style.setProperty('--text-dark', textColorDark);
    span.textContent = displayToken;
    span.dataset.index = index;

    // ID Chip
    const idSpan = document.createElement("span");
    idSpan.className = "token-id cursor-pointer transition-colors";
    idSpan.textContent = t.id;
    idSpan.dataset.index = index;

    const highlight = () => {
      document.querySelectorAll(".token-chip").forEach(el => el.classList.add("dimmed"));
      document.querySelectorAll(".token-id").forEach(el => el.classList.add("dimmed"));
      span.classList.remove("dimmed");
      span.classList.add("highlighted", "active-card");
      idSpan.classList.remove("dimmed");
      idSpan.classList.add("font-bold", "text-indigo-600", "dark:text-indigo-400");
    };

    const removeHighlight = () => {
      document.querySelectorAll(".token-chip").forEach(el => {
        el.classList.remove("dimmed", "highlighted", "active-card");
      });
      document.querySelectorAll(".token-id").forEach(el => {
        el.classList.remove("dimmed", "font-bold", "text-indigo-600", "dark:text-indigo-400");
      });
    };

    span.addEventListener("mouseenter", highlight);
    span.addEventListener("mouseleave", removeHighlight);
    idSpan.addEventListener("mouseenter", highlight);
    idSpan.addEventListener("mouseleave", removeHighlight);

    viz.appendChild(span);
    idsContainer.appendChild(idSpan);
    if (index < tokens.length - 1) {
      idsContainer.appendChild(document.createTextNode(", "));
    }
  });
}

// Theme handling
function initTheme() {
  const themeToggle = document.getElementById("theme-toggle");
  const themeIcon = document.getElementById("theme-icon");

  if (!themeToggle || !themeIcon) return;

  const isDark = localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches);
  
  if (isDark) {
    document.documentElement.classList.add('dark');
    themeIcon.textContent = 'light_mode';
  } else {
    document.documentElement.classList.remove('dark');
    themeIcon.textContent = 'dark_mode';
  }

  themeToggle.addEventListener("click", () => {
    if (document.documentElement.classList.contains('dark')) {
      document.documentElement.classList.remove('dark');
      localStorage.theme = 'light';
      themeIcon.textContent = 'dark_mode';
    } else {
      document.documentElement.classList.add('dark');
      localStorage.theme = 'dark';
      themeIcon.textContent = 'light_mode';
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  updateModelStatus();
  fetchModels();

  let debounceTimer;
  const inputText = document.getElementById("input-text");
  if (inputText) {
    inputText.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(tokenize, 150);
    });
  }

  const clearBtn = document.getElementById("clear-btn");
  if (clearBtn && inputText) {
    clearBtn.addEventListener("click", () => {
      inputText.value = "";
      renderTokens([]);
    });
  }

  const showWhitespace = document.getElementById("show-whitespace");
  if (showWhitespace) {
    showWhitespace.addEventListener("change", () => {
      renderTokens(currentTokens);
    });
  }

  const fileInput = document.getElementById("model-file");
  const fileNameSpan = document.getElementById("model-file-name");
  if (fileInput && fileNameSpan) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        fileNameSpan.textContent = e.target.files[0].name;
      } else {
        fileNameSpan.textContent = "Choose a vocabulary file...";
      }
    });
  }
});
