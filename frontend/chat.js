/**
 * chat.js — WebSocket client for the Resolve AI Assistant
 *
 * Handles:
 *  - WebSocket connection + reconnect
 *  - Message rendering with markdown-lite support
 *  - Typing indicator
 *  - Confirmation panel (confirm / cancel)
 *  - Model selector UI (live switching, no restart)
 *  - Action log sidebar
 *  - Auto-resizing textarea
 */

/* ══════════════════════════════════════════════════════════════════════════════
   State
══════════════════════════════════════════════════════════════════════════════ */
let ws = null;
let isWaiting = false;
let currentProvider = 'anthropic';
let currentModel    = '';
let currentThreadId = new URLSearchParams(window.location.search).get('chat');
if (!currentThreadId) {
  currentThreadId = crypto.randomUUID();
  window.history.replaceState(null, '', `?chat=${currentThreadId}`);
}

/* ══════════════════════════════════════════════════════════════════════════════
   DOM refs
══════════════════════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const messagesEl       = $('messages');
const inputEl          = $('message-input');
const sendBtn          = $('send-btn');
const confirmPanel     = $('confirmation-panel');
const confirmBody      = $('confirm-body');
const btnConfirm       = $('btn-confirm');
const btnCancel        = $('btn-cancel');
const modelBadge       = $('model-badge');
const providerSelect   = $('provider-select');
const modelSelect      = $('model-select');
const applyModelBtn    = $('apply-model-btn');
const resolveStatus    = $('resolve-status');
const resolveDot       = $('resolve-dot');
const llmStatus        = $('llm-status');
const llmDot           = $('llm-dot');
const sidebarToggle    = $('sidebar-toggle');
const mobileSidebarToggle = $('mobile-sidebar-toggle');
const chatList         = $('chat-list');
const newChatBtn       = $('new-chat-btn');

// Proposal panel (Add New Capability) — strictly separate from confirmPanel

let _currentProposal   = null;

/* ══════════════════════════════════════════════════════════════════════════════
   WebSocket & Chat History
══════════════════════════════════════════════════════════════════════════════ */
async function loadChatHistory() {
  try {
    const res = await fetch(`/api/chats/${currentThreadId}`);
    const data = await res.json();
    if (data.messages && data.messages.length > 0) {
      // Clear welcome message
      messagesEl.innerHTML = '';
      data.messages.forEach(msg => {
        appendMessage(msg.role, msg.content, msg.details || null);
      });
      scrollToBottom();
    }
  } catch (e) {
    console.error("Failed to load history", e);
  }
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/chat/${currentThreadId}`);

  ws.addEventListener('open', () => {
    console.log('WS connected to thread', currentThreadId);
    updateInputState();
  });

  ws.addEventListener('message', evt => {
    const data = JSON.parse(evt.data);
    handleServerMessage(data);
  });

  ws.addEventListener('close', () => {
    console.log('WS closed — reconnecting in 2s');
    ws = null;
    updateInputState();
    setTimeout(connect, 2000);
  });

  ws.addEventListener('error', err => {
    console.error('WS error', err);
  });
}

function sendWS(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

/* ══════════════════════════════════════════════════════════════════════════════
   Message handling
══════════════════════════════════════════════════════════════════════════════ */
function handleServerMessage(data) {
  removeTypingBubble();

  if (data.type === 'typing') {
    showTypingBubble();
    return;
  }

  if (data.type === 'response') {
    if (confirmationTimeout) {
      clearTimeout(confirmationTimeout);
      confirmationTimeout = null;
    }
    
    if (data.needs_confirmation) {
      showConfirmation(data.needs_confirmation);
    } else {
      hideConfirmation();
      if (data.content) {
        appendMessage('assistant', data.content, data.details || null);
        // The first completed turn can create the conversation title.
        loadChats();
      }
    }
    isWaiting = false;
    updateInputState();
  } else if (data.type === 'verify_result') {
    handleVerifyResult(data.valid);
  }
}

/* ══════════════════════════════════════════════════════════════════════════════
   Render messages
══════════════════════════════════════════════════════════════════════════════ */
function appendMessage(role, content, details = null) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.setAttribute('aria-label', role === 'user' ? 'You' : 'Resolve AI');
  avatar.innerHTML = avatarMarkup(role);

  const body = document.createElement('div');
  body.className = 'message-body';

  const contentEl = document.createElement('div');
  contentEl.className = 'message-content';
  contentEl.innerHTML = renderMarkdown(content);

  body.appendChild(contentEl);

  if (details && details.length > 0) {
    const detailsWrapper = document.createElement('details');
    detailsWrapper.className = 'technical-details';
    
    const summary = document.createElement('summary');
    summary.textContent = 'Show details';
    detailsWrapper.addEventListener('toggle', () => {
      summary.textContent = detailsWrapper.open ? 'Hide details' : 'Show details';
    });
    detailsWrapper.appendChild(summary);

    const detailsContent = document.createElement('div');
    detailsContent.className = 'details-content';
    
    details.forEach(detail => {
      const item = document.createElement('div');
      item.className = 'details-item';
      
      const isExecResult = detail.hasOwnProperty('success');
      const toolStr = detail.tool || 'unknown';
      let msgStr = '';
      let icon = '';
      
      if (isExecResult) {
        icon = detail.success ? '✅' : '❌';
        msgStr = detail.message || '';
        item.innerHTML = `<span>${icon} <code>${escHtml(toolStr)}</code> &mdash; ${escHtml(msgStr)}</span>`;
        
        // Show normalization traceability
        if (detail.data && detail.data.original_args && detail.data.normalized_args) {
          const orig = detail.data.original_args;
          const norm = detail.data.normalized_args;
          const diffs = [];
          for (const k in orig) {
            if (orig[k] !== norm[k]) diffs.push(`${k}: "${orig[k]}" → "${norm[k]}"`);
          }
          if (diffs.length > 0) {
            const normDiv = document.createElement('div');
            normDiv.style.fontSize = '0.8em';
            normDiv.style.color = 'var(--text-light)';
            normDiv.style.marginTop = '4px';
            normDiv.style.paddingLeft = '20px';
            normDiv.innerHTML = `<i>Normalized: ${escHtml(diffs.join(', '))}</i>`;
            item.appendChild(normDiv);
          }
        }
      } else {
        icon = '🛠️';
        const argsStr = Object.entries(detail.args || {}).map(([k,v]) => `${k}=${JSON.stringify(v)}`).join(', ');
        item.innerHTML = `<span>${icon} <code>${escHtml(toolStr)}</code>(${escHtml(argsStr)})</span>`;
      }
      detailsContent.appendChild(item);
    });

    detailsWrapper.appendChild(detailsContent);
    body.appendChild(detailsWrapper);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(body);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function avatarMarkup(role) {
  if (role === 'user') {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="3.2"/><path d="M5.5 20c.6-3.5 2.8-5.3 6.5-5.3s5.9 1.8 6.5 5.3"/></svg>';
  }
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 3.5 20h17L12 3Z"/><path d="M12 7.5 7.1 17h9.8L12 7.5Z"/><circle cx="12" cy="12.2" r="1.1"/></svg>';
}

function renderMarkdown(text) {
  // Light markdown renderer — no external dependency
  // Escape before applying formatting so model/tool output cannot inject HTML.
  const safeText = escHtml(String(text));
  return safeText
    // Code blocks
    .replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`)
    // Inline code
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Strikethrough
    .replace(/~~(.+?)~~/g, '<del>$1</del>')
    // Unordered list items
    .replace(/^[ \t]*[-•]\s+(.+)$/gm, '<li>$1</li>')
    .replace(/(<li>[\s\S]+?<\/li>)/g, '<ul>$1</ul>')
    // Headings
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Horizontal rule
    .replace(/^---$/gm, '<hr/>')
    // Newlines → paragraphs (skip inside pre/ul)
    .replace(/\n\n+/g, '</p><p>')
    .replace(/^(?!<[hpuo\/<])(.+)/gm, (m) => m)
    // Wrap in paragraph if not already a block element
    .replace(/^([^<\n].*)$/gm, (line) => {
      if (/^<(p|h[1-6]|ul|li|pre|hr)/.test(line)) return line;
      return `<p>${line}</p>`;
    });
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ── Typing indicator ───────────────────────────────────────────────────── */
function showTypingBubble() {
  if (document.querySelector('.typing-bubble')) return;
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant typing-bubble';
  wrapper.innerHTML = `
    <div class="message-avatar" aria-label="Resolve AI">${avatarMarkup('assistant')}</div>
    <div class="message-body">
      <div class="message-content">
        <div class="typing-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>`;
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function removeTypingBubble() {
  const el = document.querySelector('.typing-bubble');
  if (el) el.remove();
}

/* ══════════════════════════════════════════════════════════════════════════════
   Confirmation panel
══════════════════════════════════════════════════════════════════════════════ */
function showConfirmation(payload) {
  confirmBody.innerHTML = '';
  const p = document.createElement('p');
  p.textContent = payload.message || 'Please confirm execution.';
  confirmBody.appendChild(p);

  if (payload.details && payload.details.length > 0) {
    const detailsWrapper = document.createElement('details');
    detailsWrapper.className = 'technical-details';
    
    const summary = document.createElement('summary');
    summary.textContent = 'Show details';
    detailsWrapper.addEventListener('toggle', () => {
      summary.textContent = detailsWrapper.open ? 'Hide details' : 'Show details';
    });
    detailsWrapper.appendChild(summary);

    const detailsContent = document.createElement('div');
    detailsContent.className = 'details-content';
    payload.details.forEach(detail => {
      const item = document.createElement('div');
      item.className = 'details-item';
      const toolStr = detail.tool || 'unknown';
      const argsStr = Object.entries(detail.args || {}).map(([k,v]) => `${k}=${JSON.stringify(v)}`).join(', ');
      item.innerHTML = `<span>🛠️ <code>${escHtml(toolStr)}</code>(${escHtml(argsStr)})</span>`;
      detailsContent.appendChild(item);
    });
    detailsWrapper.appendChild(detailsContent);
    confirmBody.appendChild(detailsWrapper);
  }

  confirmPanel.classList.remove('hidden');
  btnConfirm.focus();
}

function hideConfirmation() {
  confirmPanel.classList.add('hidden');
}

/* ══════════════════════════════════════════════════════════════════════════════
   Proposal panel (Add New Capability)
══════════════════════════════════════════════════════════════════════════════ */


let confirmationTimeout = null;

btnConfirm.addEventListener('click', () => {
  hideConfirmation();
  isWaiting = true;
  updateInputState();
  showTypingBubble();
  sendWS({ type: 'confirm', confirmed: true });
  
  // Safety timeout in case the backend silently fails or hangs
  confirmationTimeout = setTimeout(() => {
    if (isWaiting) {
      isWaiting = false;
      updateInputState();
      removeTypingBubble();
      appendMessage('assistant', '⚠️ **Request Timed Out**: No response received from the server. The action may or may not have completed. Please check manually.');
    }
  }, 30000);
});

btnCancel.addEventListener('click', () => {
  hideConfirmation();
  sendWS({ type: 'confirm', confirmed: false });
  appendMessage('assistant', 'Action cancelled.');
});

/* ══════════════════════════════════════════════════════════════════════════════
   Send message
══════════════════════════════════════════════════════════════════════════════ */
function sendMessage() {
  const content = inputEl.value.trim();
  if (!content || isWaiting || !ws || ws.readyState !== WebSocket.OPEN) return;

  appendMessage('user', content);
  inputEl.value = '';
  autoResize();
  isWaiting = true;
  updateInputState();
  showTypingBubble();

  sendWS({
    type:     'message',
    content,
    provider: currentProvider,
    model:    currentModel,
  });
}

/* ── Input event handlers ───────────────────────────────────────────────── */
sendBtn.addEventListener('click', () => {
  if (isWaiting) {
    sendWS({ type: 'cancel' });
    return;
  }
  sendMessage();
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener('input', () => {
  autoResize();
  updateInputState();
});

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}

function updateInputState() {
  const ready = ws && ws.readyState === WebSocket.OPEN && !isWaiting;
  const hasContent = inputEl.value.trim().length > 0;
  
  if (isWaiting && ws && ws.readyState === WebSocket.OPEN) {
    sendBtn.classList.add('stop-state');
    sendBtn.title = 'Stop request';
    sendBtn.setAttribute('aria-label', 'Stop request');
    sendBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>';
    sendBtn.disabled = false;
  } else {
    sendBtn.classList.remove('stop-state');
    sendBtn.title = 'Send command';
    sendBtn.setAttribute('aria-label', 'Send command');
    sendBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
    sendBtn.disabled = !(ready && hasContent);
  }
  inputEl.disabled = !ready;
}

/* ══════════════════════════════════════════════════════════════════════════════
   Model selector
══════════════════════════════════════════════════════════════════════════════ */
async function loadModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();

    currentProvider = data.current_provider || 'anthropic';
    currentModel    = data.current_model    || '';

    providerSelect.value = currentProvider;
    populateModelSelect(data);
    updateModelBadge();
    updateLLMStatus();
  } catch (e) {
    console.error('Could not load models', e);
  }
}

function populateModelSelect(data) {
  modelSelect.innerHTML = '';
  const models = data[currentProvider] || [];

  if (models.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = currentProvider === 'ollama'
      ? 'No Ollama models found — pull a model first'
      : 'No models available';
    modelSelect.appendChild(opt);
    return;
  }

  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value   = m.id;
    opt.textContent = m.label || m.id;
    if (m.id === currentModel) opt.selected = true;
    modelSelect.appendChild(opt);
  });

  // If nothing matched, select first
  if (!currentModel && models.length > 0) {
    currentModel = models[0].id;
    modelSelect.value = currentModel;
  }
}

providerSelect.addEventListener('change', async () => {
  currentProvider = providerSelect.value;
  // Re-fetch to get the models for the newly selected provider
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    data.current_provider = currentProvider;
    populateModelSelect(data);
  } catch {}
});

applyModelBtn.addEventListener('click', async () => {
  currentProvider = providerSelect.value;
  currentModel    = modelSelect.value;

  try {
    await fetch('/api/set-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: currentProvider, model: currentModel }),
    });
    updateModelBadge();
    updateLLMStatus();
    applyModelBtn.textContent = '✓ Applied';
    setTimeout(() => { applyModelBtn.textContent = 'Apply Model'; }, 1800);
  } catch (e) {
    console.error('Could not set model', e);
  }
});

function updateModelBadge() {
  modelBadge.textContent = currentModel
    ? `${currentProvider} / ${currentModel}`
    : currentProvider;
}

function updateLLMStatus() {
  llmStatus.textContent = currentModel || currentProvider;
  llmDot.classList.add('active');
}

/* ══════════════════════════════════════════════════════════════════════════════
   Health check / Resolve status
══════════════════════════════════════════════════════════════════════════════ */
async function checkHealth() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();

    if (data.resolve_connected) {
      resolveStatus.textContent = 'Connected';
      resolveDot.className = 'status-dot active';
    } else {
      resolveStatus.textContent = 'Disconnected';
      resolveDot.className = 'status-dot';
    }
  } catch {
    resolveStatus.textContent = 'Offline';
    resolveDot.className = 'status-dot';
  }
}

// Poll health every 15s
setInterval(checkHealth, 15000);
checkHealth();

/* ══════════════════════════════════════════════════════════════════════════════
   Sidebar toggle
══════════════════════════════════════════════════════════════════════════════ */
const sidebar = document.querySelector('.sidebar');



mobileSidebarToggle.addEventListener('click', () => {
  if (window.innerWidth <= 640) {
    sidebar.classList.toggle('mobile-open');
  } else {
    sidebar.classList.toggle('collapsed');
  }
});

/* ══════════════════════════════════════════════════════════════════════════════
   Utility
══════════════════════════════════════════════════════════════════════════════ */
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

/* ══════════════════════════════════════════════════════════════════════════════
   Chat Management
══════════════════════════════════════════════════════════════════════════════ */
async function loadChats() {
  try {
    const res = await fetch('/api/chats');
    const data = await res.json();
    chatList.innerHTML = '';
    
    const rawChats = data.chats || data.threads || [];
    const chats = rawChats.map(item => {
      if (typeof item === 'string') return { id: item, title: `${item.substring(0, 8)}...` };
      const id = item.id || item.thread_id || item.threadId;
      return { id, title: item.title || `${id.substring(0, 8)}...` };
    }).filter(item => item.id);

    if (chats.length > 0) {
      chats.forEach(chat => {
        const row = document.createElement('div');
        row.className = `chat-row${chat.id === currentThreadId ? ' active' : ''}`;
        row.title = chat.id;

        const open = document.createElement('button');
        open.className = 'chat-name';
        open.type = 'button';
        open.textContent = chat.title;
        open.title = `Open ${chat.title}`;
        open.addEventListener('click', () => {
          if (chat.id !== currentThreadId) window.location.href = `?chat=${chat.id}`;
        });

        const actions = document.createElement('span');
        actions.className = 'chat-actions';

        const rename = document.createElement('button');
        rename.className = 'chat-action';
        rename.type = 'button';
        rename.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 16-.8 4.8L8 20l11.2-11.2a2.1 2.1 0 0 0-3-3L4 17Z"/><path d="m14.8 7.2 2 2"/></svg>';
        rename.title = 'Rename chat';
        rename.setAttribute('aria-label', 'Rename chat');
        rename.addEventListener('click', async event => {
          event.stopPropagation();
          const title = window.prompt('Enter a name for this chat:', chat.title);
          if (title === null) return;
          const trimmed = title.trim();
          if (!trimmed) return;
          try {
            const res = await fetch(`/api/chats/${encodeURIComponent(chat.id)}`, {
              method: 'PATCH',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ title: trimmed }),
            });
            if (!res.ok) throw new Error(`Rename failed (${res.status})`);
            await loadChats();
          } catch (error) {
            console.error('Failed to rename chat', error);
            window.alert('Could not rename this chat.');
          }
        });

        const remove = document.createElement('button');
        remove.className = 'chat-action danger';
        remove.type = 'button';
        remove.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M9 7V4h6v3l-1 13H10L9 7Z"/></svg>';
        remove.title = 'Delete chat';
        remove.setAttribute('aria-label', 'Delete chat');
        remove.addEventListener('click', async event => {
          event.stopPropagation();
          if (!window.confirm(`Delete chat “${chat.title}”? This cannot be undone.`)) return;
          try {
            const res = await fetch(`/api/chats/${encodeURIComponent(chat.id)}`, { method: 'DELETE' });
            if (!res.ok) throw new Error(`Delete failed (${res.status})`);
            if (chat.id === currentThreadId) {
              window.location.href = `?chat=${crypto.randomUUID()}`;
            } else {
              await loadChats();
            }
          } catch (error) {
            console.error('Failed to delete chat', error);
            window.alert('Could not delete this chat.');
          }
        });

        actions.append(rename, remove);
        row.append(open, actions);
        chatList.appendChild(row);
      });
    } else {
      chatList.innerHTML = '<p class="log-empty">No conversations yet.</p>';
    }
  } catch (e) {
    chatList.innerHTML = '<p class="log-empty">Failed to load.</p>';
  }
}

newChatBtn.addEventListener('click', () => {
  window.location.href = `?chat=${crypto.randomUUID()}`;
});

/* ══════════════════════════════════════════════════════════════════════════════
   Init
══════════════════════════════════════════════════════════════════════════════ */
loadModels();
loadChats();
loadChatHistory().then(() => {
  connect();
  updateInputState();
});
