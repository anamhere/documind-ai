/**
 * DocuMind AI — Frontend Application Logic
 * ==========================================
 * Handles chat interactions, file uploads, document management,
 * and communication with the FastAPI backend.
 */

// ─── Configuration ──────────────────────────────────────────────────────────
const API_BASE = window.location.origin;

// ─── State Management ───────────────────────────────────────────────────────
const state = {
    chatHistory: [],      // Array of { role: 'user'|'assistant', content: string }
    documents: [],        // Array of document objects
    isLoading: false,     // Whether we're waiting for a response
    totalChunks: 0,       // Total chunks across all documents
    pollingInterval: null // Interval for background refresh
};

// ─── DOM Elements ───────────────────────────────────────────────────────────
const elements = {
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    fileInput: document.getElementById('fileInput'),
    uploadDropzone: document.getElementById('uploadDropzone'),
    uploadBtn: document.getElementById('uploadBtn'),
    uploadProgress: document.getElementById('uploadProgress'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    documentList: document.getElementById('documentList'),
    docCount: document.getElementById('docCount'),
    emptyDocs: document.getElementById('emptyDocs'),
    headerStatus: document.getElementById('headerStatus'),
    welcomeContainer: document.getElementById('welcomeContainer'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    sidebarToggle: document.getElementById('sidebarToggle'),
    sidebar: document.getElementById('sidebar'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    totalChunks: document.getElementById('totalChunks'),
    indexSize: document.getElementById('indexSize'),
    bgParticles: document.getElementById('bgParticles'),
    summarizeAllBtn: document.getElementById('summarizeAllBtn'),
};

// ─── Initialize Application ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    initBackgroundParticles();
    loadDocuments();
});

// ─── Event Listeners ────────────────────────────────────────────────────────
function initEventListeners() {
    // Chat input — send on Enter (Shift+Enter for new line)
    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    elements.chatInput.addEventListener('input', () => {
        elements.chatInput.style.height = 'auto';
        elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 120) + 'px';
        elements.sendBtn.disabled = !elements.chatInput.value.trim();
    });

    // Send button click
    elements.sendBtn.addEventListener('click', sendMessage);

    // Upload button click
    elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());

    // Dropzone click
    elements.uploadDropzone.addEventListener('click', () => elements.fileInput.click());

    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.uploadDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadDropzone.classList.add('drag-over');
    });

    elements.uploadDropzone.addEventListener('dragleave', () => {
        elements.uploadDropzone.classList.remove('drag-over');
    });

    elements.uploadDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadDropzone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFiles(files);
        }
    });

    // Clear chat
    elements.clearChatBtn.addEventListener('click', clearChat);

    // Summarize all documents
    if (elements.summarizeAllBtn) {
        elements.summarizeAllBtn.addEventListener('click', summarizeAllDocs);
    }

    // Sidebar toggle
    elements.sidebarToggle.addEventListener('click', toggleSidebar);
    elements.mobileMenuBtn.addEventListener('click', toggleMobileSidebar);
}

// ─── Background Particles ───────────────────────────────────────────────────
function initBackgroundParticles() {
    for (let i = 0; i < 15; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        particle.style.left = Math.random() * 100 + '%';
        particle.style.width = Math.random() * 4 + 2 + 'px';
        particle.style.height = particle.style.width;
        particle.style.animationDuration = Math.random() * 15 + 10 + 's';
        particle.style.animationDelay = Math.random() * 10 + 's';
        elements.bgParticles.appendChild(particle);
    }
}

// ─── Chat Functions ─────────────────────────────────────────────────────────
async function sendMessage() {
    const query = elements.chatInput.value.trim();
    if (!query || state.isLoading) return;

    // Hide welcome screen
    if (elements.welcomeContainer) {
        elements.welcomeContainer.remove();
    }

    // Add user message to UI
    addMessageToUI('user', query);
    state.chatHistory.push({ role: 'user', content: query });

    // Clear input
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    elements.sendBtn.disabled = true;

    // Show typing indicator
    const typingEl = showTypingIndicator();

    // Update status
    setStatus('Thinking...', true);
    state.isLoading = true;

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                chat_history: state.chatHistory.slice(-10), // Send last 10 messages
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to get response');
        }

        const data = await response.json();

        // Remove typing indicator
        typingEl.remove();

        // Refresh documents if the /web command triggered
        if (query.startsWith("/web ")) {
            await loadDocuments();
        }

        // Add assistant message to UI
        addMessageToUI('assistant', data.answer, data.sources);
        state.chatHistory.push({ role: 'assistant', content: data.answer });

        // Update status
        const statusMsg = data.has_context
            ? `Answered using ${data.chunks_used} context chunks from ${data.sources.length} document(s)`
            : 'Ready — Upload a document to begin';
        setStatus(statusMsg, false);

    } catch (error) {
        typingEl.remove();
        addMessageToUI('assistant', `⚠️ Error: ${error.message}. Please try again.`);
        setStatus('Error occurred — please try again', false);
        showToast('error', error.message);
    }

    state.isLoading = false;
}

function addMessageToUI(role, content, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user'
        ? 'You'
        : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><circle cx="12" cy="15" r="2"/></svg>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (role === 'assistant') {
        contentDiv.innerHTML = renderMarkdown(content);
    } else {
        contentDiv.textContent = content;
    }

    // Add source citations
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sources.forEach(source => {
            const tag = document.createElement('span');
            tag.className = 'source-tag';
            tag.textContent = `📄 ${source}`;
            sourcesDiv.appendChild(tag);
        });
        contentDiv.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    elements.chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typingIndicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><circle cx="12" cy="15" r="2"/></svg>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

    contentDiv.appendChild(typing);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    elements.chatMessages.appendChild(messageDiv);

    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    return messageDiv;
}

// ─── Markdown Rendering (Simple) ────────────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return '';

    let html = text
        // Escape HTML first
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')

        // Code blocks (```code```)
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')

        // Inline code (`code`)
        .replace(/`([^`]+)`/g, '<code>$1</code>')

        // Bold (**text**)
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')

        // Italic (*text*)
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')

        // Links ([text](url)) - Specifically needed for /export downloads!
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="download-link" style="color: #a78bfa; text-decoration: underline; font-weight: 500;">$1</a>')

        // Headers
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')

        // Blockquotes
        .replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>')

        // Unordered lists
        .replace(/^[-*] (.+)$/gm, '<li>$1</li>')

        // Ordered lists
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')

        // Line breaks & paragraphs
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    // Wrap consecutive <li> items in <ul>
    html = html.replace(/(<li>.*?<\/li>)(\s*<br>?\s*<li>)/g, '$1$2');
    html = html.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
    // Clean up nested <ul> tags
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Wrap in paragraph
    html = '<p>' + html + '</p>';

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>\s*(<h[1-3]>)/g, '$1');
    html = html.replace(/(<\/h[1-3]>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<blockquote>)/g, '$1');
    html = html.replace(/(<\/blockquote>)\s*<\/p>/g, '$1');

    return html;
}

// ─── File Upload Functions ──────────────────────────────────────────────────
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFiles(files);
    }
    // Reset file input so the same file can be re-selected
    e.target.value = '';
}

async function handleFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    // Validate file type
    const allowedTypes = ['.pdf', '.txt', '.md', '.csv'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(ext)) {
        showToast('error', `Unsupported file type: ${ext}. Allowed: ${allowedTypes.join(', ')}`);
        return;
    }

    // Show progress
    elements.uploadProgress.style.display = 'block';
    elements.progressFill.style.width = '20%';
    elements.progressText.textContent = `Processing "${file.name}"...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        elements.progressFill.style.width = '50%';
        elements.progressText.textContent = 'Extracting text & generating embeddings...';

        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await response.json();

        elements.progressFill.style.width = '100%';
        elements.progressText.textContent = 'Done!';

        showToast('success', data.message);

        // Reload document list
        await loadDocuments();

        // Auto-close sidebar on mobile after successful upload
        if (window.innerWidth <= 768) {
            const overlay = document.querySelector('.sidebar-overlay');
            if (overlay && overlay.classList.contains('active')) {
                toggleMobileSidebar();
            }
        }

        // Update status
        setStatus(`${state.documents.length} document(s) indexed — Ask me anything!`);

    } catch (error) {
        showToast('error', `Upload failed: ${error.message}`);
    }

    // Hide progress after a delay
    setTimeout(() => {
        elements.uploadProgress.style.display = 'none';
        elements.progressFill.style.width = '0%';
    }, 1500);
}

// ─── Document Management ────────────────────────────────────────────────────
async function summarizeAllDocs() {
    if (state.isLoading) return;
    
    state.isLoading = true;
    elements.summarizeAllBtn.classList.add('loading');
    setStatus('Summarizing workspace documents...', true);
    
    try {
        const response = await fetch(`${API_BASE}/api/summarize-all`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) throw new Error('Failed to summarize documents.');
        
        const data = await response.json();
        showToast('success', data.message);
        
        // Refresh the document list to show new summaries
        await loadDocuments();
        
    } catch (error) {
        console.error('Summarization error:', error);
        showToast('error', error.message);
    } finally {
        elements.summarizeAllBtn.classList.remove('loading');
        setStatus('Ready — Upload a document to begin', false);
        
        // Auto-close sidebar on mobile after summarization
        if (window.innerWidth <= 768) {
            const overlay = document.querySelector('.sidebar-overlay');
            if (overlay && overlay.classList.contains('active')) {
                toggleMobileSidebar();
            }
        }
    }
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/api/documents`);
        if (!response.ok) throw new Error('Failed to load documents');

        const data = await response.json();
        state.documents = Array.isArray(data) ? data : (data.documents || []);
        state.totalChunks = data.total_chunks || state.documents.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
        
        renderDocumentList();
        updateStats();

        // Start polling if any document is still "Summarizing..."
        startPollingIfProcessing();

    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

/**
 * Intelligent Background Polling
 * Refreshes the document list while AI is working in the background.
 */
function startPollingIfProcessing() {
    const isProcessing = state.documents.some(doc => 
        doc.summary && (
            doc.summary.toLowerCase().includes('summarizing') || 
            doc.summary.toLowerCase().includes('processing') ||
            doc.summary.toLowerCase().includes('scraping')
        )
    );

    if (isProcessing && !state.pollingInterval) {
        console.log("RAG: AI is working in background. Starting UI polling...");
        state.pollingInterval = setInterval(async () => {
            await loadDocuments();
        }, 3000);
    } else if (!isProcessing && state.pollingInterval) {
        console.log("RAG: All documents summarized. Stopping polling.");
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
    }
}
function renderDocumentList() {
    // Update count
    elements.docCount.textContent = state.documents.length;

    // Clear existing items (keep empty state)
    const existingItems = elements.documentList.querySelectorAll('.doc-item');
    existingItems.forEach(item => item.remove());

    if (state.documents.length === 0) {
        elements.emptyDocs.style.display = 'flex';
        return;
    }

    elements.emptyDocs.style.display = 'none';

    state.documents.forEach(doc => {
        const ext = (doc.file_type || '.txt').replace('.', '').toUpperCase();
        const extClass = (doc.file_type || '.txt').replace('.', '').toLowerCase();

        const docEl = document.createElement('div');
        docEl.className = 'doc-item';
        docEl.innerHTML = `
            <div class="doc-icon ${extClass}">${ext}</div>
            <div class="doc-info">
                <div class="doc-name" title="${doc.original_filename || doc.file_name}">${doc.original_filename || doc.file_name}</div>
                <div class="doc-meta">${doc.chunk_count} chunks · ${doc.file_size_readable || ''}</div>
                ${doc.summary && doc.summary !== "Summary not available." ? `<div class="doc-summary" style="font-size: 0.75rem; color: #a78bfa; margin-top: 4px; line-height: 1.3; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;" title="${doc.summary}">✨ ${doc.summary}</div>` : ''}
            </div>
            <button class="doc-delete" title="Delete document" data-id="${doc.doc_id}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                </svg>
            </button>
        `;

        // Delete button handler
        docEl.querySelector('.doc-delete').addEventListener('click', () => deleteDocument(doc.doc_id));

        elements.documentList.appendChild(docEl);
    });
}

async function deleteDocument(docId) {
    try {
        const response = await fetch(`${API_BASE}/api/documents/${docId}`, {
            method: 'DELETE',
        });

        if (!response.ok) throw new Error('Failed to delete document');

        showToast('success', 'Document deleted');
        await loadDocuments();

        if (state.documents.length === 0) {
            setStatus('Ready — Upload a document to begin');
        }

    } catch (error) {
        showToast('error', `Delete failed: ${error.message}`);
    }
}

// ─── UI Helper Functions ────────────────────────────────────────────────────
function setStatus(message, isProcessing = false) {
    const statusEl = elements.headerStatus;
    statusEl.innerHTML = `
        <span class="status-dot" style="${isProcessing ? 'background: var(--warning); box-shadow: 0 0 8px rgba(251,191,36,0.5);' : ''}"></span>
        <span>${message}</span>
    `;
}

function updateStats() {
    elements.totalChunks.textContent = state.totalChunks;
    elements.indexSize.textContent = `${state.totalChunks} vectors`;
}

function clearChat() {
    state.chatHistory = [];
    elements.chatMessages.innerHTML = '';

    // Re-add welcome screen
    const welcome = document.createElement('div');
    welcome.className = 'welcome-container';
    welcome.id = 'welcomeContainer';
    welcome.innerHTML = `
        <div class="welcome-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="url(#welcomeGrad)" stroke-width="1.5">
                <defs>
                    <linearGradient id="welcomeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#818cf8"/>
                        <stop offset="100%" style="stop-color:#c084fc"/>
                    </linearGradient>
                </defs>
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <circle cx="12" cy="15" r="3"/>
                <path d="M12 12v-1"/>
            </svg>
        </div>
        <h2>Welcome to DocuMind AI</h2>
        <p>Upload a document and ask questions about it. I'll find the answers using intelligent document analysis.</p>
        <div class="welcome-features">
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <h4>Upload Documents</h4>
                <p>PDF, TXT, MD, CSV</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h4>Smart Search</h4>
                <p>Semantic-powered retrieval</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💡</div>
                <h4>AI Answers</h4>
                <p>Context-grounded responses</p>
            </div>
        </div>
    `;
    elements.chatMessages.appendChild(welcome);
    showToast('info', 'Chat cleared');
}

function toggleSidebar() {
    elements.sidebar.classList.toggle('collapsed');
}

function toggleMobileSidebar() {
    elements.sidebar.classList.toggle('mobile-open');

    // Handle overlay
    let overlay = document.querySelector('.sidebar-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        document.body.appendChild(overlay);
        overlay.addEventListener('click', () => {
            elements.sidebar.classList.remove('mobile-open');
            overlay.classList.remove('active');
        });
    }
    overlay.classList.toggle('active');
}

// ─── Toast Notifications ────────────────────────────────────────────────────
function showToast(type, message) {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️',
    };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
        <span class="toast-message">${message}</span>
    `;

    container.appendChild(toast);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
