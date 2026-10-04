// 🛠️ BACKEND CONFIGURE PORT ROUTE
const BACKEND_URL = "http://127.0.0.1:8000";

// DOM Element Registry
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileDetails = document.getElementById('file-details');
const fileNameSpan = document.getElementById('file-name');
const uploadStatus = document.getElementById('upload-status');
const progressBar = document.getElementById('progress-bar');
const clearBtn = document.getElementById('clear-btn');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusText = document.getElementById('status-text');

// 1. Check if the Python Backend Server is Online on App Launch
async function checkBackendStatus() {
    try {
        const response = await fetch(`${BACKEND_URL}/`);
        if (response.ok) {
            statusText.innerText = "Backend Active";
            document.querySelector('.status-indicator').classList.add('online');
        }
    } catch (error) {
        statusText.innerText = "Offline (Run Server)";
        document.querySelector('.status-indicator').classList.remove('online');
        appendSystemMessage("🤖 System Warning: The Python server is not reachable. Please make sure uvicorn is running in your CMD terminal window.");
    }
}

// 2. Handle Document Upload File Stream Selection
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Drag and drop event listeners
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#4a90e2'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#cbd5e1'; });
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#cbd5e1';
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert("Please drop a valid PDF document. Other extensions are not supported yet.");
        return;
    }

    // Update UI Loading Frame Displays
    fileNameSpan.innerText = file.name;
    fileDetails.classList.remove('hidden');
    uploadStatus.innerText = "Splitting and vectorizing document into FAISS local index...";
    progressBar.style.backgroundColor = '#4a90e2';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${BACKEND_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            uploadStatus.innerText = `🟩 ${data.message}`;
            progressBar.style.backgroundColor = '#2ecc71';
            
            // Enable Chat inputs once knowledge data ingestion finishes successfully
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.placeholder = "Ask a question about this file...";
            
            appendSystemMessage(`🤖 *Teacher Notification:* I have finished parsing "${file.name}". Go ahead and ask me any questions about it!`);
        } else {
            throw new Error(data.detail || "Server vector processing crash.");
        }
    } catch (error) {
        uploadStatus.innerText = "❌ Processing failed.";
        progressBar.style.backgroundColor = '#e74c3c';
        alert(`Ingestion Error: ${error.message}`);
    }
}

// 3. Handle Message Transmission and Chat Responses
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Render user question bubble to chat viewport window
    appendUserMessage(text);
    userInput.value = "";

    // Show temporary AI loading placeholder dots
    const loadingId = appendLoadingMessage();

    try {
        const response = await fetch(`${BACKEND_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text })
        });

        const data = await response.json();

        // Remote temporary loading node
        document.getElementById(loadingId).remove();

        if (response.ok) {
            appendTeacherMessage(data.answer);
        } else {
            throw new Error(data.detail || "Generation failure.");
        }
    } catch (error) {
        document.getElementById(loadingId)?.remove();
        appendSystemMessage(`❌ Error getting response: ${error.message}`);
    }
}

// UI Rendering Node Generators
function appendUserMessage(message) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-msg';
    msgDiv.innerHTML = `<div class="avatar">👨‍🎓</div><div class="bubble">${escapeHtml(message)}</div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function appendTeacherMessage(message) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-msg';
    
    // Ensure message is a string before running the string replacement rule
    const stringMessage = typeof message === 'string' ? message : String(message);
    const formattedText = escapeHtml(stringMessage).replace(/\n/g, '<br>');
    
    msgDiv.innerHTML = `<div class="avatar">🤖</div><div class="bubble">${formattedText}</div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function escapeHtml(text) {
    //Ensure input is treated explicitly as string text data
    const stringInput = typeof text === 'string' ? text : String(text);
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return stringInput.replace(/[&<>"']/g, function(m) { return map[m]; });
}


function appendSystemMessage(message) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-msg';
    msgDiv.innerHTML = `<div class="avatar">⚙️</div><div class="bubble" style="font-style: italic; color:#666;">${message}</div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function appendLoadingMessage() {
    const id = 'loading-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-msg';
    msgDiv.id = id;
    msgDiv.innerHTML = `<div class="avatar">🤖</div><div class="bubble" style="color:#999;">Thinking and searching memory data...</div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }
function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}


// 4. Handle Clear Session Action
clearBtn.addEventListener('click', async () => {
    if (!confirm("Are you sure you want to wipe memory history and start a new study session?")) return;

    try {
        const response = await fetch(`${BACKEND_URL}/clear`, { method: 'POST' });
        const data = await response.json();

        if (response.ok) {
            // 1. Wipe chat screen down to initial state
            chatMessages.innerHTML = `
                <div class="message system-msg">
                    <div class="avatar">🤖</div>
                    <div class="bubble">
                        Memory wiped successfully! I am ready for a new document. Upload your next study guide on the left whenever you are ready!
                    </div>
                </div>
            `;
            
            // 2. Lock text entry fields back up safely
            userInput.disabled = true;
            sendBtn.disabled = true;
            userInput.placeholder = "Ask a targeted question about your document...";
            
            // 3. Hide ingestion display metrics panel completely
            fileDetails.classList.add('hidden');
            fileInput.value = ""; // Clear file selector memory buffer
            
            alert(data.message);
        } else {
            throw new Error(data.detail || "Server reset execution failure.");
        }
    } catch (error) {
        alert(`Clear Session Error: ${error.message}`);
    }
});

// Run connectivity test check instantly on startup
checkBackendStatus();
