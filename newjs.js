// RAG System Classes
class Document {
    constructor(text, metadata = {}) {
        this.text = text;
        this.metadata = metadata;
    }
}

class VectorDB {
    constructor() {
        this.chunks = [];
        this.embeddings = [];
        this.metadata = [];
    }
    
    add(chunk, embedding, metadata = {}) {
        this.chunks.push(chunk);
        this.embeddings.push(embedding);
        this.metadata.push(metadata);
    }
    
    search(queryEmbedding, topK = 3) {
        if (this.embeddings.length === 0) {
            return [];
        }
        
        // Calculate cosine similarity between query and all embeddings
        const similarities = this.embeddings.map(emb => this.cosineSimilarity(queryEmbedding, emb));
        
        // Get indices of top k similar embeddings
        const indices = this.argSort(similarities).slice(-topK).reverse();
        
        return indices.map(i => ({
            chunk: this.chunks[i],
            similarity: similarities[i],
            metadata: this.metadata[i]
        }));
    }
    
    // Helper function to calculate cosine similarity
    cosineSimilarity(vecA, vecB) {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }
    
    // Helper function to get indices that would sort an array
    argSort(array) {
        return array
            .map((value, index) => ({ value, index }))
            .sort((a, b) => a.value - b.value)
            .map(item => item.index);
    }
    
    // Get chunk count
    get count() {
        return this.chunks.length;
    }
    
    // Get unique document sources
    get sources() {
        return [...new Set(this.metadata
            .filter(meta => meta.source)
            .map(meta => meta.source))];
    }
    
    // Save to localStorage
    save(key = 'ragVectorDB') {
        localStorage.setItem(key, JSON.stringify({
            chunks: this.chunks,
            embeddings: this.embeddings,
            metadata: this.metadata
        }));
    }
    
    // Load from localStorage
    load(key = 'ragVectorDB') {
        const data = JSON.parse(localStorage.getItem(key));
        if (data) {
            this.chunks = data.chunks;
            this.embeddings = data.embeddings;
            this.metadata = data.metadata;
            return true;
        }
        return false;
    }
}

class RAGSystem {
    constructor() {
        this.vectorDB = new VectorDB();
        this.documents = [];
        this.apiKey = 'sk-5f79657cfd4748de8929adfbe6d06ea7';
        this.endpoint = 'https://api.deepseek.com/chat/completions';
        this.embeddingEndpoint = 'https://api.deepseek.com/embeddings';
    }
    
    // Process and index a document
    async addDocument(document) {
        this.documents.push(document);
        const chunks = this.chunkDocument(document);
        
        for (const chunk of chunks) {
            const embedding = await this.getEmbedding(chunk.text);
            this.vectorDB.add(chunk.text, embedding, chunk.metadata);
        }
        
        // Save updated vector DB
        this.vectorDB.save();
        
        // Update status bar
        this.updateStatusBar();
        
        return chunks.length;
    }
    
    // Split document into chunks
    chunkDocument(document, chunkSize = 200, overlap = 50) {
        const words = document.text.split(/\s+/);
        const chunks = [];
        
        for (let i = 0; i < words.length; i += chunkSize - overlap) {
            const chunkText = words.slice(i, i + chunkSize).join(' ');
            // Copy metadata and add chunk position info
            const chunkMetadata = {...document.metadata};
            chunkMetadata.chunkId = chunks.length;
            chunkMetadata.startIdx = i;
            chunkMetadata.endIdx = Math.min(i + chunkSize, words.length);
            
            chunks.push({
                text: chunkText,
                metadata: chunkMetadata
            });
        }
        
        return chunks;
    }
    
    // Get embedding for text
    async getEmbedding(text) {
        try {
            const response = await fetch(this.embeddingEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    model: "deepseek-embeddings",
                    input: text
                })
            });
            
            const data = await response.json();
            if (data.data && data.data[0] && data.data[0].embedding) {
                return data.data[0].embedding;
            }
            
            // If API doesn't work, return a random embedding for testing
            console.warn("Using fallback random embedding");
            return Array.from({length: 384}, () => Math.random());
            
        } catch (error) {
            console.error("Error getting embedding:", error);
            // Return a random embedding for testing
            return Array.from({length: 384}, () => Math.random());
        }
    }
    
    // Retrieve relevant context for a query
    async retrieve(query, topK = 3) {
        const queryEmbedding = await this.getEmbedding(query);
        const results = this.vectorDB.search(queryEmbedding, topK);
        
        if (results.length === 0) {
            return "No relevant information found.";
        }
        
        // Format retrieved chunks with source information
        const formattedContext = results.map((result, i) => {
            const sourceInfo = result.metadata.source ? 
                `[来源: ${result.metadata.source}]` : "";
            return `片段 ${i+1} ${sourceInfo} (相关度: ${result.similarity.toFixed(2)}):\n${result.chunk}`;
        });
        
        return formattedContext.join("\n\n");
    }
    
    // Generate a response based on query and context
    async generate(query, context) {
        const prompt = `
        根据以下检索到的信息来回答用户的问题。
        如果信息中不包含答案，请回复"我没有足够的信息来回答这个问题。"

        检索到的信息:
        ${context}

        用户问题: ${query}
        
        回答:
        `;
        
        try {
            const response = await fetch(this.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    model: "deepseek-chat",
                    messages: [
                        { role: "system", content: "你是一个有帮助的助手，基于提供的信息来回答问题。" },
                        { role: "user", content: prompt }
                    ],
                    stream: false
                })
            });
            
            const data = await response.json();
            if (data.choices && data.choices.length > 0) {
                return data.choices[0].message.content;
            } else {
                return "生成回答失败，请重试。";
            }
        } catch (error) {
            console.error("Error generating response:", error);
            return "生成回答时发生错误。";
        }
    }
    
    // Complete RAG pipeline: retrieve + generate
    async query(query) {
        const context = await this.retrieve(query);
        const answer = await this.generate(query, context);
        return { answer, context };
    }
    
    // Update status bar information
    updateStatusBar() {
        const documentStatus = document.getElementById('document-status');
        const chunkCount = document.getElementById('chunk-count');
        
        if (documentStatus && chunkCount) {
            const sources = this.vectorDB.sources;
            documentStatus.textContent = sources.length > 0 ? 
                `已加载文档: ${sources.join(', ')}` : 
                '未加载文档';
                
            chunkCount.textContent = `${this.vectorDB.count} 个片段`;
        }
    }
    
    // Clear all data
    clear() {
        this.vectorDB = new VectorDB();
        this.documents = [];
        localStorage.removeItem('ragVectorDB');
        this.updateStatusBar();
    }
}

// Format message text
function formatMessage(text) {
    if (!text) return '';
    
    // Process markdown-style code blocks
    text = text.replace(/```([a-z]*)\n([\s\S]*?)```/g, function(match, language, code) {
        return `<div class="code-block"><div class="code-header">${language || 'code'}</div><pre>${code}</pre></div>`;
    });
    
    // Handle bold text
    let lines = text.split('\n');
    let formattedLines = lines.map(line => {
        // Process bold text (**text**)
        line = line.replace(/\*\*(.*?)\*\*/g, '<span class="bold-text">$1</span>');
        return line;
    });
    
    // Process sections
    let processedText = formattedLines.join('\n');
    let sections = processedText
        .split('###')
        .filter(section => section.trim())
        .map(section => {
            // Remove extra line breaks and spaces
            let lines = section.split('\n').filter(line => line.trim());
            
            if (lines.length === 0) return '';
            
            // Process each section
            let result = '';
            let currentIndex = 0;
            
            while (currentIndex < lines.length) {
                let line = lines[currentIndex].trim();
                
                // If starts with a number (like "1.")
                if (/^\d+\./.test(line)) {
                    result += `<p class="section-title">${line}</p>`;
                }
                // If it's a subtitle (starts with a dash)
                else if (line.startsWith('-')) {
                    result += `<p class="subsection"><span class="bold-text">${line.replace(/^-/, '').trim()}</span></p>`;
                }
                // If it's body text (contains a colon)
                else if (line.includes(':')) {
                    let [subtitle, content] = line.split(':').map(part => part.trim());
                    result += `<p><span class="subtitle">${subtitle}</span>: ${content}</p>`;
                }
                // Regular text
                else {
                    result += `<p>${line}</p>`;
                }
                currentIndex++;
            }
            return result;
        });
    
    return sections.join('');
}

// Global RAG instance
const ragSystem = new RAGSystem();

// Display message
function displayMessage(role, message, isRagContext = false) {
    const messagesContainer = document.getElementById('messages');
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role}`;
    
    const avatar = document.createElement('img');
    avatar.src = role === 'user' ? 'user-avatar.png' : 'bot-avatar.png';
    avatar.alt = role === 'user' ? 'User' : 'Bot';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (isRagContext) {
        messageContent.className += ' rag-context';
        messageContent.innerHTML = `<div class="context-header">检索到的相关内容</div><div class="context-content">${formatMessage(message)}</div>`;
    } else {
        // User messages shown directly, bot messages need formatting
        messageContent.innerHTML = role === 'user' ? message : formatMessage(message);
    }

    messageElement.appendChild(avatar);
    messageElement.appendChild(messageContent);
    messagesContainer.appendChild(messageElement);
    
    // Smooth scroll to bottom
    messageElement.scrollIntoView({ behavior: 'smooth' });
}

// Upload and process document
async function uploadDocument() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.txt,.pdf,.docx';
    
    fileInput.onchange = async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        displayMessage('user', `正在上传文档: ${file.name}`);
        
        try {
            const text = await readFileAsText(file);
            const document = new Document(text, {
                source: file.name,
                type: file.type,
                uploadedAt: new Date().toISOString()
            });
            
            displayMessage('bot', `正在处理文档: ${file.name}...`);
            
            const chunksCount = await ragSystem.addDocument(document);
            displayMessage('bot', `文档处理成功！已创建 ${chunksCount} 个检索片段。`);
        } catch (error) {
            console.error("Error processing document:", error);
            displayMessage('bot', `处理文档时出错: ${error.message}`);
        }
    };
    
    fileInput.click();
}

// Read file as text
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => resolve(event.target.result);
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
}

// Send message with RAG
async function sendMessage() {
    const inputElement = document.getElementById('chat-input');
    const message = inputElement.value;
    if (!message.trim()) return;

    displayMessage('user', message);
    inputElement.value = '';

    // Show loading animation
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.style.display = 'block';
    }

    try {
        // Use RAG to get answer
        const result = await ragSystem.query(message);
        
        // Hide loading animation
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }

        // Display context and answer
        if (result.context && result.context !== "No relevant information found.") {
            displayMessage('bot', result.context, true);
        }
        displayMessage('bot', result.answer);
    } catch (error) {
        // Hide loading animation
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }

        displayMessage('bot', '出错了，请稍后再试。');
        console.error('Error:', error);
    }
}

// Clear all RAG data
function clearRAGData() {
    ragSystem.clear();
    displayMessage('bot', '所有文档数据已清除。');
}

// Add theme toggle
function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const chatContainer = document.querySelector('.chat-container');
    const messages = document.querySelector('.messages');
    
    // Toggle container dark mode
    chatContainer.classList}
