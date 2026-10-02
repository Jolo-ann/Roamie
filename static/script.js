const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatBox = document.getElementById('chat-box');

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage(message, 'user-message');
    userInput.value = '';

    const typingIndicator = showTypingIndicator();

    try {
        const response = await fetch('/get_response', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        
        removeTypingIndicator(typingIndicator);
        appendMessage(data.reply, 'bot-message');
    } catch (error) {
        removeTypingIndicator(typingIndicator);
        appendMessage("Error: Could not process request.", 'bot-message');
    }
});

function appendMessage(text, className) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    
    let formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
        .replace(/\n/g, '<br>')
        .replace(/---/g, '<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 10px 0;">');

    messageDiv.innerHTML = `<div class="message-content">${formattedText}</div>`;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function showTypingIndicator() {
    const indicatorDiv = document.createElement('div');
    indicatorDiv.classList.add('message', 'bot-message', 'typing-indicator');
    indicatorDiv.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    chatBox.appendChild(indicatorDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return indicatorDiv;
}

function removeTypingIndicator(element) {
    if (element && element.parentNode) {
        element.parentNode.removeChild(element);
    }
}