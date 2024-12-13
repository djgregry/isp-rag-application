document.getElementById("chat-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    // Get user input
    const userInput = document.getElementById("user-input").value.trim();

    // Clear input field
    document.getElementById("user-input").value = "";

    // Skip empty messages
    if (userInput === "") 
        return;

    // Append the user message to the conversation
    appendMessage(userInput, 'user')

    // Get the chat-container element
    const chatContainer = document.querySelector('.chat-container');

    // Collect all <p> elements inside chat-container
    const chats = chatContainer.querySelectorAll('p.bot, p.user');

    // Convert to an array
    const chatArray = Array.from(chats).map(chat => ({
        role: chat.classList.contains('bot') ? 'assistant' : 'user',
        content: chat.textContent.trim()
    }));

    // Generate bot response
    const botResponse = await generateBotResponse(chatArray)
    
    // Append bot response to the conversation
    appendMessage(`${botResponse.response}`, 'bot');
});


function appendMessage(message, sender) {
    const chatContainer = document.querySelector(".chat-container");

    // Create a paragraph to hold the message
    const messageText = document.createElement("p");
    messageText.classList.add(sender); // Adds 'user' or 'bot' class
    messageText.textContent = message;

    // Append the message to the chat container
    chatContainer.appendChild(messageText);

    // Scroll to the bottom of the chat container
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Function to generate bot response by making an API call
async function generateBotResponse(chat) {
    try {
        // Send a POST request to the server with the chat history as JSON
        const response = await fetch("http://127.0.0.1:8000/generate-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ "chat": chat })
        });

        // Check the response is successful
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const json = await response.json();

        return json;

    } catch (error) {
        console.error('Error:', error);
        return "An error occurred.";
    }
}