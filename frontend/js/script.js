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

    // Prevent user from sending messages while generating a response
    document.getElementById("user-input").disabled = true;
    document.getElementById("chat-form").querySelector("button").disabled = true;

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

    // If url is available is present in the response, add a button
    if (botResponse.url) {
        appendLoadButton(botResponse.url);
    }

    document.getElementById("user-input").disabled = false;
    document.getElementById("chat-form").querySelector("button").disabled = false;
});


function appendLoadButton(url) {
    const chatContainer = document.querySelector(".chat-container");

    // Create a wrapper div for the button
    const buttonDiv = document.createElement("div");
    buttonDiv.classList.add("button-div");

    // Create article loading button
    const button = document.createElement("button");
    button.textContent = "Load Full Article";
    button.classList.add("load-button");

    // Add event listener to handle button clicks
    button.addEventListener("click", async function() {
        console.log(url);
        const response = await handleLoadDocuments(url);
        console.log(response);
    });

    // Append button to chat container
    buttonDiv.appendChild(button);
    chatContainer.appendChild(buttonDiv);

    // Scroll to the bottom of the chat container
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Function to load article contents into database via API call
async function handleLoadDocuments(url) {
    try {
        // Send POST request to the server with url as JSON
        const response = await fetch("http://127.0.0.1:8000/load-content", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ "url": url })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`)
        }

        const json = await response.json();

        return json;

    } catch(error) {
        console.error('Error:', error);
        return { response : "An error occurred." }
    }
}


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
        return { response: "An error occurred.", url: null};
    }
}