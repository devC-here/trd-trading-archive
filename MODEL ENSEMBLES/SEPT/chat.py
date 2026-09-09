import google.generativeai as genai

# Configure directly with your API key (or use env var GOOGLE_API_KEY)
genai.configure(api_key="AIzaSyD28XS1hiMGszIKVT2Th0BKfuvaF_SvKc8")

# Pick chat-capable model
model = genai.GenerativeModel("gemini-2.5-pro")

# Start a chat session
chat = model.start_chat(history=[])

print("💬 Gemini Chat started (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Ending chat.")
        break

    response = chat.send_message(user_input)
    print("Gemini:", response.text, "\n")
