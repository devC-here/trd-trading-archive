import google.generativeai as genai
import json
import os

# === CONFIG ===
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Gemini API Key
# You can get one from Google AI Studio: https://makersuite.google.com/
API_KEY = "AIzaSyDq0P1iGOp4cT9kiqFIxozvmcL_hCPSXQo" 
HISTORY_FILE = "chat_history.json" # This file won't be used by get_gemini_trade_advice but is kept for the original main()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# === Load saved history or create empty ===
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# === Save updated history ===
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_gemini_trade_advice(prompt_text: str) -> str:
    """
    Sends a structured prompt to the Gemini model to get trade advice.
    This function does NOT manage conversational history; it's for one-off queries.

    Args:
        prompt_text (str): The full prompt to send to the Gemini model.

    Returns:
        str: The text response from the Gemini model, or an error message.
    """
    try:
        # For a one-off advice, we just send the prompt directly.
        # If you needed conversational memory for this specific function,
        # you would pass a history list here.
        response = model.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        print(f"Error getting response from Gemini: {e}")
        return f"AI Advice Error: {e}"

# === Original Main Chat Logic (kept if you still want an interactive chat) ===
def main():
    history = load_history()

    print("🧠 Gemini Chat – Memory Enabled (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Add user message
        history.append({"role": "user", "parts": [user_input]})

        # Get Gemini response
        response = model.generate_content(history)
        print("Gemini:", response.text.strip())

        # Add Gemini reply to history
        history.append({"role": "model", "parts": [response.text]})

        # Save updated history
        save_history(history)

if __name__ == "__main__":
    main()