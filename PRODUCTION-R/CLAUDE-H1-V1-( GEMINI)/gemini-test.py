from google import genai
import time

def run_with_fallback(api_key_var, user_prompt):
    client = genai.Client(api_key=api_key_var)
    
    # Define your priority list (Plan A, Plan B, Plan C)
    model_priority = [
        "gemini-2.5-flash",        # Plan A: Top Choice
        "gemini-2.5-flash-lite",   # Plan B: Fast & Reliable
        "gemini-2.0-flash",        # Plan C: Very Stable fallback
        "gemini-1.5-flash"         # Plan D: Final fallback
    ]

    for model_id in model_priority:
        try:
            print(f"Trying {model_id}...")
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt
            )
            print(f"Success using {model_id}!")
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            # Only fallback if it's a server error (503, 500, 429)
            if "503" in error_msg or "500" in error_msg or "429" in error_msg:
                print(f"Server busy on {model_id}. Switching to next available...")
                continue 
            else:
                # If it's a coding error or safety block, don't fallback, just stop.
                return f"Critical Error on {model_id}: {e}"

    return "All models failed to respond. Please try again later."

# --- CONFIG ---
MY_API_KEY = "AIzaSyCraR3GRXVHLhYTzLLZozhZfT-qESjtmFo"

if __name__ == "__main__":
    prompt = "Write a one-sentence summary of why AI is useful."
    result = run_with_fallback(MY_API_KEY, prompt)
    print("\nFinal Result:")
    print(result)