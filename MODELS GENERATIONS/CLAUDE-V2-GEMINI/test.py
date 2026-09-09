from google import genai

# =========================
# CONFIG
# =========================
GOOGLE_API_KEY= "AIzaSyCraR3GRXVHLhYTzLLZozhZfT-qESjtmFo"

client = genai.Client(api_key= "AIzaSyCraR3GRXVHLhYTzLLZozhZfT-qESjtmFo")

from google import genai
import time

# =========================
# CONFIGURATION
# =========================
API_KEY = GOOGLE_API_KEY  # <--- Paste your key here
client = genai.Client(api_key=API_KEY)

TRADING_PROMPT = """
You are a trading system. Return ONLY one word: BUY, SELL, or HOLD.
DATA: Momentum: bearish (0.50), Breakout: bullish (0.54), VWAP: bearish (0.90)
Decision:
"""

# =========================
# TEST RUNNER
# =========================
def test_all_available_models():
    print("🔍 Fetching all models assigned to your API key...")
    try:
        # Get the list of models
        all_models = client.models.list()
        
        # Filter for models that actually generate content (ignore embeddings/etc)
        # In the new SDK, we check if 'generate_content' is a supported method
        testable_models = [
            m.name for m in all_models 
            if "generateContent" in m.supported_generation_methods
        ]

        print(f"✅ Found {len(testable_models)} text-generation models.\n")
        print(f"{'MODEL NAME':<35} | {'STATUS':<10} | {'RESPONSE'}")
        print("-" * 65)

        for model_id in testable_models:
            try:
                # We use a short timeout for the test
                response = client.models.generate_content(
                    model=model_id,
                    contents="REPLY 'READY'"
                )
                
                # If basic check passes, run your trading prompt
                trading_response = client.models.generate_content(
                    model=model_id,
                    contents=TRADING_PROMPT
                )
                
                output = trading_response.text.strip()
                print(f"{model_id:<35} | ✅ OK       | {output}")
                
            except Exception as e:
                # Capture why it failed (e.g., "Model Not Found", "Rate Limit")
                error_msg = str(e).split('\n')[0][:30] # Keep error brief
                print(f"{model_id:<35} | ❌ FAIL     | {error_msg}...")
            
            # Small sleep to avoid hitting rate limits during the test
            time.sleep(0.5)

    except Exception as e:
        print(f"❌ Critical Error fetching list: {e}")

if __name__ == "__main__":
    test_all_available_models()