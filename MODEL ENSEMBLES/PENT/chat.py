import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Optional: Keep browser open after script ends
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_argument("--start-maximized")

# Start browser
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://chat.openai.com/")

# Step 1: Wait for user to login manually (or automate if you have credentials + 2FA bypass)
print("Please log in to ChatGPT manually...")
while "chat" not in driver.current_url:
    time.sleep(2)
print("Logged in!")

# Step 2: Wait for the chat input box to load
input_xpath = "//textarea[@data-id='root-message-input']"
while True:
    try:
        chat_input = driver.find_element(By.XPATH, input_xpath)
        break
    except:
        time.sleep(1)

# Step 3: Send a prompt
prompt = "Write a short story about a time-traveling cat."
chat_input.send_keys(prompt)
chat_input.send_keys(Keys.ENTER)

print("Prompt sent!")

# Step 4: Optional: Wait for response (basic)
time.sleep(15)  # Wait for response to render

# Optional: Scrape the last response (simplified)
responses = driver.find_elements(By.CLASS_NAME, "markdown")
if responses:
    print("Latest GPT Response:\n")
    print(responses[-1].text)
else:
    print("Could not read response.")

# Optional: Keep running or close driver.quit()
