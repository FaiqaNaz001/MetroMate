import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from your .env file

request = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=json.dumps({
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    }).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["GROQ_API_KEY"],
        "User-Agent": "MetroChat/0.1",
    },
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        reply = json.load(response)
    print(reply["choices"][0]["message"]["content"])
except urllib.error.HTTPError as error:
    print("Groq returned an error:", error.code)
    print(error.read().decode("utf-8"))