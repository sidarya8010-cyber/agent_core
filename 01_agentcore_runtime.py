import json
import os
import urllib.request

from dotenv import load_dotenv

from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
load_dotenv()


def generate_response(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is not set, so the LLM was not called."

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"LLM call failed: {exc}"


@app.entrypoint
def agent_invocation(payload, context):
    """Simple prompt-to-response AgentCore entrypoint."""
    prompt = payload.get("prompt", "") if isinstance(payload, dict) else str(payload)
    if not prompt:
        prompt = "what is genai"

    response = generate_response(prompt)
    return {"result": response}


if __name__ == "__main__":
    app.run()
