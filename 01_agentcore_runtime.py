from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()


@app.entrypoint
def agent_invocation(payload, context):
    query = payload.get("prompt", "") if isinstance(payload, dict) else str(payload)
    if not query:
        query = "what is genai"

    return {"result": f"Runtime is working. Received prompt: {query}"}


if __name__ == "__main__":
    response = agent_invocation({"prompt": "Explain roaming activation."}, None)
    print(response["result"])
