import os

from dotenv import load_dotenv
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

_ = load_dotenv()
app = BedrockAgentCoreApp()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = """You are a helpful FAQ assistant.

Your goal is to answer user questions accurately and directly.

Guidelines:
1. Provide a clear, concise answer to the user's question.
2. Do not call any external tools.
3. Do not attempt retrieval-augmented generation.
4. If you do not know the answer, say so clearly.

Think step-by-step and keep the response focused."""


@app.entrypoint
def agent_invocation(payload, context):
    query = payload.get("prompt", "") if isinstance(payload, dict) else str(payload)
    if not query:
        query = "what is genai"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]
    result = model.invoke(messages)
    return {"result": result.content}


if __name__ == "__main__":
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="what is genai?"),
    ]
    result = model.invoke(messages)
    print(result.content)
