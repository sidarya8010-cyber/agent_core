import os

from dotenv import load_dotenv
from bedrock_agentcore.runtime import BedrockAgentCoreApp

_ = load_dotenv()
app = BedrockAgentCoreApp()

MODEL = None

system_prompt = """You are a helpful FAQ assistant.

Your goal is to answer user questions accurately and directly.

Guidelines:
1. Provide a clear, concise answer to the user's question.
2. Do not call any external tools.
3. Do not attempt retrieval-augmented generation.
4. If you do not know the answer, say so clearly.

Think step-by-step and keep the response focused."""


def get_model():
    global MODEL
    if MODEL is None:
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_groq import ChatGroq

        MODEL = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return MODEL


@app.entrypoint
def agent_invocation(payload, context):
    query = payload.get("prompt", "") if isinstance(payload, dict) else str(payload)
    if not query:
        query = "what is genai"

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]
    # model = get_model()
    # result = model.invoke(messages)
    return {"result": "result is completed"}


if __name__ == "__main__":
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="what is genai?"),
    ]
    model = get_model()
    result = model.invoke(messages)
    print(result.content)
