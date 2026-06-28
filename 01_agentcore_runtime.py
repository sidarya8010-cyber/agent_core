import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = BedrockAgentCoreApp()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


def handle_customer_question(question: str) -> str:
    try:
        response = llm.invoke(
            [
                (
                    "system",
                    "You are a friendly customer support assistant. Answer customer questions clearly and politely. Do not use tools. If you are unsure, say so briefly and suggest contacting support.",
                ),
                ("human", question),
            ]
        )
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        return f"I’m sorry, I could not answer right now. Error: {exc}"


@app.entrypoint
def agent_invocation(payload, context):
    query = payload.get("prompt", "") if isinstance(payload, dict) else str(payload)
    if not query:
        query = "what is genai"

    if any(keyword in query.lower() for keyword in ["cancel", "subscription", "refund", "billing", "charge", "support"]):
        response = handle_customer_question(query)
    else:
        response = f"Runtime is working. Received prompt: {query}"

    return {"result": response}


if __name__ == "__main__":
    app.run()
