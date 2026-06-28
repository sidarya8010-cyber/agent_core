import os

from dotenv import load_dotenv
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_groq import ChatGroq
from langchain.agents import create_agent

_ = load_dotenv()
app = BedrockAgentCoreApp()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = """You are a helpful FAQ assistant with access to a knowledge base.

Your goal is to answer user questions accurately using the available tools.

Guidelines:
1. Start by using the search_faq tool to find relevant information
2. If the initial search doesn't provide enough info, use search_detailed_faq for more results
3. If the query is complex, use reformulate_query to search different aspects
4. Synthesize information from multiple tool calls if needed
5. Always provide a clear, concise answer based on the retrieved information
6. If you cannot find relevant information, clearly state that

Think step-by-step and use tools strategically to provide the best answer."""

agent = create_agent(
    model=model,
    system_prompt=system_prompt
)


@app.entrypoint
def agent_invocation(payload, context):
    query = payload.get("prompt", "") if isinstance(payload, dict) else str(payload)
    if not query:
        query = "what is genai"

    result = agent.invoke({"messages": [("human", query)]})
    return {"result": result["messages"][-1].content}


if __name__ == "__main__":
    result = agent.invoke({"messages": [("human", "Explain roaming activation.")]})
    print(result["messages"][-1].content)
