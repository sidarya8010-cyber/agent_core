import csv
import json
import os
import re
import urllib.request
from typing import List

from dotenv import load_dotenv

from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
load_dotenv()


def load_faq_csv(path: str) -> List[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"].strip()
            answer = row["answer"].strip()
            docs.append({"question": question, "answer": answer})
    return docs


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_match(query: str, entry: dict) -> tuple[float, str]:
    q_tokens = set(normalize_text(query).split())
    question = normalize_text(entry["question"])
    answer = normalize_text(entry["answer"])
    question_tokens = set(question.split())
    answer_tokens = set(answer.split())

    overlap = len(q_tokens & question_tokens) + 0.5 * len(q_tokens & answer_tokens)
    if not q_tokens:
        return 0.0, entry["answer"]
    return overlap, f"Q: {entry['question']}\nA: {entry['answer']}"


def search_faq(query: str, num_results: int = 3) -> str:
    docs = load_faq_csv("./lauki_qna.csv")
    scored = sorted(
        ((score_match(query, doc)[0], score_match(query, doc)[1]) for doc in docs),
        reverse=True,
    )
    top_results = [text for _, text in scored[:num_results] if text]
    if not top_results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join(top_results)
    return f"Found {len(top_results)} relevant FAQ entries:\n\n{context}"


def search_detailed_faq(query: str, num_results: int = 5) -> str:
    return search_faq(query, num_results=num_results)


def reformulate_query(original_query: str, focus_aspect: str) -> str:
    reformulated = f"{focus_aspect} related to {original_query}"
    return search_faq(reformulated, num_results=3)


def build_prompt(query: str, context: str) -> str:
    return (
        "You are a helpful FAQ assistant. Use the retrieved context to answer the user.\n"
        f"User question: {query}\n\nContext:\n{context}"
    )


def call_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is not configured."

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
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

    with urllib.request.urlopen(req, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


@app.entrypoint
def agent_invocation(payload, context):
    """Handler for agent invocation in AgentCore runtime"""
    query = payload.get("prompt", "No prompt found in input")
    context_text = search_faq(query)
    prompt = build_prompt(query, context_text)
    answer = call_groq(prompt)
    return {"result": answer}


if __name__ == "__main__":
    app.run()
