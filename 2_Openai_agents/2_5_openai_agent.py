from dotenv import load_dotenv
import os
import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from agents import Agent, Runner, FunctionTool

load_dotenv(override=True)

# --- OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Database ---
DB_PATH = "c:\\code\\agenticai\\1_openai_api\\faqs.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def load_faqs():
    cursor.execute("SELECT topic, answer FROM faqs")
    return dict(cursor.fetchall())

knowledge_base = load_faqs()

# --- Hugging Face embeddings ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings_index = {}

def compute_embedding(text):
    return embedding_model.encode(text)

print("Computing embeddings for knowledge base...")
for topic_key, answer in knowledge_base.items():
    embeddings_index[topic_key] = compute_embedding(answer)
print("Embeddings ready!")

# --- Cosine similarity ---
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# --- Tool callback function with OpenAI generation ---
async def faq_invoker(tool_context, params):
    args = json.loads(params) if isinstance(params, str) else params
    user_query = args.get("topic", "")

    query_embedding = compute_embedding(user_query)

    # Find the most relevant FAQ
    best_topic = None
    best_score = -1
    for topic_key, embedding in embeddings_index.items():
        score = cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_score = score
            best_topic = topic_key

    if best_topic:
        # --- Generate answer using OpenAI gpt-4o-mini ---
        prompt = (
            f"Provide a concise and clear answer based on the following FAQ:\n\n"
            f"FAQ: {knowledge_base[best_topic]}\n\n"
            f"Question: {user_query}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    else:
        return "I'm sorry, I couldn't find specific information about that topic."

# --- Tool schema ---
faq_schema = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "The topic or question asked by the customer."}
    },
    "required": ["topic"]
}

# --- Register FunctionTool ---
faq_tool = FunctionTool(
    name="get_faq_answer",
    description="Provides answers to FAQs using semantic search and OpenAI generation.",
    params_json_schema=faq_schema,
    on_invoke_tool=faq_invoker
)

# --- Define Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="Answer questions using the knowledge base and OpenAI for clear responses.",
    tools=[faq_tool]
)

# --- Chat function ---
async def chat_with_support(message):
    session = await Runner.run(faq_agent, message)
    return session.final_output

# --- Main loop ---
if __name__ == "__main__":
    import asyncio

    async def main():
        print("Customer Support Bot is running. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting.")
                break
            response = await chat_with_support(user_input)
            print("Bot:", response)

    asyncio.run(main())
