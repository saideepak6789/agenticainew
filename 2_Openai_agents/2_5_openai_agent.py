from dotenv import load_dotenv
import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from agents import Agent, Runner, function_tool
import asyncio

# --- Setup ---
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Database connection ---
DB_PATH = r"c:\code\agenticai\faqs.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def load_faqs():
    cursor.execute("SELECT topic, answer FROM faqs")
    return dict(cursor.fetchall())

knowledge_base = load_faqs()

# --- Embeddings ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Computing embeddings for FAQ knowledge base...")
embeddings_index = {
    topic: embedding_model.encode(answer)
    for topic, answer in knowledge_base.items()
}
print(f"Loaded {len(embeddings_index)} FAQs. Embeddings ready!")


# --- Cosine similarity ---
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# --- FAQ Tool ---
@function_tool
async def get_faq_answer(topic: str) -> str:
    """
    Find the most relevant FAQ answer using embeddings stored in SQLite,
    then generate a clear and natural response using OpenAI.
    """
    query_embedding = embedding_model.encode(topic)

    # Find the best matching topic
    best_topic, best_score = None, -1
    for t, emb in embeddings_index.items():
        score = cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_topic, best_score = t, score

    # Generate answer using OpenAI if a match is found
    if best_topic:
        prompt = (
            f"User asked: {topic}\n\n"
            f"Relevant FAQ: {knowledge_base[best_topic]}\n\n"
            "Write a concise, natural-sounding customer support answer."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    return "I'm sorry, I couldn't find specific information about that topic."


# --- Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="You are a friendly support assistant. Use your FAQ tool to help users.",
    tools=[get_faq_answer],
)


# --- Chat function ---
async def chat_with_support(message):
    session = await Runner.run(faq_agent, message)
    return session.final_output


# --- Main loop ---
async def main():
    print("Customer Support Bot running. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        response = await chat_with_support(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    asyncio.run(main())
