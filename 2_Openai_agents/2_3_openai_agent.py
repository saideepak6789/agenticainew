from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from agents import Agent, Runner, function_tool
import asyncio

# --- Setup ---
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Embedding model & knowledge base ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

knowledge_base = {
    "shipping_time": "Our standard shipping time is 3-5 business days.",
    "return_policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment_methods": "We accept credit cards, debit cards, and PayPal.",
    "customer_support": "You can reach our support team 24/7 via email or chat."
}

# --- Precompute embeddings ---
embeddings_index = {
    topic: embedding_model.encode(answer)
    for topic, answer in knowledge_base.items()
}
print("Embeddings ready!")


# --- Cosine similarity helper ---
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# --- FAQ Tool ---
@function_tool
async def get_faq_answer(topic: str) -> str:
    """
    Finds the most relevant FAQ answer using sentence embeddings,
    then refines the response using OpenAI generation.
    """
    # Encode user query
    query_embedding = embedding_model.encode(topic)

    # Find the best matching topic
    best_topic, best_score = None, -1
    for t, emb in embeddings_index.items():
        score = cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_topic, best_score = t, score

    # If a match is found, generate an answer using OpenAI
    if best_topic:
        prompt = (
            f"User asked: {topic}\n\n"
            f"FAQ: {knowledge_base[best_topic]}\n\n"
            "Write a clear, natural-sounding customer support response."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    return "I'm sorry, I couldn't find information about that topic."


# --- Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="You are a helpful assistant who answers customer FAQs using your tool.",
    tools=[get_faq_answer],
)


# --- Chat handler ---
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
