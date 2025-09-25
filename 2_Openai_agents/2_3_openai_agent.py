from dotenv import load_dotenv
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from agents import Agent, Runner, FunctionTool

load_dotenv(override=True)

# --- OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Embeddings ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # local embeddings
knowledge_base = {
    "shipping_time": "Our standard shipping time is 3-5 business days.",
    "return_policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment_methods": "We accept credit cards, debit cards, and PayPal.",
    "customer_support": "You can reach our support team 24/7 via email or chat."
}

embeddings_index = {}
for topic_key, answer in knowledge_base.items():
    embeddings_index[topic_key] = embedding_model.encode(answer)
print("Embeddings ready!")

# --- Cosine similarity ---
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# --- Tool callback with generation ---
async def faq_invoker(tool_context, params):
    args = json.loads(params) if isinstance(params, str) else params
    user_query = args.get("topic", "")
    query_embedding = embedding_model.encode(user_query)

    # Retrieve best FAQ
    best_topic = None
    best_score = -1
    for topic_key, embedding in embeddings_index.items():
        score = cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_score = score
            best_topic = topic_key

    if best_topic:
        # Generate human-like answer using OpenAI
        prompt = (
            f"Answer the user's question clearly and concisely based on this FAQ:\n\n"
            f"FAQ: {knowledge_base[best_topic]}\n"
            f"Question: {user_query}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    else:
        return "I'm sorry, I couldn't find information about that topic."

# --- Tool schema ---
faq_schema = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "Customer question"}
    },
    "required": ["topic"]
}

# --- FunctionTool ---
faq_tool = FunctionTool(
    name="get_faq_answer",
    description="Provides answers to FAQs using Hugging Face embeddings and OpenAI generation.",
    params_json_schema=faq_schema,
    on_invoke_tool=faq_invoker
)

# --- Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="Answer questions using the FAQ knowledge base and OpenAI for clear responses.",
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
        print("Customer Support Bot running. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = await chat_with_support(user_input)
            print("Bot:", response)

    asyncio.run(main())
