from dotenv import load_dotenv
import os
import sqlite3
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from agents import Agent, Runner, function_tool
import asyncio

# --- Setup ---
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Database ---
DB_PATH = "faqs.db"
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
    Retrieves the most relevant FAQ answer using embeddings,
    and generates a concise reply using OpenAI.
    """
    query_embedding = embedding_model.encode(topic)

    # Find the most relevant FAQ
    best_topic, best_score = None, -1
    for t, emb in embeddings_index.items():
        score = cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_topic, best_score = t, score

    # Generate response with OpenAI
    if best_topic:
        prompt = (
            f"User asked: {topic}\n\n"
            f"Relevant FAQ: {knowledge_base[best_topic]}\n\n"
            f"Write a concise, natural customer support response."
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
    instructions="You are a friendly support assistant. Use your FAQ tool to answer user questions.",
    tools=[get_faq_answer],
)


# --- Async chat handler for Gradio ---
async def chat_with_support(message, chat_history):
    session = await Runner.run(faq_agent, message)
    chat_history = chat_history or []
    chat_history.append((message, session.final_output))
    return chat_history, chat_history


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 💬 Customer Support Bot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about our products or policies...")
    clear = gr.Button("Clear")

    async def respond(user_message, chat_history):
        return await chat_with_support(user_message, chat_history)

    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
