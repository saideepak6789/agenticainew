from dotenv import load_dotenv
import os
import json
import sqlite3
import numpy as np
import gradio as gr

from sentence_transformers import SentenceTransformer
from agents import Agent, Runner, FunctionTool
from openai import OpenAI

load_dotenv(override=True)

# --- OpenAI client ---
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

    # Retrieve most relevant FAQ
    best_topic = None
    best_score = -1
    for topic_key, embedding in embeddings_index.items():
        score = cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_score = score
            best_topic = topic_key

    if best_topic:
        # Use OpenAI gpt-4o-mini to generate answer from retrieved FAQ
        prompt = (
            f"Provide a clear and concise answer based on the following FAQ:\n\n"
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

# --- Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="Answer questions using the knowledge base and OpenAI for concise answers.",
    tools=[faq_tool]
)

# --- Chat function for Gradio ---
async def chat_with_support(message, chat_history):
    session = await Runner.run(faq_agent, message)
    chat_history = chat_history or []
    chat_history.append((message, session.final_output))
    return chat_history, chat_history

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Customer Support Bot")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about our services or policies...")
    clear = gr.Button("Clear")

    async def respond(user_message, chat_history):
        return await chat_with_support(user_message, chat_history)

    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
