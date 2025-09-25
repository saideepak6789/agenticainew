import os
import json
import numpy as np
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

from agents import Agent, Runner, FunctionTool

# --- Load environment ---
load_dotenv(override=True)
client = OpenAI()

# --- Load PDF ---
PDF_PATH = "c://code//agenticai//2_openai_agents//new_india_assurance.pdf"
reader = PdfReader(PDF_PATH)
pdf_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text

# --- Split PDF text into chunks ---
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

pdf_chunks = split_text(pdf_text)

# --- Initialize Hugging Face embedding model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Compute embeddings for PDF chunks ---
chunk_embeddings = []

def compute_embedding(text):
    return embedding_model.encode(text)

print("Computing embeddings for PDF chunks...")
for chunk in pdf_chunks:
    chunk_embeddings.append({
        "text": chunk,
        "embedding": compute_embedding(chunk)
    })
print("Embeddings ready!")

# --- Cosine similarity ---
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- Tool callback function (RAG with OpenAI generation) ---
async def rag_invoker(tool_context, params):
    args = json.loads(params) if isinstance(params, str) else params
    user_query = args.get("topic", "")

    # Embed the query locally using Hugging Face
    query_embedding = compute_embedding(user_query)

    # Retrieve most relevant chunk
    best_chunk = None
    best_score = -1
    for chunk in chunk_embeddings:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        if score > best_score:
            best_score = score
            best_chunk = chunk["text"]

    if best_chunk:
        # --- Use OpenAI gpt-4o-mini to generate answer ---
        prompt = (
            f"Answer the user's question based on the following document content. "
            f"Be concise and clear.\n\nDocument: {best_chunk}\n\nQuestion: {user_query}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    else:
        return "I'm sorry, I couldn't find information related to that query."

# --- Tool schema ---
rag_schema = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "The user's question or topic."}
    },
    "required": ["topic"]
}

# --- Register FunctionTool ---
rag_tool = FunctionTool(
    name="get_pdf_answer",
    description="Answers questions by retrieving relevant information from the PDF and generating a concise answer using OpenAI.",
    params_json_schema=rag_schema,
    on_invoke_tool=rag_invoker
)

# --- Define Agent ---
rag_agent = Agent(
    name="Customer Support RAG Bot",
    instructions="You are a helpful customer support assistant. Answer questions using the PDF content or the provided tool.",
    tools=[rag_tool]
)

# --- Chat function for Gradio ---
async def chat_with_rag(message, chat_history):
    session = await Runner.run(rag_agent, message)
    chat_history = chat_history or []
    chat_history.append((message, session.final_output))
    return chat_history, chat_history

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Customer Support Bot (Hugging Face embeddings + OpenAI gpt-4o-mini)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about New India Assurance policies...")
    clear = gr.Button("Clear")

    async def respond(user_message, chat_history):
        return await chat_with_rag(user_message, chat_history)

    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
