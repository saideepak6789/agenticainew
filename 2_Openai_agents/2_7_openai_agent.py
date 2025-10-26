import os
import numpy as np
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool
import asyncio

# --- Load environment ---
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load PDF ---
PDF_PATH = "c://code//agenticai//2_openai_agents//new_india_assurance.pdf"
reader = PdfReader(PDF_PATH)
pdf_text = "".join([page.extract_text() or "" for page in reader.pages])

# --- Split PDF text into chunks ---
def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

pdf_chunks = split_text(pdf_text)

# --- Embeddings setup ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Embedding {len(pdf_chunks)} chunks from the PDF...")

chunk_embeddings = [
    {"text": chunk, "embedding": embedding_model.encode(chunk)}
    for chunk in pdf_chunks
]
print("✅ Embeddings ready!")

# --- Cosine similarity ---
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# --- Define Tool ---
@function_tool
async def get_pdf_answer(topic: str) -> str:
    """
    Answers questions by retrieving relevant information from the PDF using embeddings,
    and generating a concise response using OpenAI.
    """
    query_embedding = embedding_model.encode(topic)

    # Find the most relevant chunk
    best_chunk, best_score = None, -1
    for chunk in chunk_embeddings:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        if score > best_score:
            best_score = score
            best_chunk = chunk["text"]

    # Generate final answer using OpenAI
    if best_chunk:
        prompt = (
            f"You are a helpful assistant answering questions about insurance policies.\n\n"
            f"Document content:\n{best_chunk}\n\n"
            f"User question: {topic}\n\n"
            f"Provide a concise and accurate answer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    return "I'm sorry, I couldn't find information related to that query."


# --- Agent ---
rag_agent = Agent(
    name="Customer Support RAG Bot",
    instructions=(
        "You are a helpful customer support assistant. "
        "Answer questions using the PDF content through your RAG tool."
    ),
    tools=[get_pdf_answer]
)


# --- Async chat handler ---
async def chat_with_rag(message, chat_history):
    session = await Runner.run(rag_agent, message)
    chat_history = chat_history or []
    chat_history.append((message, session.final_output))
    return chat_history, chat_history


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Customer Support RAG Bot (PDF + Hugging Face + GPT-4o-mini)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about New India Assurance policies...")
    clear = gr.Button("Clear")

    async def respond(user_message, chat_history):
        return await chat_with_rag(user_message, chat_history)

    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
