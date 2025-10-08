# pip install mailersend
import os
import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv
from mailersend import emails
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from crewai import Agent, Task, Crew
from crewai.tools import tool

# ---------------------------------------------------
# Setup
# ---------------------------------------------------
load_dotenv(override=True)

ntfy_topic = os.getenv("NTFY_URGENT_TICKETS_TOPIC")
serp_api_key = os.getenv("SERPAPI_API_KEY")
mailersend_api_key = os.getenv("MAILERSEND_API_KEY")

# Load embeddings + VectorDB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local(
    "c://code//agentic_ai//4_crewai//customer_support_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

df = pd.read_pickle("c://code//agentic_ai//4_crewai//customer_tickets.pkl")

# ---------------------------------------------------
# Define tools (CrewAI-compatible)
# ---------------------------------------------------
@tool("VectorSearch")
def vector_search(query: str):
    """Look up support tickets in the vector database"""
    results = vectordb.similarity_search(query, k=1)
    if results:
        doc = results[0]
        return {
            "answer": doc.metadata.get("answer", "No answer"),
            "priority": doc.metadata.get("priority", "low").lower(),
        }
    return {"answer": None, "priority": None}

@tool("SendNTFY")
def send_ntfy(message: str):
    """Send urgent push notification for high priority tickets"""
    url = f"https://ntfy.sh/{ntfy_topic}"
    resp = requests.post(url, data=message.encode())
    return f"ntfy push sent (status {resp.status_code})"

@tool("SendEmail")
def send_test_email(query: str = None):
    """Send an email for medium priority tickets"""
    print("In email send logic")
    
    mailer = emails.NewEmail(mailersend_api_key)
    mail_body = {}

    mail_from = {
        "name": "Chatbot",
        "email": "sender@test-nrw7gymkvkog2k8e.mlsender.net",
    }

    recipients = [{
        "name": "Our Customer", 
        "email": "ekahate@gmail.com",
    }]
	
    reply_to = [{
        "name": "Chatbot", 
         "email": "receiver@test-nrw7gymkvkog2k8e.mlsender.net",
    }]

    mailer.set_mail_from(mail_from, mail_body)
    mailer.set_mail_to(recipients, mail_body)
    mailer.set_subject("Support Ticket", mail_body)
    body_text = f"Customer raised a medium priority issue:\n\n{query or 'No query text provided'}"
    mailer.set_html_content(body_text.replace("\n", "<br>"), mail_body)
    mailer.set_plaintext_content(body_text, mail_body)
    mailer.set_reply_to(reply_to, mail_body)

    resp = mailer.send(mail_body)
    
    print("Email sent")
    return f"Email sent (response: {resp})"

@tool("LowPriorityAck")
def low_priority_ack(_=None):
    """Send acknowledgement for low priority tickets"""
    return "We have noted your request/question and will attend to it within 48 hours."

@tool("SerpSearch")
def serp_fallback(query: str):
    """Fallback web search using SerpAPI"""
    if not serp_api_key:
        return "SERPAPI_API_KEY not set in .env"

    params = {"q": query, "api_key": serp_api_key, "engine": "google"}
    try:
        resp = requests.get("https://serpapi.com/search", params=params)
        data = resp.json()
        if "organic_results" in data and len(data["organic_results"]) > 0:
            top_result = data["organic_results"][0].get("snippet", "")
            return top_result
        return "No results found via SerpAPI"
    except Exception as e:
        return f"SerpAPI request failed: {str(e)}"

# ---------------------------------------------------
# LLM
# ---------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------
# CrewAI Agents with tools
# ---------------------------------------------------
router_agent = Agent(
    role="Ticket Router",
    goal="Classify ticket priority and call the right tool",
    backstory="Experienced support triage agent. Must ALWAYS run VectorSearch first. "
              "If priority=high → call SendNTFY. "
              "If medium → call SendEmail. "
              "If low → call LowPriorityAck. "
              "If no match → call SerpSearch.",
    llm=llm,
    tools=[vector_search, send_ntfy, send_test_email, low_priority_ack, serp_fallback],
    verbose=True,
)

review_agent = Agent(
    role="Support Supervisor",
    goal="Review router agent’s actions and produce final customer-facing response",
    backstory="Senior support manager ensuring quality and consistency of replies",
    llm=llm,
    tools=[],
    verbose=True,
)

# ---------------------------------------------------
# Main handler
# ---------------------------------------------------
def handle_ticket(user_query: str):
    if not user_query:
        return "Please enter a support request."

    routing_task = Task(
        description=f"Handle customer ticket: {user_query}",
        expected_output="Tool calls + raw draft response",
        agent=router_agent,
    )

    review_task = Task(
        description="Review router’s results and craft a polished customer support response.",
        expected_output="Final customer-friendly support message",
        agent=review_agent,
    )

    crew = Crew(
        agents=[router_agent, review_agent],
        tasks=[routing_task, review_task],
        verbose=True,
    )

    try:
        final_result = crew.kickoff()
        return str(final_result)
    except Exception as e:
        return f"Support flow failed: {str(e)}"

# ---------------------------------------------------
# Gradio UI
# ---------------------------------------------------
with gr.Blocks(title="Customer Support Crew (LLM + ReAct)") as demo:
    gr.Markdown("# Customer Support Crew (LLM + ReAct Loop Inside CrewAI)")
    gr.Markdown("VectorDB → Router Agent (tools) → Supervisor Agent (review)")

    with gr.Row():
        with gr.Column(scale=1):
            user_query = gr.Textbox(
                label="Your Query",
                placeholder="e.g., I cannot log in to my account"
            )
            run_btn = gr.Button("Submit Ticket", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(
                label="Support Response",
                lines=20,
                max_lines=30,
                show_copy_button=True
            )

    run_btn.click(
        fn=handle_ticket,
        inputs=user_query,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
