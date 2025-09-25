from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=True)
client = OpenAI()

# --- Step 1: Create a vector store ---
vector_store = client.vector_stores.create(name="faqs_store")
print(f"Vector store created: {vector_store.id}")

# --- Step 2: Upload your local faq.txt file ---
faq_file = client.files.create(
    file=open("c:\\code\\agenticai\\2_openai_agents\\faq.txt", "rb"),
    purpose="assistants"
)

# Attach the uploaded file to the vector store
client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=faq_file.id
)
print("faq.txt uploaded into vector store.")

# --- Step 3: Ask a question ---
response = client.responses.create(
    model="gpt-4o-mini",
    input="Tell me about the warranty policy."
)

print(response.output_text)


print("\nBot Answer:", response.output_text)
