# Data taken from https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
import pandas as pd
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

def create_and_save_embeddings():
    """
    Read customer_tickets.csv, create embeddings from 'body',
    and save FAISS DB with answer + priority metadata
    """
    
    print("Reading customer_tickets.csv...")
    try:
        df = pd.read_csv('customer_tickets.csv')
        print(f"Loaded {len(df)} tickets from CSV")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        required_cols = ['body', 'answer', 'priority']
        for col in required_cols:
            if col not in df.columns:
                print("Available columns:", list(df.columns))
                raise ValueError(f"Missing required column: {col}")
        
        # Drop rows with missing bodies
        df = df.dropna(subset=['body'])
        df = df[df['body'].str.strip() != '']
        print(f"Processing {len(df)} valid tickets...")
        
    except FileNotFoundError:
        print("Error: customer_tickets.csv not found in current directory")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Step 2: Create documents (body as content, metadata = answer + priority)
    print("Creating documents...")
    documents = []
    for idx, row in df.iterrows():
        body = str(row['body']).strip()
        answer = str(row['answer']).strip() if pd.notna(row['answer']) else ""
        priority = str(row['priority']).strip() if pd.notna(row['priority']) else "unknown"
        
        doc = Document(
            page_content=body,
            metadata={
                "answer": answer,
                "priority": priority,
                "csv_index": idx
            }
        )
        documents.append(doc)
    
    # Step 3: Split documents (for long bodies)
    print("Splitting documents...")
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator=" "
    )
    docs = splitter.split_documents(documents)
    print(f"Created {len(docs)} document chunks")
    
    # Step 4: Initialize embeddings
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Step 5: Create FAISS vector DB
    print("Creating embeddings and vector database...")
    vectordb = FAISS.from_documents(docs, embeddings)
    print("Vector database created successfully!")
    
    # Step 6: Save everything
    print("Saving embeddings to files...")
    vectordb.save_local("customer_support_faiss")
    df.to_pickle("customer_tickets.pkl")
    
    embedding_info = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "total_documents": len(docs),
        "total_tickets": len(df)
    }
    with open("embedding_info.pkl", "wb") as f:
        pickle.dump(embedding_info, f)
    
    print("Embeddings saved successfully!")
    print(f"Files created:")
    print(f"   - customer_support_faiss/ (FAISS vector database)")
    print(f"   - customer_tickets.pkl (original data)")
    print(f"   - embedding_info.pkl (metadata)")

def load_embeddings_example():
    """
    Example: load embeddings, run query, fetch best answers + priority
    """
    print("\n" + "="*50)
    print("EXAMPLE: How to load saved embeddings")
    print("="*50)
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectordb = FAISS.load_local(
            "customer_support_faiss", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        df = pd.read_pickle("customer_tickets.pkl")
        with open("embedding_info.pkl", "rb") as f:
            info = pickle.load(f)
        
        print("Successfully loaded embeddings!")
        print(f"Loaded {info['total_tickets']} tickets with {info['total_documents']} document chunks")
        
        # Test search
        test_query = "I forgot my password, how do I reset it?"
        results = vectordb.similarity_search(test_query, k=3)
        
        print(f"\nTest query: {test_query}")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Body: {doc.page_content}")
            print(f"   → Suggested Answer: {doc.metadata.get('answer')}")
            print(f"   → Priority: {doc.metadata.get('priority')}")
            
    except FileNotFoundError:
        print("Embedding files not found. Run create_and_save_embeddings() first.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")

if __name__ == "__main__":
    create_and_save_embeddings()
    load_embeddings_example()
