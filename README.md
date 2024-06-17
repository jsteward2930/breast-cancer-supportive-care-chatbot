# breast-cancer-supportive-care-chatbot
This chatbot provides information about breast cancer supportive care resources, financial assistance resources, emotional support resources, and answers to treatment questions.

Features:

Clinical trials search

Financial assistance resources

Supportive care resources

Emotional support resources

Treatment Q&A

Radiology scan analysis

Installation:

Clone the repository: git clone https://github.com/jsteward2930/breast-cancer-supportive-care-chatbot

Navigate to the project directory: cd breast-cancer-supportive-care-chatbot

Install the required dependencies: pip install -r requirements.txt

Usage:

Configure your environment variables:

Create a .env file in the root directory of your project and add the following variables:

PINECONE_API_KEY=your_pinecone_api_key 

OPENAI_API_KEY=your_openai_api_key 

CLAUDE_API_KEY=your_claude_api_key 

NVIDIA_API_KEY=your_nvidia_api_key 

GEMINI_API_KEY=your_gemini_api_key

Run the application: python src/chatbotv31.py

Access the Gradio Interface:

Open your web browser and navigate to the local server address provided in the terminal (e.g., http://127.0.0.1:7860).

Pinecone Database:

The chatbot uses Pinecone to store and query vectors. Here are the details of the Pinecone database configuration:

Index Name: breast-cancer-rag

Metric: cosine

Dimensions: 1536

Cloud Provider: AWS

Region: us-east-1

Type: Serverless

Vector Count: 1,313 (as of the last update)

Pinecone Setup

Initialize Pinecone:

Ensure you have the Pinecone client installed: pip install pinecone-client

Initialize Pinecone in your script: import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

Create and Connect to Index:

Check if the index exists and create it if it doesn't: 

if "breast-cancer-rag" not in pinecone.list_indexes(): pinecone.create_index( name="breast-cancer-rag", dimension=1536, metric="cosine", pod_type="s1.x1" ) index = pinecone.Index("breast-cancer-rag")

Add and Query Vectors:

Add vectors to the index: index.upsert(items=[("id", vector)])

Query vectors from the index: results = index.query(queries=[query_vector], top_k=10)

License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License - see the LICENSE file for details.

Contact

For any inquiries, please contact jsteward2930@example.com.

Additional Resources

Official Documentation

API Reference

Gradio

LangChain

OpenAI

Pinecone

NeMo Guardrails
