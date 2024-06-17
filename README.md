# Breast-Cancer-Supportive-Care-Chatbot
This chatbot was developed for educational purposes only and is designed to provide users with information on breast cancer supportive care resources, clinical trials, financial assistance resources, emotional support resources, answers to treatment questions, radiology scan analysis and genomic testing education. The chatbot uses OpenAI's gpt-4-turbo model as its core language model for understanding and generating text. LangChain, a library that simplifies LLM workflows, was used for for managing prompts, chains (sequences of LLM calls), and interacting with the application's vector store. This application uses Pinecone Vector Database as the vector store where PDF and CSV documents with information on breast cancer nonprofits, clinical trials and other relevant information were uploaded for gpt-4-turbo to access via retrieval augmented generation (RAG). The chatbot also allows users to upload radiology and mammography scans in DICOM file format for educational analysis using gemini-1.5-flash for image analysis and response to the user. Users can also upload genomic and biomarker testing reports for educational analysis using claude-3-opus from Anthropic. Nemo Gaurdrails from Nvidia was used for dialog management between the LLM and the user (see prompts.yml file). 

Features:

- Clinical trials search and treatment Q&A
- Financial assistance resources
- Supportive care resources
- Emotional support resources
- Radiology scan analysis
- Genomic testing report upload and analysis

Installation:

- Clone the repository: git clone https://github.com/jsteward2930/breast-cancer-supportive-care-chatbot
- Navigate to the project directory: cd breast-cancer-supportive-care-chatbot
- Install the required dependencies: pip install -r requirements.txt


Usage:

Configure your environment variables:

Create a .env file in the root directory of your project and add the following variables:

PINECONE_API_KEY=your_pinecone_api_key 
OPENAI_API_KEY=your_openai_api_key 
CLAUDE_API_KEY=your_claude_api_key 
NVIDIA_API_KEY=your_nvidia_api_key 
GEMINI_API_KEY=your_gemini_api_key


Before running the chatbotv31.py file, be sure to add the local path of the prompts.yml file to line 107 to properly initialize Nemo Gaurdrails. Additionally, add the local path of the ribbon-breast-cancer.png file to line 292 for proper display in the Gradio user interface. 

Line 106: # Initialize NeMo Guardrails
Line 107: config_path = "/Users/josephsteward/Nvidia/a-breast-rag-chatbot/files-breast-cancer-chatbot/prompts.yml"  # Path to your config file
config

Line 291: # Add the pink ribbon image with the correct path
Line 292: gr.Image(value="/path/ribbon-breast-cancer.png", elem_id="pink-ribbon-image", label="Support Breast Cancer Awareness", width=250, height=250)

Run the application: python src/chatbotv31.py


Access the Gradio Interface:
- Open your web browser and navigate to the local server address provided in the terminal (e.g., http://127.0.0.1:7860).


Pinecone Database:

The chatbot uses Pinecone to store and query vectors. Here are the details of the Pinecone database configuration:

- Index Name: breast-cancer-rag
- metric: cosine
- Dimensions: 1536
- Cloud Provider: AWS
- Region: us-east-1
- Type: Serverless
- Vector Count: 1,313 (as of the last update)


Pinecone Setup

Initialize Pinecone:
- Ensure you have the Pinecone client installed: pip install pinecone-client
- Initialize Pinecone in your script: import pinecone
- pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))


Create and Connect to Index:

Check if the index exists and create it if it doesn't: 
- if "breast-cancer-rag" not in pinecone.list_indexes(): pinecone.create_index( name="breast-cancer-rag", dimension=1536, metric="cosine", pod_type="s1.x1" ) index = pinecone.Index("breast-cancer-rag")


Add and Query Vectors:
- Add vectors to the index: index.upsert(items=[("id", vector)])
- Query vectors from the index: results = index.query(queries=[query_vector], top_k=10)


License: This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License - see the LICENSE file for details.


Contact: For any inquiries, please contact jsteward2930@example.com


Additional Resources: 

Gradio: https://www.gradio.app/docs

LangChain: https://python.langchain.com/v0.2/docs/introduction/

OpenAI: https://platform.openai.com/docs/introduction

Pinecone: https://docs.pinecone.io/guides/get-started/quickstart

NeMo Guardrails: https://docs.nvidia.com/nemo-guardrails/index.html

Google Gemini: https://ai.google.dev/gemini-api/docs

Anthropic: https://docs.anthropic.com/en/api/getting-started
