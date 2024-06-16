import os
import gradio as gr
import requests
import numpy as np
from openai import OpenAI
from PIL import Image, ImageOps
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from nemoguardrails import LLMRails, RailsConfig
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import base64
import io

# Load environment variables
load_dotenv()

# Retrieve environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Debugging: Print environment variables to ensure they are loaded correctly
print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
print(f"CLAUDE_API_KEY: {CLAUDE_API_KEY}")  # Ensure this is printed correctly
print(f"NVIDIA_API_KEY: {NVIDIA_API_KEY}")
print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")  # Ensure this is printed correctly

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

try:
    # Check if the index exists, and create it if it doesn't
    if "breast-cancer-rag" not in pc.list_indexes().names():
        pc.create_index(
            name="breast-cancer-rag",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
except Exception as e:
    print(f"Error during Pinecone initialization or index creation: {e}")
    exit()

# Connect to Pinecone index
index = pc.Index("breast-cancer-rag")

# Setup LangChain components
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
vector_store = Pinecone(index, embeddings, text_key="text")
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=OPENAI_API_KEY)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant that provides information to breast cancer patients about their genomic test results. Use the following summaries to answer the user's question:"
    ),
    HumanMessagePromptTemplate.from_template("{summaries}\n\nUSER QUESTION: {question}")
])

# Placeholder prompt for the retriever
retriever_prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{input}")
])

# Setup Retrieval Chain
retriever = create_history_aware_retriever(
    retriever=vector_store.as_retriever(),
    prompt=retriever_prompt_template,
    llm=llm
)

qa_chain = load_qa_with_sources_chain(
    llm,
    chain_type="stuff",
    prompt=prompt_template
)

# Initialize NeMo Guardrails
config_path = "/Users/josephsteward/Nvidia/a-breast-rag-chatbot/files-breast-cancer-chatbot/prompts.yml"  # Path to your config file
config = RailsConfig.from_path(config_path)
app = LLMRails(config=config, llm=llm)

tokenizer = AutoTokenizer.from_pretrained('UFNLP/gatortronS')
config = AutoConfig.from_pretrained('UFNLP/gatortronS')
model = AutoModelForCausalLM.from_pretrained("UFNLP/gatortronS") 

def call_claude_api(full_text):
    max_tokens = 102398  # Maximum tokens allowed by Claude API
    truncated_text = full_text[:max_tokens - 500]  # Reserve some tokens for the prompt and response

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-v1",
        "prompt": f"\n\nHuman: Summarize the following text for a breast cancer patient:\n\n{truncated_text}\n\nAssistant:",
        "max_tokens_to_sample": 1024
    }
    response = requests.post("https://api.anthropic.com/v1/complete", json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['completion']
    else:
        print(f"Error calling Claude API: {response.content}")
        return None  # Return None if the API call fails

def process_uploaded_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=300, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(documents)

    # Concatenate all chunks into one text
    full_text = "\n".join(chunk.page_content for chunk in chunked_documents)

    # Attempt to call Claude API with retry logic
    summary = call_claude_api(full_text)

    # If the above call is successful, proceed to store the summary in Pinecone
    if summary is not None:
        # Store the summary in Pinecone, ensuring metadata size is within limits
        summary_metadata = {"summary": summary[:40960]}  # Truncate summary if necessary
        vector_store.add_texts([summary], metadatas=[summary_metadata])

        return summary
    return None

# Function to initialize the chatbot with a greeting message
def initialize_chatbot():
    greeting_message = (
        "Welcome! I am a chatbot for Breast Cancer Supportive Care and Patient Education. "
        "This chatbot provides information about breast cancer supportive care resources, "
        "financial assistance resources, emotional support resources, and answers to treatment questions. "
        "Click on one of the tabs to the left to get started and I'm happy to assist you today?\n\n"
        "It is important to note that I am not a medical professional and cannot provide medical advice. "
        "The information I provide is intended for educational purposes only."
    )
    return [(greeting_message, "")]

# Define Gradio interface for clinical trials search
def clinical_trials_search(history, message, pdf_file=None):
    if not history:
        history = initialize_chatbot()
    if pdf_file:
        summary = process_uploaded_pdf(pdf_file)
        history.append(("Summary of the uploaded PDF:", summary))
        return history

    context = vector_store.similarity_search(message, k=4)
    for doc in context:
        doc.metadata['source'] = 'clinical_trials'
    answer = qa_chain({"question": message, "input_documents": context})
    history.append((message, answer["output_text"]))
    return history

# Define Gradio interface for financial assistance resources
def financial_assistance_resources(history, message):
    if not history:
        history = initialize_chatbot()
    context = vector_store.similarity_search(message, k=4)
    for doc in context:
        doc.metadata['source'] = 'financial_assistance_resources'
    answer = qa_chain({"question": message, "input_documents": context})
    history.append((message, answer["output_text"]))
    return history

# Define Gradio interface for supportive care resources
def supportive_care_resources(history, message):
    if not history:
        history = initialize_chatbot()
    context = vector_store.similarity_search(message, k=4)
    for doc in context:
        doc.metadata['source'] = 'supportive_care_resources'
    answer = qa_chain({"question": message, "input_documents": context})
    history.append((message, answer["output_text"]))
    return history

# Define Gradio interface for emotional support resources
def emotional_support_resources(history, message):
    if not history:
        history = initialize_chatbot()
    context = vector_store.similarity_search(message, k=4)
    for doc in context:
        doc.metadata['source'] = 'emotional_support_resources'
    answer = qa_chain({"question": message, "input_documents": context})
    history.append((message, answer["output_text"]))
    return history

# Define Gradio interface for treatment questions and answers
def treatment_qa(history, message):
    if not history:
        history = initialize_chatbot()
    context = vector_store.similarity_search(message, k=4)
    for doc in context:
        doc.metadata['source'] = 'treatment_questions'
    answer = qa_chain({"question": message, "input_documents": context})
    history.append((message, answer["output_text"]))
    return history

# Function to preprocess DICOM files and other image types
def preprocess_image(file):
    try:
        dicom = dcmread(file.name)
        data = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        image = Image.fromarray((data * 255).astype(np.uint8))
        image = ImageOps.grayscale(image)  # Ensure the image is in grayscale
    except Exception as e:  # Catch exceptions that may occur during DICOM loading
        print(f"Error loading DICOM file: {e}")
        return None  # Return None to indicate an error

    # Convert to RGB format (Gemini might work better with RGB)
    image = image.convert("RGB")  
    return image

def analyze_radiology_scan(file):
    image = preprocess_image(file)  

    if image is None:
        return "Error: Unable to process the uploaded image. Please ensure it's a valid DICOM file."

    # Gemini Image Description
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')  

    description_prompt = """
    Please provide a detailed scientific summary of the image. Describe the features and characteristics observed, including shapes, sizes, textures, or any notable findings. 
    Explain how these observations relate to the scientific analysis of breast cancer, discussing any implications or relevant scientific concepts. Avoid giving medical advice, but explain the scientific aspects of breast cancer and how the observed features in the scan might relate in that context. 
    Ensure the summary is in plain language, including simple explanations where helpful.
    """
    response = gemini_model.generate_content([description_prompt, image])

    # Extract image description, handling multiple parts and edge cases
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        image_description = ''.join(part.text for part in response.candidates[0].content.parts)
    else:
        image_description = "No image description generated."

    return f"Image description:\n{image_description}"

def radiology_scan_analysis(history, file):
    if not history:
        history = initialize_chatbot()
    image = preprocess_image(file)  # Process the image and store it for display
    report = analyze_radiology_scan(file)
    history.append(("", report))  # Append only the report to the chatbot
    return history, image  # Return both the updated history and the image 

# Gradio Interface
with gr.Blocks(css=".gradio-container {max-width: 900px; margin: auto;} #sidebar {min-width: 200px;} .tab-content {flex: 1; padding: 20px;} .tab-button {background-color: #f1f1f1; border: none; text-align: left; padding: 10px; cursor: pointer;} .tab-button:hover {background-color: #ddd;} .tab-button.active {background-color: #ccc;}") as iface:
    tab_state = gr.State(0)
    tabs = []

    def update_tab(index):
        return [gr.update(visible=(i == index)) for i in range(6)]

    with gr.Row():
        with gr.Column(elem_id="sidebar"):
            gr.Markdown("# Breast Cancer Supportive Care Chatbot\n\nThis chatbot provides information about breast cancer supportive care resources, financial assistance resources, emotional support resources, and answers to treatment questions.")

            # Add the pink ribbon image with the correct path
            gr.Image(value="/Users/josephsteward/Nvidia/a-breast-rag-chatbot/files-breast-cancer-chatbot/ribbon-breast-cancer.png", elem_id="pink-ribbon-image", label="Support Breast Cancer Awareness", width=250, height=250)

            buttons = []
            for i, tab_name in enumerate([
                "Find Supportive Care Resources and Nonprofits",
                "Biomarker and Genomic Testing Resources and Education",
                "Financial Assistance Resources and Nonprofits",
                "Emotional Support Resources and Nonprofits",
                "Treatment Education and Clinical Trials Search",
                "Radiology Scan Analysis and Education"
            ]):
                button = gr.Button(tab_name, elem_id=f"button-{i}")
                button.click(fn=lambda i=i: update_tab(i), inputs=[], outputs=tabs)
                buttons.append(button)
    
        with gr.Column(elem_id="content"):
            chatbot_interface = gr.Chatbot(value=initialize_chatbot())  # Initialize with greeting message
            
            with gr.Tab(label="Find Nonprofits with Supportive Care Resources", visible=False) as tab:
                gr.Markdown("## Find Nonprofits with Supportive Care Resources\n\nAsk any questions about breast cancer nonprofits that have free supportive care resources in your local area. Ask questions like 'Can you help me find breast cancer nonprofits in New York?'")
                message_input_support = gr.Textbox(label="Message")
                submit_button_support = gr.Button("Submit")
                submit_button_support.click(supportive_care_resources, inputs=[chatbot_interface, message_input_support], outputs=chatbot_interface)
                tabs.append(tab)
            
            with gr.Tab(label="Biomarker and Genomic Testing Results Upload", visible=False) as tab1:
                gr.Markdown("## Biomarker and Genomic Testing Results Upload\n\nUpload your genomic testing results to get a summary and ask questions about the results.")
                pdf_input = gr.File(label="Upload PDF", type="filepath")
                message_input_trials = gr.Textbox(label="Message")
                submit_pdf_button = gr.Button("Submit")
                submit_pdf_button.click(clinical_trials_search, inputs=[chatbot_interface, message_input_trials, pdf_input], outputs=chatbot_interface)
                tabs.append(tab1)
                
            with gr.Tab(label="Financial Assistance Resources and Nonprofits", visible=False) as tab2:
                gr.Markdown("## Financial Assistance Resources and Nonprofits\n\nAsk any questions about financial assistance resources and nonprofits that can help with breast cancer treatment costs.")
                message_input_financial = gr.Textbox(label="Message")
                submit_button_financial = gr.Button("Submit")
                submit_button_financial.click(financial_assistance_resources, inputs=[chatbot_interface, message_input_financial], outputs=chatbot_interface)
                tabs.append(tab2)
                
            with gr.Tab(label="Emotional Support Resources and Nonprofits", visible=False) as tab3:
                gr.Markdown("## Emotional Support Resources and Nonprofits\n\nAsk any questions about emotional support resources and nonprofits that provide emotional support for breast cancer patients.")
                message_input_emotional = gr.Textbox(label="Message")
                submit_button_emotional = gr.Button("Submit")
                submit_button_emotional.click(emotional_support_resources, inputs=[chatbot_interface, message_input_emotional], outputs=chatbot_interface)
                tabs.append(tab3)
                
            with gr.Tab(label="Treatment Questions and Answers", visible=False) as tab4:
                gr.Markdown("## Treatment Questions and Answers\n\nAsk any questions about breast cancer treatments and get answers based on the latest research and guidelines.")
                message_input_treatment = gr.Textbox(label="Message")
                submit_button_treatment = gr.Button("Submit")
                submit_button_treatment.click(treatment_qa, inputs=[chatbot_interface, message_input_treatment], outputs=chatbot_interface)
                tabs.append(tab4)
                
            with gr.Tab(label="Radiology Scan Analysis", visible=False) as tab5:
                gr.Markdown("## Radiology Scan Analysis")
                with gr.Row():
                    file_input = gr.File(label="Upload Radiology Scan", file_types=["image", ".dcm"])
                    image_display = gr.Image(label="Uploaded Image")
                submit_file_button = gr.Button("Submit")
                submit_file_button.click(radiology_scan_analysis, inputs=[chatbot_interface, file_input], outputs=[chatbot_interface, image_display])
                tabs.append(tab5)

iface.launch(share=True)
