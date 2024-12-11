import logging
import time
import os
import locale
import re
import torch
import bs4
import json
import numpy as np
import time
import pandas as pd
import openai
from openai import OpenAI
import uuid
import fitz
import requests
from pprint import pprint
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import tempfile
from collections import defaultdict
import subprocess
import pkg_resources
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate  
from langchain.chains import LLMChain  
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    OnlinePDFLoader,
    PyMuPDFLoader,
    PubMedLoader,
)
from langchain_community.vectorstores import FAISS, Chroma, Qdrant
from langchain_community.utils.math import cosine_similarity
from langchain.schema import Document
from qdrant_client import QdrantClient
import boto3
import sqlite3

import warnings
import traceback
import sys
import pdfplumber
warnings.filterwarnings("ignore")

DB_FILE = "header_cache.db"

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("load_pipeline_test")
logging.basicConfig(level=logging.INFO)

GROUND_TRUTH_FILE = "./train/bilt/bilt.csv"

ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
ground_truth_df = ground_truth_df.rename(columns={
    'Question': 'question', 
    'Answer': 'ground_truth'
})
print("Renamed columns:", ground_truth_df.columns.tolist())

def get_secret(secret_name):
    region_name = "us-east-1"
    client = boto3.client("secretsmanager", region_name=region_name)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except Exception as e:
        print(f"Error retrieving secret {secret_name}: {e}")
        return None
    
def init_cache_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS header_cache (
            company_name TEXT PRIMARY KEY,
            headers TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_cache_to_db(company_name, headers):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO header_cache (company_name, headers)
        VALUES (?, ?)
    """, (company_name, json.dumps(headers)))
    conn.commit()
    conn.close()
    
def load_cache_from_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT company_name, headers FROM header_cache")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: json.loads(row[1]) for row in rows}

def reset_cache_and_db() -> Tuple[bool, str]:
    """
    Reset the header cache and delete the SQLite database file.
    
    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: Success status (True if reset successful, False otherwise)
            - str: Status message describing the outcome
    """
    try:
        # Reset the global header cache dictionary
        global HEADER_CACHE
        HEADER_CACHE = {}
        
        # Delete the SQLite database file if it exists
        if os.path.exists(DB_FILE):
            try:
                os.remove(DB_FILE)
                print(f"Successfully deleted existing database file: {DB_FILE}")
            except PermissionError:
                return False, f"Permission denied when trying to delete {DB_FILE}"
            except Exception as e:
                return False, f"Error deleting database file: {str(e)}"
        
        # Reinitialize the database with fresh tables
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS header_cache (
                    company_name TEXT PRIMARY KEY,
                    headers TEXT
                )
            """)
            conn.commit()
            conn.close()
            print("Successfully reinitialized database with fresh tables")
            
            return True, "Successfully reset cache and database"
            
        except Exception as e:
            return False, f"Error reinitializing database: {str(e)}"
            
    except Exception as e:
        return False, f"Unexpected error during reset: {str(e)}"


def verify_reset() -> Tuple[bool, str]:
    """
    Verify that the cache and database were properly reset.
    
    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: Verification status (True if verified, False otherwise)
            - str: Status message describing the verification results
    """
    try:
        # Verify cache is empty
        if len(HEADER_CACHE) > 0:
            return False, "Cache is not empty after reset"
        
        # Verify database exists and is empty
        if not os.path.exists(DB_FILE):
            return False, "Database file does not exist after reset"
            
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM header_cache")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            return False, f"Database contains {count} records after reset"
            
        return True, "Cache and database verified empty"
        
    except Exception as e:
        return False, f"Error during verification: {str(e)}"


def model_fn(model_dir):
    """
    Initialize and return the RAG pipeline components: vector store, LLM, and embeddings.
    Load required components or resources from the provided model directory.
    """
    print("RAG: model_fn initialization started")

    secret = get_secret("arn:aws:secretsmanager:us-east-1:686255941112:secret:HuggingfaceQdrantOpenAPI-UKUXlz")
    if not secret:
        raise ValueError("Failed to retrieve secrets from AWS Secrets Manager")

    openai_api_key = secret.get("OPENAI_API_KEY", "Key not found")
    huggingface_token = secret.get("HUGGINGFACE_TOKEN", "Key not found")
    qdrant_key = secret.get("QDRANT_API_KEY", "Key not found")
    qdrant_url = "https://5ccc316a-ba45-4910-b5f9-15eb181ae895.us-east4-0.gcp.cloud.qdrant.io:6333"
    bucket_name = "capstoneragmodel"

    s3_model_path = model_dir

    openai.api_key = openai_api_key

    logger.info("Initializing HuggingFace embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")

    try:
        logger.info("Connecting to Qdrant vector store")
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        logger.info("[DEBUG] Connected to Qdrant cloud instance")

        collection_name = "policy_pulse_db"

        test_embedding = embeddings.embed_query("")
        if isinstance(test_embedding, list):
            test_embedding = np.array(test_embedding)
        embedding_size = test_embedding.shape[0]
        logger.info(f"[DEBUG] Detected embedding size: {embedding_size}")

        try:
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": embedding_size,
                    "distance": "Cosine"
                }
            )
            logger.info(f"[DEBUG] Collection '{collection_name}' recreated successfully")
        except Exception as e:
            logger.info(f"[ERROR] Failed to recreate collection: {e}")
            raise

        try:
            vectorstore = Qdrant(
                client=qdrant_client,
                collection_name=collection_name,
                embeddings=embeddings
            )
            documents = []  
            if documents:
                vectorstore.add_documents(documents)
                logger.info("[DEBUG] Documents added to Qdrant collection")
        except Exception as e:
            logger.error(f"[ERROR] Failed to add documents to vectorstore: {e}")
            raise

    except Exception as e:
        logger.error(f"[ERROR] Error setting up vector store: {e}")
        raise

    logger.info("Loading Mistral pipeline from the provided model directory")
    mistral_pipe = load_pipeline_from_s3(bucket_name=bucket_name, s3_path=s3_model_path)
    mistral_llm_lc = HuggingFacePipeline(pipeline=mistral_pipe)

    pipeline_components = {
        "llm": mistral_llm_lc,               
        "embeddings": embeddings,            
        "vectorstore": vectorstore,          
        "huggingface_token": huggingface_token,  
        "vectorstore": vectorstore, 
        "qdrant_url": qdrant_url, 
        "qdrant_key": qdrant_key, 
        "open_api_key": openai_api_key, 
        "bucket_name": bucket_name
    }

    logger.info("RAG pipeline components initialized successfully")
    print("RAG: model_fn initialization complete")
    return pipeline_components



#*************************************************************************************************************************************#****************############## Begining of Helper Functions   ######****************************************************************************************************#
#******************************************************************************************************************************************************#

def load_pipeline_from_s3(bucket_name, s3_path, logger=None):
    """
    Load the model and tokenizer from S3 and initialize the HuggingFace pipeline.
    """
    import os
    import shutil

    local_dir = "./temp_model_dir"
    os.makedirs(local_dir, exist_ok=True)

    s3_client = boto3.client("s3")

    paginator = s3_client.get_paginator("list_objects_v2")
    objects = list(paginator.paginate(Bucket=bucket_name, Prefix=s3_path))

    if logger:
        logger.info(f"Downloading model files from s3://{bucket_name}/{s3_path}...")

    for page in objects:
        for obj in page.get("Contents", []):
            s3_file_path = obj["Key"]
            local_file_path = os.path.join(local_dir, os.path.relpath(s3_file_path, s3_path))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3_client.download_file(bucket_name, s3_file_path, local_file_path)
            if logger:
                logger.info(f"Downloaded {s3_file_path} to {local_file_path}")

    if logger:
        logger.info("Loading model and tokenizer from downloaded files...")
    model = AutoModelForCausalLM.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)

    mistral_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1000,
        temperature=0.55,
        top_p=0.9,
        do_sample=True
    )
    if logger:
        logger.info("Pipeline loaded successfully.")

    shutil.rmtree(local_dir, ignore_errors=True)
    return mistral_pipe

def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request body.

    Args:
        request_body (str): The body of the incoming HTTP request.
        request_content_type (str): The content type of the incoming request.

    Returns:
        dict: The deserialized JSON object.

    Raises:
        ValueError: If the content type is unsupported or the JSON payload is invalid.
    """
    logger.info("input_fn called with content type: %s", request_content_type)
    print("Debug: input_fn invoked with content type:", request_content_type)

    if request_content_type == "application/json":
        try:
            data = json.loads(request_body)
            print("Debug: Successfully deserialized request body")
            logger.info("Successfully deserialized JSON request body")

            required_keys = {"company_name", "uuid", "question_type", "question"}
            missing_keys = required_keys - data.keys()
            if missing_keys:
                raise ValueError(f"Missing required keys in JSON payload: {', '.join(missing_keys)}")

            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON request body: {e}")
            raise ValueError(f"Invalid JSON payload: {e}")
    else:
        logger.error(f"Unsupported content type: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")

def fn_local_with_paths(input_data: Dict, bucket_name) -> Tuple[Optional[str], Optional[str], str, str, str]:
    """
    Retrieve the uploaded policy file and GDPR document from an S3 bucket.
    """
    print("Received input data: ", input_data)

    question_type = input_data.get("question_type", "Policy")
    uuid = input_data.get("uuid")
    if not uuid:
        raise ValueError("Input JSON payload missing 'uuid' field")

    company_name = input_data.get("company_name", "").strip() 
    if question_type == "Policy" and not company_name:
        raise ValueError("Input JSON payload missing 'company_name' field for Policy questions.")

    s3_client = boto3.client("s3")

    gdpr_local_path = "/tmp/gdpr.pdf"
    policy_local_path = None

    if question_type == "Policy":
        policy_s3_key = f"pdf_uploads/{uuid}.pdf"
        policy_local_path = f"/tmp/{uuid}.pdf"
        try:
            print(f"Downloading policy file from S3: bucket={bucket_name}, key={policy_s3_key}")
            s3_client.download_file(bucket_name, policy_s3_key, policy_local_path)
            print(f"Policy file downloaded to: {policy_local_path}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to download the policy file for {company_name} from S3: {str(e)}")

    print(f"[DEBUG] Returning paths - Policy: {policy_local_path}, GDPR: {gdpr_local_path}")
    return policy_local_path, gdpr_local_path, company_name, uuid, question_type


def load_pdf_content_with_page_count(pdf_path_or_url: str, is_url: bool = False) -> Tuple[List[str], int]:
    """
    Load the content of a PDF from a file path or URL and return the text from each page along with the page count.

    Args:
        pdf_path_or_url (str): Path to the PDF file or URL of the PDF.
        is_url (bool): Whether the input is a URL.

    Returns:
        Tuple[List[str], int]: A list of text contents from each page and the total page count.
    """
    try:
        if is_url:
            response = requests.get(pdf_path_or_url, stream=True)
            response.raise_for_status() 
            print(f"[DEBUG] Downloading PDF from URL: {pdf_path_or_url}")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(response.content)
                temp_pdf_path = temp_pdf.name
                print(f"[DEBUG] PDF downloaded to temporary path: {temp_pdf_path}")

            pdf_path = temp_pdf_path
        else:
            pdf_path = pdf_path_or_url

        document = fitz.open(pdf_path)
        print(f"[DEBUG] Opened PDF: {pdf_path}, Total Pages: {len(document)}")
        
        contents = [page.get_text() for page in document]
        num_pages = len(contents)
        document.close()

        if is_url:
            os.remove(temp_pdf_path)
            print(f"[DEBUG] Temporary file removed: {temp_pdf_path}")

        return contents, num_pages
    except Exception as e:
        print(f"[ERROR] Failed to load PDF content from {pdf_path_or_url}: {e}")
        return [], 0

def extract_formatted_text(pdf_path: str) -> List[Dict]:
    """
    Extract text with its formatting information from PDF.
    Returns list of dictionaries containing text and its formatting properties.
    """
    formatted_blocks = []
    try:
        doc = fitz.open(pdf_path)
        position = 0

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            prev_y1 = None

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        y0 = line["bbox"][1]
                        line_spacing = y0 - prev_y1 if prev_y1 is not None else 0
                        prev_y1 = line["bbox"][3]

                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                formatted_blocks.append({
                                    "text": text,
                                    "font_name": span["font"],
                                    "font_size": span["size"],
                                    "is_bold": "bold" in span["font"].lower() or span["flags"] & 2**4 != 0,
                                    "line_spacing": line_spacing,
                                    "position": position,
                                    "page_num": page_num + 1
                                })
                            position += len(text) + 1

        return formatted_blocks
    except Exception as e:
        print(f"Error extracting formatted text: {e}")
        return []

def analyze_document_formatting(blocks: List[Dict]) -> Dict:
    """
    Analyze document formatting to establish baseline metrics.
    """
    stats = defaultdict(list)

    for block in blocks:
        stats["font_sizes"].append(block["font_size"])
        stats["line_spacings"].append(block["line_spacing"])

    return {
        "avg_font_size": sum(stats["font_sizes"]) / len(stats["font_sizes"]),
        "max_font_size": max(stats["font_sizes"]),
        "avg_line_spacing": sum(stats["line_spacings"]) / len(stats["line_spacings"]) if stats["line_spacings"] else 0
    }

def identify_potential_headers(blocks: List[Dict], format_stats: Dict) -> List[Dict]:
    """
    Identify potential headers based on formatting characteristics and convert them into confirmed_headers format.
    """
    potential_headers = []
    print("calculating potential headers")

    for block in blocks:
        formatting_score = 0
        characteristics = []

        # Check font size
        if block["font_size"] > format_stats["avg_font_size"]:
            formatting_score += 2
            characteristics.append("larger_font")

        # Check if bold
        if block["is_bold"]:
            formatting_score += 2
            characteristics.append("bold")

        # Check line spacing
        if block["line_spacing"] > format_stats["avg_line_spacing"] * 1.5:
            formatting_score += 1
            characteristics.append("increased_spacing")

        # Check text length
        word_count = len(block["text"].split())
        if word_count <= 10:
            formatting_score += 1
            characteristics.append("short_text")

        # Check for title case or all caps
        if block["text"].istitle() or block["text"].isupper():
            formatting_score += 1
            characteristics.append("title_case_or_caps")

        # If the formatting score is high enough, treat as a confirmed header
        if formatting_score >= 3:
            potential_headers.append({
                "header": block["text"],
                "position": block["position"]
            })

    return potential_headers



def chunk_policy_by_headers(headers: List[Dict], blocks: List[Dict]) -> List[Dict]:
    """
    Chunk the policy text based on section headers using the original formatted blocks.
    """
    chunks = []
    full_text = ""
    current_position = 0

    for block in blocks:
        full_text += block["text"] + "\n"

    for i, header_data in enumerate(headers):
        start_pos = header_data["position"]
        end_pos = headers[i + 1]["position"] if i + 1 < len(headers) else len(full_text)
        chunk_text = full_text[start_pos:end_pos].strip()

        chunks.append({
            "header": header_data["header"],
            "content": chunk_text
        })

    return chunks

init_cache_db()
HEADER_CACHE = load_cache_from_db()

def process_policy_with_cache(pdf_path: str, company_name: str, openai_api_key: str) -> Dict:
    if company_name in HEADER_CACHE:
        print(f"[INFO] Using cached headers for company: {company_name}")
        cached_headers = HEADER_CACHE[company_name]
    else:
        try:
            formatted_blocks = extract_formatted_text(pdf_path)
            if not formatted_blocks:
                return {"status": "error", "pdf_path": pdf_path, "message": "Failed to extract formatted text"}

            format_stats = analyze_document_formatting(formatted_blocks)

            confirmed_headers = identify_potential_headers(formatted_blocks, format_stats)

            HEADER_CACHE[company_name] = confirmed_headers
            save_cache_to_db(company_name, confirmed_headers)
            cached_headers = confirmed_headers

        except Exception as e:
            return {"status": "error", "pdf_path": pdf_path, "message": str(e)}

    if 'formatted_blocks' not in locals():
        print("entering here formatted_blocks:")
        formatted_blocks = extract_formatted_text(pdf_path)
    chunks = chunk_policy_by_headers(cached_headers, formatted_blocks)

    return {
        "status": "success",
        "pdf_path": pdf_path,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    
def further_chunk_policy(company_name: str, uuid: str, policy: Dict, chunk_size: int = 1000, chunk_overlap: int = 50) -> List[Dict]:
    """
    Further split each chunk's content for a single company policy using a Recursive Text Splitter.

    Args:
        policy (Dict): A dictionary representing the policy document with its chunks.
        chunk_size (int): The maximum size of each sub-chunk.
        chunk_overlap (int): The overlap between consecutive sub-chunks.

    Returns:
        List[Dict]: A list of refined sub-chunks with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    refined_chunks = []


    for chunk in policy["chunks"]:
        try:
            sub_chunks = text_splitter.split_text(chunk["content"])

            for i, sub_chunk in enumerate(sub_chunks, 1):
                refined_chunks.append({
                    "page_content": sub_chunk,
                    "metadata": {
                        "company_name": company_name,
                        "uuid": uuid,
                        "pdf_path": policy["pdf_path"],
                        "header": chunk["header"],
                        "chunk_index": i,
                        "total_sub_chunks": len(sub_chunks),
                        "document_type": "Policy"
                    }
                })
        except Exception as e:
            print(f"Error processing chunk for company {company_name} with header '{chunk['header']}': {str(e)}")
            continue

    return refined_chunks


def further_chunk_gdpr_content(company_name: str, uuid: str,  initial_documents: List[Document]) -> List[Document]:
    """Further split GDPR content into smaller chunks while preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    gdpr_chunks = []

    for document in initial_documents:
        sub_chunks = text_splitter.split_text(document.page_content)
        for sub_chunk in sub_chunks:
            gdpr_chunks.append(Document(
                page_content=sub_chunk,
                metadata={
                    "header": document.metadata.get("header"),
                    "uuid": uuid,
                    "document_type": "GDPR"
                }
            ))
    print(f"[DEBUG] Further chunked GDPR content for company {company_name}")
    return gdpr_chunks


def prepare_documents_for_vectorstore(chunks: List[Document]) -> List[Document]:
    """Prepare documents for vector store with consistent metadata handling."""
    documents = []
    
    bucket_name = "capstoneragmodel"
    s3_base_path = "pdf_uploads"

    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            chunk = Document(page_content=chunk['page_content'], metadata=chunk['metadata'])

        doc_type = chunk.metadata.get("document_type", "Unknown")
        metadata_company_name = chunk.metadata.get("company_name", "").lower()
        uuid = chunk.metadata.get("uuid", "")

        doc_link = (
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
            if doc_type == "GDPR"
            else f"https://{bucket_name}.s3.amazonaws.com/{s3_base_path}/{uuid}.pdf"
        )

        metadata = {
            "chunk_number": idx + 1,
            "document_type": doc_type,
            "uuid": uuid,
            "company_name": metadata_company_name,  # Retain company_name for readability
            "header": chunk.metadata.get("header", ""),
            "doc_link": doc_link
        }

        documents.append(Document(
            page_content=chunk.page_content,
            metadata=metadata
        ))
        print(f"[DEBUG] Prepared document {idx + 1} with metadata: {metadata}")

    return documents


def setup_vectorstore(documents, embeddings, qdrant_url, qdrant_api_key):
    """
    Setup Qdrant Vector Store with provided documents.
    """
    from qdrant_client import QdrantClient
    from langchain.vectorstores import Qdrant

    print("[INFO] Setting up Qdrant vector store")

    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    collection_name = "policy_pulse_db"

    test_embedding = embeddings.embed_query("")
    if isinstance(test_embedding, list):
        test_embedding = np.array(test_embedding)
    embedding_size = test_embedding.shape[0]

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={"size": embedding_size, "distance": "Cosine"}
    )

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    vectorstore.add_documents(documents)

    return vectorstore

def create_policy_prompt_template() -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for the GDPR compliance expert.

    Returns:
        ChatPromptTemplate: Configured template for generating structured answers.
    """
    prompt_role = "You are a GDPR compliance expert tasked with providing accurate, structured answers."
    prompt_task = "Using only the provided context, generate a comprehensive response that includes all required sections."
    prompt_guidance = "Think carefully and ensure that your answer is well-structured and solid for law firms and startups."
    prompt_context = "Context:\n{context}"
    prompt_question = "Question:\n{question}"
    prompt_instruction = """Answer the question by referring specifically to the context.
    Structure your response exactly as follows:

    Main Answer:
    [Provide a concise yet comprehensive answer using the context information.]

    Key Points:
    • [List key points as bullet points]
    • [Each point should start with a bullet point]
    • [Extract at least 3-4 key points]

    GDPR References:
    [Must include specific GDPR articles mentioned or most relevant to the context. Use format: 'Article X - Title']"""

    policy_pulse_template = f"""<s>[INST] {prompt_role}
    {prompt_task}
    {prompt_guidance}

    {prompt_context}

    {prompt_question}

    {prompt_instruction} [/INST]"""

    return ChatPromptTemplate.from_template(policy_pulse_template)

def process_single_question(uuid, company_name, question_type, question, qdrant_vectorstore, gdpr_retriever, rag_chain_mistral, ground_truth_df):
    """
    Process a single question using the RAG system.
    """
    logger.info(f"Processing question: '{question}' for company: {company_name}")

    try:
        gdpr_results = []
        policy_results = []

        if question_type == 'Policy':
            logger.info(f"Retrieving Policy content for UUID: {uuid}")
            try:
                combined_filter = {"uuid": uuid}
                policy_retriever = qdrant_vectorstore.as_retriever(search_kwargs={'filter': combined_filter})
                policy_results = policy_retriever.get_relevant_documents(question)
                logger.info(f"Retrieved {len(policy_results)} policy documents")

                logger.info("Retrieving related GDPR articles")
                gdpr_results = gdpr_retriever.get_relevant_documents(question)
                logger.info(f"Retrieved {len(gdpr_results)} GDPR documents")
            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                raise

        elif question_type == 'GDPR':
            logger.info("Retrieving GDPR content")
            try:
                gdpr_results = gdpr_retriever.get_relevant_documents(
                    question, metadata_filters={"document_type": question_type}
                )
                logger.info(f"Retrieved {len(gdpr_results)} GDPR documents")
            except Exception as e:
                logger.error(f"Error retrieving GDPR documents: {str(e)}")
                raise

        # Process retrieved documents
        try:
            context_snippets = []
            for doc in policy_results + gdpr_results:
                if hasattr(doc, 'page_content') and doc.page_content:
                    context_snippets.append(doc.page_content[:500])
            
            context_snippets_str = " ".join(context_snippets) if context_snippets else ""
            logger.info(f"Combined {len(context_snippets)} context snippets")

            if not context_snippets_str:
                logger.warning("No context retrieved")
                return {
                    "question": question,
                    "contexts": "No relevant context found.",
                    "answer": "Unable to generate answer due to lack of context.",
                    "ground_truth": "N/A"
                }

            # Generate answer
            logger.info("Generating answer using RAG chain")
            rag_input = {
                "context": context_snippets_str,
                "question": question
            }
            
            # Debug log the input
            logger.debug(f"RAG input - Question: {question}")
            logger.debug(f"RAG input - Context length: {len(context_snippets_str)}")
            
            response = rag_chain_mistral.invoke(rag_input)
            generated_answer = clean_generated_response(response)
            logger.info("Answer generated successfully")

            # Get ground truth
            ground_truth = "No ground truth available."
            if not ground_truth_df.empty:
                matching_truth = ground_truth_df[ground_truth_df['question'] == question]
                if not matching_truth.empty:
                    ground_truth = matching_truth['ground_truth'].iloc[0]
                    logger.info("Ground truth retrieved successfully")

            return {
                "question": question,
                "contexts": context_snippets_str,
                "answer": generated_answer,
                "ground_truth": ground_truth
            }

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    except Exception as e:
        logger.error(f"Error in process_single_question: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "question": question,
            "contexts": "Error processing question",
            "answer": f"Error: {str(e)}",
            "ground_truth": "N/A"
        }

def clean_generated_response(response: str) -> str:
    """
    This function cleans the generated response to remove any unwanted content
    such as initial instructions or repeated context.
    """
    cleaned_response = re.sub(r'Human: <s>\[INST\].*?\[/INST\]', '', response, flags=re.DOTALL)

    cleaned_response = re.sub(r'Context:\s*{.*?}\n', '', cleaned_response, flags=re.DOTALL)

    cleaned_response = re.sub(r'Question:\s*{.*?}\n', '', cleaned_response, flags=re.DOTALL)

    cleaned_response = cleaned_response.strip()

    return cleaned_response

def get_ground_truth(question, ground_truth_df):
    """
    Retrieve the ground truth answer for a given question from the ground truth DataFrame.

    Args:
        question (str): The input question.
        ground_truth_df (pd.DataFrame): DataFrame containing questions and their ground truth answers.

    Returns:
        str: The ground truth answer, or a default message if not found.
    """
    match = ground_truth_df.loc[ground_truth_df['question'] == question]
    if not match.empty:
        return match['ground_truth'].iloc[0]  # Return the first match
    return "No ground truth available"

def chunk_gdpr_by_section(gdpr_contents: List[str]) -> List[Document]:
    """
    Chunk GDPR text into sections with caching.
    Uses a constant key for caching since GDPR content is the same for all companies.
    """
    try:
        gdpr_cache_key = "GDPR_GLOBAL"  # Single cache key for all GDPR questions
        
        # Check cache first
        if gdpr_cache_key in HEADER_CACHE:
            print("[INFO] Using cached GDPR sections")
            return HEADER_CACHE[gdpr_cache_key]

        print("[INFO] Processing GDPR document and creating new cache")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
        documents = []
        full_text = "\n".join(gdpr_contents)

        # Rest of your existing GDPR processing code...
        recital_pattern = re.compile(r"\((\d+)\)\s", re.MULTILINE)
        article_pattern = re.compile(r"(Article\s(\d+))\b", re.IGNORECASE)

        all_matches = [(match.start(), match.group(1), 'recital') for match in re.finditer(recital_pattern, full_text)]
        all_matches += [(match.start(), match.group(1), 'article', match.group(2)) for match in re.finditer(article_pattern, full_text)]
        all_matches.sort()

        for i, (start, header, section_type, *article_number) in enumerate(all_matches):
            end = all_matches[i + 1][0] if i + 1 < len(all_matches) else len(full_text)
            section_text = full_text[start:end].strip()

            section_header = None
            if section_type == 'article' and article_number:
                article_number = article_number[0].strip()
                section_header = f"Article {article_number}"
            elif section_type == 'recital':
                section_header = f"Recital {header}"

            chunks = text_splitter.split_text(section_text)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"header": section_header}
                ))

        # Cache using global GDPR key
        HEADER_CACHE[gdpr_cache_key] = documents
        save_cache_to_db(gdpr_cache_key, documents)
        
        print(f"\nTotal documents created and cached: {len(documents)}")
        return documents

    except Exception as e:
        print(f"Error processing GDPR contents: {e}")
        return []
#*************************************************************************************************************************************#****************############## End of Helper Function   ######**********************************************************************************************************#
#******************************************************************************************************************************************************#

def predict_fn(input_data, model, ground_truth_df):
    """
    Predict function to process a question and include the ground truth answer.
    """
    logger.info("predict_fn started with input:")
    logger.info(f"Input data: {json.dumps(input_data, indent=2)}")

    bucket_name = "capstoneragmodel"
    vectorstore = model["vectorstore"]
    base_embeddings = model["embeddings"]
    qdrant_url = model["qdrant_url"]
    qdrant_key = model["qdrant_key"]
    openai_api_key = model["open_api_key"]
    mistral_llm_lc = model["llm"]
    
    # Validate input data
    if not isinstance(input_data, dict):
        logger.error("Input data must be a dictionary")
        raise ValueError("Input data must be a dictionary")
        
    sample_question = input_data.get("question")
    if not sample_question:
        logger.error("No question found in input data")
        raise ValueError("No question found in input data")
        
    logger.info(f"Processing question: {sample_question}")
    
    try:
        policy_paths, gdpr_path, company_name, uuid, question_type = fn_local_with_paths(input_data, bucket_name)
        logger.info(f"Paths retrieved - Policy: {policy_paths}, GDPR: {gdpr_path}")
        logger.info(f"Metadata - Company: {company_name}, UUID: {uuid}, Type: {question_type}")
        
        refined_chunks = []
        gdpr_sentence_chunks = []
        
        if question_type == "Policy":
            policy_contents, num_pages = load_pdf_content_with_page_count(policy_paths)
            logger.info(f"Loaded policy PDF with {num_pages} pages")
            result = process_policy_with_cache(policy_paths, company_name, openai_api_key)
            refined_chunks = further_chunk_policy(company_name, uuid, result)
            logger.info(f"Generated {len(refined_chunks)} policy chunks")
        
        elif question_type == "GDPR":
            gdpr_contents, num_pages = load_pdf_content_with_page_count(gdpr_path)
            logger.info(f"Loaded GDPR PDF with {num_pages} pages")
            gdpr_sentence_chunks = chunk_gdpr_by_section(gdpr_contents)
            gdpr_final_chunks = further_chunk_gdpr_content(company_name,uuid, gdpr_sentence_chunks)
            refined_chunks = gdpr_final_chunks
            logger.info(f"Generated {len(refined_chunks)} GDPR chunks")
            
        all_chunks = refined_chunks
        all_documents = prepare_documents_for_vectorstore(all_chunks)
        logger.info(f"Prepared {len(all_documents)} documents for vector store")

        qdrant_vectorstore = setup_vectorstore(
            documents=all_documents,
            embeddings=base_embeddings,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_key
        )
        logger.info("Vector store setup complete")
        
        # Setup retrievers
        combined_filter = {"document_type": "GDPR"}
        gdpr_retriever = qdrant_vectorstore.as_retriever(search_kwargs={'filter': combined_filter})
        logger.info(f"Retriever setup complete for company: {company_name}")
        
        policy_prompt = create_policy_prompt_template()
        rag_chain_mistral = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | policy_prompt
            | mistral_llm_lc
            | RunnablePassthrough()
        )
        logger.info("RAG chain setup complete")
        
        # Process question
        logger.info("Processing question with RAG chain")
        result = process_single_question(
            uuid=uuid,
            company_name=company_name,
            question_type=question_type,
            question=sample_question,
            qdrant_vectorstore=qdrant_vectorstore,
            gdpr_retriever=gdpr_retriever,
            rag_chain_mistral=rag_chain_mistral,
            ground_truth_df=ground_truth_df
        )
        
        # Add ground truth
        logger.info("Adding ground truth to result")
        result['ground_truth'] = get_ground_truth(sample_question, ground_truth_df)
        
        # Convert to DataFrame
        result_df = pd.DataFrame([result])
        logger.info("Created result DataFrame")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
        
def predict_fn_multiple(input_data_list, model, ground):
    """
    Process multiple questions from a list of input data using the RAG pipeline.

    Args:
        input_data_list (list): List of dictionaries containing company name, UUID, question type, etc.
        model: RAG pipeline components.

    Returns:
        pd.DataFrame: A DataFrame containing the results for all processed questions.
    """
    logger.info("Processing multiple questions...")

    results = []  # List to collect results for all questions

    for input_data in input_data_list:
        try:
            # Process each question using the original predict_fn logic
            logger.info(f"Processing question: {input_data.get('question', '')}")
              
            print("ground:", ground.head(1))
            
            # Retrieve the result for the current question
            result = predict_fn(input_data, model, ground)
            
            # Append the result to the results list
            if isinstance(result, pd.DataFrame):
                results.append(result)  # Append DataFrame result
            else:
                results.append(pd.DataFrame([result]))  # Handle dictionary output as a DataFrame
        except Exception as e:
            logger.error(f"Error processing question {input_data.get('question', '')}: {str(e)}")
            results.append(pd.DataFrame([{"error": str(e), "question": input_data.get("question", "")}]))

    final_results = pd.concat(results, ignore_index=True)
    
    output_file = "./ragas_eval/bilt/bilt.csv"
    final_results.to_csv(output_file, index=False)
    print(f"[INFO] Results saved to {output_file}")

    return final_results

    
def output_fn(prediction, content_type):
    """
    Serialize the prediction output for the client.

    Args:
        prediction (str): The prediction result as a plain string.
        content_type (str): The desired content type for the response.

    Returns:
        str: Serialized prediction output in the specified format.
    """
    if content_type == "application/json":
        try:
            serialized_output = json.dumps({"answer": prediction})
            deserialized_response = json.loads(serialized_output)
            answer = deserialized_response.get("answer", "")
            if not answer:
                return "Error: Response does not contain a valid 'answer' field."

            try:
                main_answer, key_points, gdpr_references = "", "", ""

                if "Main Answer:" in answer:
                    main_answer = answer.split("Main Answer:")[1].split("Key Points:")[0].strip()
                if "Key Points:" in answer:
                    key_points = answer.split("Key Points:")[1].split("GDPR References:")[0].strip()
                if "GDPR References:" in answer:
                    gdpr_references = answer.split("GDPR References:")[1].strip()

                formatted_output = (
                    f"\nMain Answer:\n{main_answer}\n\n"
                    f"Key Points:\n{key_points}\n\n"
                    f"GDPR References:\n{gdpr_references}"
                    )
                return formatted_output
            except Exception as e:
                return f"Error in formatting: {e}"
        
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize prediction to JSON: {e}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def main():
    """
    Main function to test the pipeline loading and processing with input data.
    """
    success, message = reset_cache_and_db()
    if not success:
        print(f"Failed to reset cache and database: {message}")
        return
        
    verify_success, verify_message = verify_reset()
    if not verify_success:
        print(f"Failed to verify reset: {verify_message}")
        return
    start_time = time.time()

    model = model_fn('models/mistral')

    try:
        with open("exp.json", "r") as f:
            input_data_list = json.load(f)  # Load the list of input data
    except Exception as e:
        print(f"[ERROR] Failed to load input data from exp.json: {e}")
        return

    result_df = predict_fn_multiple(input_data_list, model, ground_truth_df)  

    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()


