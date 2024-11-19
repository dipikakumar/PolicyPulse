!pip install openai==0.28.0
!pip install sentence_transformers


import logging
import openai
import os
import torch
import os
import bs4
import json
import numpy as np
import time
import pandas as pd
from pprint import pprint
import locale
from transformers import AutoTokenizer , AutoModelForCausalLM
from transformers import pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utils.math import cosine_similarity
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PubMedLoader
from langchain.schema import Document

from langchain.document_loaders import PyMuPDFLoader
import fitz
import requests
from langchain.embeddings import HuggingFaceEmbeddings
import logging
from typing import Tuple, List, Optional
from pathlib import Path
import tempfile
import openai
import fitz
from typing import List, Dict
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

openai.api_key  = os.getenv("KEY")
gdpr_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
base_embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")


def load_pdf_content_with_page_count(pdf_path: str) -> Tuple[List[str], int]:
    """
    Load the content of a PDF and return the text from each page along with the total page count.

    This function uses PyMuPDF to extract text from all pages of a PDF document,
    returning the text content as a list of strings (one string per page)
    and the total number of pages in the document.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        tuple: (List of page contents as strings, Total number of pages)
    """
    try:
        document = fitz.open(pdf_path)
        contents = [page.get_text() for page in document]
        num_pages = len(contents)
        document.close()
        return contents, num_pages
    except Exception as e:
        logger.error(f"Failed to load PDF content from {pdf_path}: {e}")
        return [], 0


def load_policy_documents(gdpr_url: str) -> Tuple[List[List[str]], List[str]]:
    """
    Load multiple policy documents and the GDPR document with proper error handling.

    Parameters:
    - policy_paths (List[str]): List of paths to company policy PDFs
    - gdpr_url (str): URL to the GDPR reference PDF

    Returns:
    - tuple: (list of policy contents for each document, gdpr contents)
    """
    gdpr_contents = []
    gdpr_temp_path = download_pdf_from_url(gdpr_url)

    if gdpr_temp_path:
        try:
            gdpr_contents, gdpr_pages = load_pdf_content_with_page_count(str(gdpr_temp_path))
            logger.info(f"Loaded GDPR document: {gdpr_pages} pages")
            gdpr_temp_path.unlink()

        except Exception as e:
            logger.error(f"Error loading GDPR content: {e}")
    else:
        logger.error("Failed to download GDPR document")

    return  gdpr_contents

gdpr_contents = load_policy_documents(gdpr_url)

print(f"Total pages in GDPR PDF: {len(gdpr_contents)}")

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
    Identify potential headers based on formatting characteristics.
    """
    potential_headers = []

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

        if formatting_score >= 3:
            potential_headers.append({
                "text": block["text"],
                "position": block["position"],
                "characteristics": characteristics,
                "formatting_score": formatting_score
            })

    return potential_headers

def confirm_headers_with_gpt(potential_headers: List[Dict]) -> List[Dict]:
    """
    Use GPT to confirm which potential headers are actual privacy policy section headers.
    """
    confirmed_headers = []

    for header in potential_headers:
        prompt = f"""Analyze if this text represents a privacy policy section header.

Text: "{header['text']}"

Formatting characteristics detected:
{', '.join(header['characteristics'])}

Consider:
1. Is this a typical privacy policy topic? (e.g., data collection, processing, sharing, security, rights)
2. Does it look like a header based on its formatting? (detected: {header['formatting_score']}/7 header characteristics)
3. Is it concise and descriptive?
4. Does it introduce a new section rather than being part of content?

Is this a main section header? Answer only 'yes' or 'no'."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying section headers in privacy policies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0
            )

            if response['choices'][0]['message']['content'].strip().lower() == 'yes':
                confirmed_headers.append({
                    "header": header["text"],
                    "position": header["position"]
                })

        except Exception as e:
            print(f"Error in GPT confirmation: {e}")
            continue

    return confirmed_headers

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

def process_policy(pdf_path: str) -> Dict:
    """
    Process a single PDF policy document and extract sections.
    """
    try:
        formatted_blocks = extract_formatted_text(pdf_path)
        if not formatted_blocks:
            return {"status": "error", "pdf_path": pdf_path, "message": "Failed to extract formatted text"}

        format_stats = analyze_document_formatting(formatted_blocks)
        potential_headers = identify_potential_headers(formatted_blocks, format_stats)
        confirmed_headers = confirm_headers_with_gpt(potential_headers)

        if not confirmed_headers:
            return {"status": "error", "pdf_path": pdf_path, "message": "No headers found in the document"}

        chunks = chunk_policy_by_headers(confirmed_headers, formatted_blocks)

        return {
            "status": "success",
            "pdf_path": pdf_path,
            "total_chunks": len(chunks),
            "chunks": chunks
        }

    except Exception as e:
        return {"status": "error", "pdf_path": pdf_path, "message": str(e)}

result = process_policy(policy_paths)
if result["status"] == "success":
    print(f"\nProcessed Result for {result['pdf_path']}:")
    print(f"Total number of chunks created: {result['total_chunks']}\n")
    for chunk in result["chunks"]:
        print(f"Header: {chunk['header']}")
        print(f"Content Snippet: {chunk['content'][:150]}...\n")
else:
    print(f"Error processing {result['pdf_path']}: {result['message']}")


def further_chunk_policy(policy: Dict, chunk_size: int = 1000, chunk_overlap: int = 50) -> List[Dict]:
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

    #pdf_path = policy["pdf_path"]
    company_name = policy_paths.split('/')[-1].replace('.pdf', '').capitalize()

    for chunk in policy["chunks"]:
        try:
            sub_chunks = text_splitter.split_text(chunk["content"])

            for i, sub_chunk in enumerate(sub_chunks, 1):
                refined_chunks.append({
                    "page_content": sub_chunk,
                    "metadata": {
                        "company_name": company_name,
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


refined_chunks = further_chunk_policy(result)
print(f"Total number of sub-chunks created: {len(refined_chunks)}\n")
for i, final_chunk in enumerate(refined_chunks, 1):
  print(f"Chunk {i}/{len(refined_chunks)}")
  print(f"PDF Path: {final_chunk['metadata']['pdf_path']}")
  print(f"Header: {final_chunk['metadata']['header']}")
  print(f"Sub-chunk {final_chunk['metadata']['chunk_index']}/{final_chunk['metadata']['total_sub_chunks']}")
  print(f"Content Snippet: {final_chunk['page_content'][:150]}...\n")


def chunk_gdpr_by_section(gdpr_contents: List[str]) -> List[Document]:
    """
    Chunk GDPR text into sections based on articles and recitals, without mapping to policy categories.

    Parameters:
    - gdpr_contents (List[str]): List containing GDPR text.

    Returns:
    - List[Document]: List of Document objects, each representing a chunk of a GDPR section.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
        documents = []
        full_text = "\n".join(gdpr_contents)

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
                    metadata={
                        "header": section_header
                    }
                ))

        print(f"\nTotal documents created: {len(documents)}")
        return documents

    except Exception as e:
        print(f"Error processing GDPR contents: {e}")
        return []


gdpr_sentence_chunks = chunk_gdpr_by_section(gdpr_contents)
print("\nSample chunks:")
for doc in gdpr_sentence_chunks[:5]:
  print(f"\n header: {doc.metadata['header']}")
  print(f"Content Snippet: {doc.page_content[:150]}...")


def further_chunk_gdpr_content(company_name: str, initial_documents: List[Document]) -> List[Document]:
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
                    "document_type": "GDPR"
                }
            ))
    print(f"[DEBUG] Further chunked GDPR content for company {company_name}")
    return gdpr_chunks

def initialize_mistral_pipeline(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens: int = 1000,
    temperature: float = 0.55,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    offload_folder: str = "offload_folder"
) -> pipeline:
    """
    Initialize the Mistral text-generation pipeline with quantization.

    Args:
        model_name (str): Name of the pre-trained model to load.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for text generation.
        top_p (float): Top-p sampling for nucleus sampling.
        repetition_penalty (float): Repetition penalty for text generation.
        offload_folder (str): Directory for offloading large model files.

    Returns:
        pipeline: Configured HuggingFace pipeline for text generation.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    llm_mistral_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        offload_folder=offload_folder
    )

    llm_mistral_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        truncation=True
    )

    mistral_pipe = pipeline(
        "text-generation",
        model=llm_mistral_model,
        tokenizer=llm_mistral_tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=repetition_penalty,
        torch_dtype=torch.float16
    )

    mistral_pipe.model.config.pad_token_id = mistral_pipe.model.config.eos_token_id

    return mistral_pipe

mistral_pipe = initialize_mistral_pipeline(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",  # Optional: Change if using a different model
    max_new_tokens=1000,
    temperature=0.55,
    top_p=0.9,
    repetition_penalty=1.3,
    offload_folder="offload_folder"
)

# Wrap the pipeline in HuggingFacePipeline for LangChain
mistral_llm_lc = HuggingFacePipeline(pipeline=mistral_pipe)

def process_validation_questions(json_file_path: str) -> List[str]:
    """
    Load and process validation questions from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing validation questions.

    Returns:
        List[str]: A list of processed questions.
    """
    try:
        # Load the JSON file
        with open(json_file_path, 'r') as json_file:
            loaded_validation = json.load(json_file)

        # Validate that the data is in the expected format (list of dictionaries)
        if not isinstance(loaded_validation, list):
            raise ValueError("JSON file content is not a list of questions")

        print("Loaded Validation Questions:")
        questions = []
        for idx, question_data in enumerate(loaded_validation, 1):  # Iterate through the list directly
            question_text = question_data.get("question", "No question found")
            print(f"Question {idx}: {question_text}")
            questions.append(question_text)

        # Processing each question
        for question in questions:
            print(f"Processing question: {question}")

        return questions
    except Exception as e:
        print(f"Error processing validation questions: {e}")
        return []

json_file_path = 'test.json'
processed_questions = process_validation_questions(json_file_path)


def initialize_qdrant_collection(url: str, api_key: str, collection_name: str, vectors_config: Dict[str, str]) -> None:
    """
    Initialize a Qdrant collection with the specified configuration.

    Args:
        url (str): The Qdrant instance URL.
        api_key (str): API key for accessing Qdrant.
        collection_name (str): Name of the collection to create or recreate.
        vectors_config (Dict[str, str]): Configuration for the vectors (e.g., size, distance metric).

    Returns:
        None
    """
    try:
        # Initialize the Qdrant client
        qdrant_client = QdrantClient(url=url, api_key=api_key)

        # Create or recreate the collection
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Error creating collection '{collection_name}':", e)

QDRANT_URL = "https://5ccc316a-ba45-4910-b5f9-15eb181ae895.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("KEY")

initialize_qdrant_collection(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="policy_pulse_db",
    vectors_config={"size": 128, "distance": "Cosine"}
)


def prepare_documents_for_vectorstore(chunks: List[Document]) -> List[Document]:
    """Prepare documents for vector store with consistent metadata handling."""
    documents = []

    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            chunk = Document(page_content=chunk['page_content'], metadata=chunk['metadata'])

        doc_type = chunk.metadata.get("document_type", "Unknown")

        metadata_company_name = chunk.metadata.get("company_name", "").lower()

        metadata = {
            "chunk_number": idx + 1,
            "document_type": doc_type,
            "company_name": metadata_company_name,
            "header": chunk.metadata.get("header", ""),
            "doc_link": (
                "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
                if doc_type == "GDPR"
                else f"{metadata_company_name}.pdf"
            )
        }

        documents.append(Document(
            page_content=chunk.page_content,
            metadata=metadata
        ))
        print(f"[DEBUG] Prepared document {idx + 1} with metadata: {metadata}")

    return documents

def setup_vectorstore(documents: List[Document], embeddings, qdrant_url: str, qdrant_api_key: str) -> Optional[Qdrant]:
    """Initialize and populate the vector store using a Qdrant cloud instance."""
    try:
        # Connect to the Qdrant client
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("[DEBUG] Connected to Qdrant cloud instance")

        collection_name = "policy_pulse_db"

        test_embedding = embeddings.embed_query("")
        print(f"[DEBUG] Embedding output type: {type(test_embedding)}")
        print(f"[DEBUG] Embedding output sample: {test_embedding}")

        test_embedding = embeddings.embed_query("")
        if isinstance(test_embedding, list):
          test_embedding = np.array(test_embedding)
        embedding_size = test_embedding.shape[0]
        print(f"[DEBUG] Detected embedding size: {embedding_size}")

        # Recreate the collection
        try:
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": embedding_size,
                    "distance": "Cosine"
                }
            )
            print(f"[DEBUG] Collection '{collection_name}' recreated successfully")
        except Exception as e:
            print(f"[ERROR] Failed to recreate collection: {e}")
            return None

        # Add documents to the vectorstore
        try:
            vectorstore = Qdrant(
                client=qdrant_client,
                collection_name=collection_name,
                embeddings=embeddings
            )
            vectorstore.add_documents(documents)
            print("[DEBUG] Documents added to Qdrant collection")
            return vectorstore
        except Exception as e:
            print(f"[ERROR] Failed to add documents to vectorstore: {e}")
            return None

    except Exception as e:
        print(f"[ERROR] Error setting up vector store: {e}")
        return None

company_name = "bitpay" #replace this with uuid as we discussed (replace in metadata and retrieve docs based on uuid)
gdpr_final_chunks = further_chunk_gdpr_content(company_name, gdpr_sentence_chunks)

all_chunks = gdpr_final_chunks + refined_chunks
all_documents = prepare_documents_for_vectorstore(all_chunks)
print("[DEBUG] Prepared Documents for Vector Store:")
for idx, document in enumerate(all_documents):
  print(f"Document {idx + 1}:")
  print(f"Page Content: {document.page_content[:100]}...")
  print(f"Metadata: {document.metadata}")
  print("-" * 50)

qdrant_vectorstore = setup_vectorstore(
    documents=all_documents,
    embeddings=base_embeddings,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)
combined_filter = {"document_type": "GDPR"}
gdpr_retriever = qdrant_vectorstore.as_retriever(search_kwargs={'filter': combined_filter})
print(f"[TEST] Applying retrieval filter with company_name: '{company_name}'")


prompt_role = "You are a GDPR compliance expert tasked with providing accurate, structured answers."
prompt_task = "Using only the provided context, generate a comprehensive response that includes all required sections."
prompt_context = "Context:\n{context}"
prompt_question = "Question:\n{question}"
prompt_instruction = """Answer the question by referring specifically to the context.
Structure your response exactly as follows:

Main Answer:
[Provide a concise answer using the context information]

Key Points:
• [List key points as bullet points]
• [Each point should start with a bullet point]
• [Extract at least 3-4 key points]

GDPR References:
[Must include specific GDPR articles mentioned or most relevant to the context. Use format: 'Article X - Title']"""

policy_pulse_template = f"""<s>[INST] {prompt_role}
{prompt_task}

{prompt_context}

{prompt_question}

{prompt_instruction} [/INST]"""

policy_prompt = ChatPromptTemplate.from_template(policy_pulse_template)


def classify_question_type(question: str) -> tuple:
    """
    Classify if a question is about a specific company policy or general GDPR.
    Returns "Policy" if a company name is identified in the question,
    otherwise returns "GDPR".

    Args:
        question: The question text

    Returns:
        Tuple: ("Policy" or "GDPR", matched company name or None)
    """
    company_name = "bitpay"
    if company_name:
      company_name = company_name.lower()
    return "Policy" if company_name else ("GDPR", None)

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


rag_chain_mistral = (
    {
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough()
    }
    | policy_prompt
    | mistral_llm_lc
    | RunnablePassthrough()
)


def process_single_question(question, qdrant_vectorstore, gdpr_retriever, rag_chain_mistral):
    """
    Process a single question using the RAG system and return the generated answer.

    Args:
        question (str): The question to process.
        qdrant_vectorstore: Qdrant vector store for policy documents.
        gdpr_retriever: Retriever for GDPR-related documents.
        rag_chain_mistral: RAG chain pipeline for generating answers.

    Returns:
        dict: A result including the question, context, generated answer, and document sources.
    """
    print(f"[PROCESSING] Question: {question}")

    dynamic_company_name = "bitpay"  # Replace with logic to extract or infer company name if needed
    question_type = classify_question_type(question)
    print(f"Dynamically extracted company name: {dynamic_company_name}")
    print(f"Dynamically extracted question_type: {question_type}")

    gdpr_results = []
    policy_results = []

    if question_type == 'GDPR':
        print(f"[PROCESSING] Retrieving GDPR content for question: {question}")
        gdpr_results = gdpr_retriever.get_relevant_documents(
            question, metadata_filters={"document_type": question_type}
        )

    elif question_type == 'Policy':
        print(f"[PROCESSING] Retrieving Policy content for question: {question} ({dynamic_company_name})")
        combined_filter = {"company_name": dynamic_company_name}
        policy_retriever = qdrant_vectorstore.as_retriever(search_kwargs={'filter': combined_filter})
        policy_results = policy_retriever.get_relevant_documents(question)

        if not policy_results:
            print(f"[WARNING] No policy documents found for '{dynamic_company_name}'. Using broader retrieval.")
            policy_results = qdrant_vectorstore.as_retriever().get_relevant_documents(question)

        print(f"[INFO] Retrieving GDPR articles for the question using semantic similarity")
        gdpr_results = gdpr_retriever.get_relevant_documents(question)

    else:
        print(f"[ERROR] Unable to classify question type.")
        return {
            "question": question,
            "contexts": "No relevant context found.",
            "answer": "Unable to classify question type.",
            "doc_source": "N/A"
        }

    context_snippets = [doc.page_content[:500] for doc in policy_results + gdpr_results if doc.page_content]
    context_snippets_str = " ".join(context_snippets)

    document_sources = [doc.metadata.get('doc_link', 'N/A') for doc in policy_results + gdpr_results]
    document_sources_str = ", ".join(document_sources)

    if not context_snippets_str:
        print(f"[ERROR] No context retrieved for question: {question}.")
        return {
            "question": question,
            "contexts": "No relevant context found for answering the question.",
            "answer": "No answer generated.",
            "doc_source": "N/A"
        }

    rag_input = {"context": context_snippets_str, "question": question}
    response = rag_chain_mistral.invoke(rag_input)
    generated_answer = clean_generated_response(response)

    return {
        "question": question,
        "contexts": context_snippets_str,
        "answer": generated_answer,
        "doc_source": document_sources_str
    }

#user_question = input("Ask your question: ")
user_question = 'When and where does BitPay collect personal data?'

result = process_single_question(
    question=user_question,
    qdrant_vectorstore=qdrant_vectorstore,
    gdpr_retriever=gdpr_retriever,
    rag_chain_mistral=rag_chain_mistral
)

print("Generated Answer:", result["answer"])
print("Document Sources:", result["doc_source"])
