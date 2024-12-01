
import os
import subprocess
import logging

import torch
import json
import numpy as np
import time
import transformers
#import locale
#import nltk
from dotenv import load_dotenv
#from pprint import pprint

#nltk.download('punkt')

#from transformers import AutoTokenizer , AutoModelForCausalLM
from transformers import pipeline, BitsAndBytesConfig

# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain_openai import ChatOpenAI

# Embedding Support
from langchain_openai import OpenAIEmbeddings

# Data Science
import numpy as np
from sklearn.cluster import KMeans
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_text_splitters import CharacterTextSplitter
#from langchain_core.output_parsers import StrOutputParser
#from langchain import hub
#from langchain_community.document_loaders import UnstructuredHTMLLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain.chains.summarize import load_summarize_chain
#from llama_index.llms.mistralai import MistralAI
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
#from langchain_community.vectorstores import Chroma
#from langchain.document_loaders import PyMuPDFLoader

import io
#import pdf2image
import pdfplumber
from botocore.exceptions import ClientError
from io import BytesIO
#from PyPDF2 import PdfReader

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import json
import boto3
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the OpenAI API key from an environment variable
#openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the OpenAI API key from an environment variable
#openai.api_key = os.getenv("OPENAI_API_KEY")

def model_fn(model_dir):
    """
    Load any necessary configurations or resources.
    """
    logger.info("model_fn called %s", model_dir)
    print("Sreeram: model_fn called/exit")
    #openai.api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment variable
    return None  # No specific model to load as we're using an external API

def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request body.
    """
    
    logger.info("input_fn called %s", request_body)
    print("Sreeram: input_fn called 9:40am")

    if request_content_type == "application/json":
        print("Sreeram: input_fn json loads")
        return json.loads(request_body)
    else:
        print("Sreeram: input_fn cerror")
        raise ValueError(f"Unsupported content type: {request_content_type}")

def format_text(page, page_num, text):
    """
    Extract text with its formatting information from PDF.
    Returns list of dictionaries containing text and its formatting properties.
    """
    position = 0
    formatted_blocks = []
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
                    #print(f"span is {span} text is {text}")

    return formatted_blocks

def analyze_document_formatting(blocks):
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

def identify_potential_headers(blocks, format_stats):
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
    #print(f"Sreeram potential headers: {potential_headers}")
    return potential_headers

def confirm_headers_with_gpt(potential_headers, client):
    """
    Use GPT to confirm which potential headers are actual privacy policy section headers.

    Args:
        potential_headers (List[Dict]): List of dictionaries containing potential headers
        api_key (str): Your OpenAI API key

    Returns:
        List[Dict]: Confirmed headers with their text and positions
    """
    #client = OpenAI(api_key=api_key)
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
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying section headers in privacy policies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0
            )

            answer = response.choices[0].message.content.strip().lower()

            if answer == 'yes':
                confirmed_headers.append({
                    "header": header["text"],
                    "position": header["position"]
                })

        except Exception as e:
            print(f"Error in GPT confirmation: {e}")
            continue
    #print(f"Sreeram confirmed headers: {confirmed_headers}")
    return confirmed_headers

def chunk_policy_by_headers(headers, blocks):
    """
    Chunk the policy text based on section headers using the original formatted blocks.
    """
    chunks = []
    full_text = ""
    current_position = 0

    for block in blocks:
        full_text += block["text"] + "\n"
    #print(f"chunk: {full_text}")

    for i, header_data in enumerate(headers):
        start_pos = header_data["position"]
        end_pos = headers[i + 1]["position"] if i + 1 < len(headers) else len(full_text)
        chunk_text = full_text[start_pos:end_pos].strip()
        #print(f"chunk by header {i}: {header_data}")
        chunks.append({
            #"header": header_data["header"],
            "header": header_data["text"],
            "content": chunk_text
        })
    #print(f"Sreeram chunks: {chunks}")
    return chunks

def further_chunk_policy(policy, chunk_size: int = 1000, chunk_overlap: int = 500):    
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
                        "header": chunk["header"],
                        "chunk_index": i,
                        "total_sub_chunks": len(sub_chunks),
                    }
                })
        except Exception as e:
            print(f"Error processing chunk for company {company_name} with header '{chunk['header']}': {str(e)}")
            continue
    #print(f"Sreeram refined chunks: {refined_chunks}")
    return refined_chunks


def process_policy(formatted_blocks, llm):
    """
    Process a single PDF policy document and extract sections.
    """
    try:
        #formatted_blocks = extract_formatted_text(pdf_path)
        #if not formatted_blocks:
        #    return {"status": "error", "pdf_path": pdf_path, "message": "Failed to extract formatted text"}
        #print("Sreeram calling analyze")
        format_stats = analyze_document_formatting(formatted_blocks)
        #print("Sreeram calling identify")
        potential_headers = identify_potential_headers(formatted_blocks, format_stats)
        #print("Sreeram calling confirm with gpt")
        #confirmed_headers = confirm_headers_with_gpt(potential_headers, llm)

        #if not confirmed_headers:
            #return {"status": "error", "message": "No headers found in the document"}

        #print("Sreeram chunking by headers")
        #chunks = chunk_policy_by_headers(confirmed_headers, formatted_blocks)
        chunks = chunk_policy_by_headers(potential_headers, formatted_blocks)

        return {
            "status": "success",
            "total_chunks": len(chunks),
            "chunks": chunks,
            "message": "Successful in chunking"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def prep_docs(chunks):
    """Prepare documents for vector store with consistent metadata handling."""
    documents = []

    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            chunk = Document(page_content=chunk['page_content'], metadata=chunk['metadata'])

        metadata = {
            "chunk_number": idx + 1,
            "header": chunk.metadata.get("header", ""),
        }

        documents.append(Document(
            page_content=chunk.page_content,
            metadata=metadata
        ))
        #print(f"[DEBUG] Prepared document {idx + 1} with metadata: {metadata}")

    return documents

def split_text_to_docs(pdf_path, llm):
    # Variable to accumulate extracted text
    formatted_blocks = []
    docs = ""
    refined_chunks = []

    # Open the PDF using PyMuPDF (fitz)
    doc = fitz.open(pdf_path)
    #for page_num in range(doc.page_count):
    for page_num, page in enumerate(doc):
        page = doc[page_num]

        # Try to extract actual text from the page
        page_text = page.get_text("text")

        if page_text.strip():
            #text += page_text
            formatted_blocks.extend(format_text(page, page_num, page_text))
            #print(f"PDF: {page_text}")
        else:
            # Render the page as an image
            pix = page.get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes()))

            # Use OCR on the page image to extract text
            ocr_text = pytesseract.image_to_string(image)
            print(f"OCR: {ocr_text}")
            # Accumulate extracted text if OCR detected any text
            if ocr_text.strip():  # Only add if text was found
                #extracted_text += f"\n--- Page {page_num + 1} ---\n"
                #text += ocr_text
                formatted_blocks.extend(format_text(page, page_num, ocr_text))

    #text = text.replace('\t', ' ')
    #print(f"Sreeram formatted blocks: {formatted_blocks}")
    if(formatted_blocks) :
        result = process_policy(formatted_blocks, llm)

        if (result.get("status")) == "success":
            refined_chunks = further_chunk_policy(result)
            docs = prep_docs(refined_chunks)
        else:
            print(f"Sreeram process_policy conked {result.get('message')}")

    # Close the document
    doc.close()
        
    return docs

def debug_this(docs):
    print("Inside debug this")
    for doc in docs:
        print(f"doc is: {doc}")
        #print(f"Header: {doc.metadata.get('header', '')}")
        #print(f"content: {doc['page_content']}")


def predict_fn(input_data, model):
    """
    Call OpenAI API based on the input prompt and return the response.
    """

    # Parse the input payload
    #input_data = json.loads(event["body"])
    print("Sreeram: predict input data we got is: ")
    print(input_data)
    file_name = input_data.get("file_name", None)
    if file_name is None:
        print("Sreeram: JSON payload is not good")
        raise ValueError("Input JSON payload missing 'text' field")

    
    print("Sreeram: predict filename we got is "+ file_name)
    file_key = "pdf_uploads/" + file_name   
    print("Sreeram: predict pdf we got is "+file_key)
    logger.info("pdf we got is : %s", file_key)
    
    base_filename = os.path.splitext(file_name)[0]
    logger.info("file we got is : %s", base_filename)
    print("Sreeram: file we got is : "+ base_filename)

    #Our core code of summarization lands here
    s3_client = boto3.client('s3')
    pdf_path = '/tmp/input.pdf'
    bucket_name = 'capstoneragmodel'
    s3_client.download_file(bucket_name, file_key, pdf_path)
    print("Sreeram bucket: " + bucket_name + " filekey: "+file_key+ " pdf_path: "+pdf_path)
        
    openAIApiKey = os.environ.get("OPENAI_API_KEY")
    if not openAIApiKey:
        raise ValueError("API key not found in environment variables.")

    print(f"Your API key is: {openAIApiKey}")
    logger.info(f"Your API key is: {openAIApiKey}")

    #openAIApiKey = "skproj_d8BxU8_XoEjNa7XSa7MYfyWSPwLsMsTDsU9BAoI5B6NpJxv1U9WOfdVVxDVfzPm1uIWvdOVT3BlbkFJhVc6z6W6TNC_Sa4TeniWGFRVyv4a327s71gTCaeH7HSIRk6wROaD0W0g-qjGcFrLJDS9ACmZgA"

    llm = OpenAI(api_key=openAIApiKey)

    # Combine the pages, and replace the tabs with spaces

    docs = split_text_to_docs(pdf_path, llm)

    #text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=1000, chunk_overlap=500)
    #splits = text_splitter.create_documents([text])
    #return docs
    embeddings = OpenAIEmbeddings(openai_api_key=openAIApiKey)

    vectors = np.array(embeddings.embed_documents([x.page_content for x in docs]))

    # Assuming 'embeddings' is a list or array of 1536-dimensional embeddings
    # Choose the number of clusters, this can be adjusted based on the doc's content.
    # Usually if you have 10 passages from a doc you can tell what it's about
    num_clusters = min(5, len(vectors))
    print(f"Your is 6: {num_clusters}")
    logger.info(f"Your is 6: {num_clusters}")

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    # Find the closest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):

        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)

    llm3 = ChatOpenAI(temperature=0,
                openai_api_key=openAIApiKey,
                max_tokens=1000,
                model='gpt-3.5-turbo'
            )

    mapPrompt = PromptTemplate(
        template = """Write a concise summary of the following. The summary should be a list of bullet points. The summary cannot be more than 5 bullet points. The text is:
                    {text}
            CONCISE SUMMARY:""",
            input_variables=["text"]
        )

    reducePrompt = PromptTemplate(
        template= """The following is set of bullet-point summaries:
            {text}
            Take these and distill it into a final, consolidated summary of the main themes. The final summary should be at most 15 sentences. Remove the bullet points that are not relevant to the whole text.
            Helpful Answer: """,
            input_variables=['text']
        )

    map_chain = load_summarize_chain(llm=llm3,
                            chain_type="stuff",
                            prompt=mapPrompt)

    selected_docs = [docs[doc] for doc in selected_indices]

    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):

        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])

        # Append that summary to your list
        summary_list.append(chunk_summary)

    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)

    llm4 = ChatOpenAI(temperature=0,
                openai_api_key=openAIApiKey,
                max_tokens=3000,
                model='gpt-4',
                request_timeout=120
            )

    reduce_chain = load_summarize_chain(llm=llm4,
                            chain_type="stuff",
                            prompt=reducePrompt)
    output_text = reduce_chain.run([summaries])


    #output_text = "I got the file " + file_name
    logger.info("Writing output: %s", output_text)

    directory = "summaries"
    file_key = directory + "/" + base_filename + '.txt'

    logger.info("file to be created is %s:", file_key)

    # Initialize the S3 client
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=output_text)

    print("Sreeram: predict exit")
    print(output_text)
    return ("Success Summarization")  # Extract response text

def output_fn(prediction, content_type):
    """
    Serialize the prediction result back to JSON.
    """
    
    print("Sreeram: output called")
    logger.info("output_fn called %s", prediction)
    if content_type == "application/json":
        return json.dumps({"generated_text": prediction})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    print("Sreeram: output exit")

