
import os
import subprocess
import logging

import torch
import json
import numpy as np
import time
import transformers
import locale
import nltk

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

#import pdf2image
import pdfplumber
from botocore.exceptions import ClientError
from io import BytesIO
#from PyPDF2 import PdfReader

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

import boto3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the OpenAI API key from an environment variable
#openai.api_key = os.getenv("OPENAI_API_KEY")

def model_fn(model_dir):
    """
    Load any necessary configurations or resources.
    """
    #openai.api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment variable
    return None  # No specific model to load as we're using an external API

def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request body.
    """
    
    if request_content_type == "application/json":
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Call OpenAI API based on the input prompt and return the response.
    """

    # Parse the input payload
    input_data = json.loads(event["body"])
    file_name = input_data.get("file_name", "Provide a file name")
    
    file_key = "pdf_uploads/" + file_name   
    logger.info("pdf we got is : %s", file_key)
    
    base_filename = os.path.splitext(input_data)[0]
    logger.info("file we got is : %s", base_filename)

    #Our core code of summarization lands here
    s3_client = boto3.client('s3')
    pdf_path = '/tmp/input.pdf'
    s3_client.download_file(bucket_name, file_key, pdf_path)
        
    # Variable to accumulate extracted text
    text = ""

    # Open the PDF using PyMuPDF (fitz)
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc[page_num]

        # Try to extract actual text from the page
        page_text = page.get_text("text")

        if page_text.strip():
            text += page_text
        else:
            # Render the page as an image
            pix = page.get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes()))

            # Use OCR on the page image to extract text
            ocr_text = pytesseract.image_to_string(image)

            # Accumulate extracted text if OCR detected any text
            if ocr_text.strip():  # Only add if text was found
                #extracted_text += f"\n--- Page {page_num + 1} ---\n"
                text += ocr_text

    text = text.replace('\t', ' ')

    # Close the document
    doc.close()
        
    # Combine the pages, and replace the tabs with spaces

    openAIApiKey = "skproj_d8BxU8_XoEjNa7XSa7MYfyWSPwLsMsTDsU9BAoI5B6NpJxv1U9WOfdVVxDVfzPm1uIWvdOVT3BlbkFJhVc6z6W6TNC_Sa4TeniWGFRVyv4a327s71gTCaeH7HSIRk6wROaD0W0g-qjGcFrLJDS9ACmZgA"

    llm = OpenAI(temperature=0, openai_api_key=openAIApiKey)

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=1000, chunk_overlap=500)
    splits = text_splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(openai_api_key=openAIApiKey)

    vectors = np.array(embeddings.embed_documents([x.page_content for x in splits]))

    # Assuming 'embeddings' is a list or array of 1536-dimensional embeddings
    # Choose the number of clusters, this can be adjusted based on the doc's content.
    # Usually if you have 10 passages from a doc you can tell what it's about
    num_clusters = 5

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

    selected_docs = [splits[doc] for doc in selected_indices]

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

    bucket_name = 'capstoneragmodel'
    directory = "summaries"
    file_key = directory + "/" + base_filename + '.txt'

    logger.info("file to be created is %s:", file_key)

    # Initialize the S3 client
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=output_text)

    # Extract the text result from the OpenAI API response
    #output_text = response["choices"][0]["text"].strip()

    return ("Success Summarization")  # Extract response text

def output_fn(prediction, content_type):
    """
    Serialize the prediction result back to JSON.
    """
    
    if content_type == "application/json":
        return json.dumps({"generated_text": prediction})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
def handler(event, context):
    """
    This function receives input from a SageMaker endpoint request,
    forwards it to the OpenAI API, and returns the response.
    """
    try:
        # Parse the input payload
        input_data = json.loads(event["body"])
        file_name = input_data.get("file_name", "Provide a file name")
        file_key = "pdf_uploads/" + file_name
        #logger.info("Got input event : %s", event)
        logger.info("pdf we got is : %s", file_key)
        base_filename = os.path.splitext(input_data)[0]
        logger.info("file we got is : %s", base_filename)

        #Our core code of summarization lands here
        s3_client = boto3.client('s3')
        pdf_path = '/tmp/input.pdf'
        s3_client.download_file(bucket_name, file_key, pdf_path)
        
        # Variable to accumulate extracted text
        text = ""

        # Open the PDF using PyMuPDF (fitz)
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Try to extract actual text from the page
            page_text = page.get_text("text")

            if page_text.strip():
                text += page_text
            else:
                # Render the page as an image
                pix = page.get_pixmap()
                image = Image.open(io.BytesIO(pix.tobytes()))

                # Use OCR on the page image to extract text
                ocr_text = pytesseract.image_to_string(image)

                # Accumulate extracted text if OCR detected any text
                if ocr_text.strip():  # Only add if text was found
                    #extracted_text += f"\n--- Page {page_num + 1} ---\n"
                    text += ocr_text

        text = text.replace('\t', ' ')

        # Close the document
        doc.close()
        
        # Combine the pages, and replace the tabs with spaces

        openAIApiKey = "skproj_d8BxU8_XoEjNa7XSa7MYfyWSPwLsMsTDsU9BAoI5B6NpJxv1U9WOfdVVxDVfzPm1uIWvdOVT3BlbkFJhVc6z6W6TNC_Sa4TeniWGFRVyv4a327s71gTCaeH7HSIRk6wROaD0W0g-qjGcFrLJDS9ACmZgA"

        %env provider_API_key=openAIApiKey

        llm = OpenAI(temperature=0, openai_api_key=openAIApiKey)

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=1000, chunk_overlap=500)
        splits = text_splitter.create_documents([text])

        embeddings = OpenAIEmbeddings(openai_api_key=openAIApiKey)

        vectors = np.array(embeddings.embed_documents([x.page_content for x in splits]))

        # Assuming 'embeddings' is a list or array of 1536-dimensional embeddings
        # Choose the number of clusters, this can be adjusted based on the doc's content.
        # Usually if you have 10 passages from a doc you can tell what it's about
        num_clusters = 5

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

        selected_docs = [splits[doc] for doc in selected_indices]

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
                             prompt=reducePrompt,
        output_text = reduce_chain.run([summaries])                            )
        
        End of core code
        
        output_text = "I got the file " + input_data
        logger.info("Returning output: %s", output_text)

        bucket_name = 'capstoneragmodel'
        directory = "summaries"
        file_key = directory + "/" + base_filename + '.txt'

        logger.info("file to be created is %s:", file_key)

        # Initialize the S3 client
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=output_text)

        # Extract the text result from the OpenAI API response
        #output_text = response["choices"][0]["text"].strip()

        # Return the output as JSON
        return {
            "statusCode": 200,
            "body": json.dumps({"result": output_text})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
'''
