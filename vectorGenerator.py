## This file will contain the code to read the pdf and generate the vectors
## for the generating vector we will be using FAISS and we will store the FAISS files in S3 on AWS

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS


bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


## This function will read the data and  split the documents into chunk
##@TODO: PyPDF may not give the best result. WE need to try calling llamaParser API and see if that works better
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    return vectorstore_faiss

def save_faiss_s3(s3_key,s3_bucket,faiss_embed):
    s3=boto3.client('s3')

    try:
        s3.put_object(Bucket = s3_bucket, Key = s3_key, Body =faiss_embed )
        print("faiss saved to s3")

    except Exception as e:
        print("Error when saving the code to s3")


## there may be multiple faiss files int the S3 bucket. We want the lastest ones
def read_faiss_s3(s3_key,s3_bucket):

    # List objects in the specified S3 bucket
    response = s3.list_objects_v2(Bucket=bucket_name)
    files = response.get('Contents', [])
    # Sort files by the LastModified attribute
    files.sort(key=lambda x: x['LastModified'], reverse=True)
    # Get the latest file
    latest_file = files[0]['Key']
    return latest_file