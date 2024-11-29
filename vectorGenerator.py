## This will be a separate lambda which will be used to generate the vectors

import numpy as np
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from io import BytesIO
import boto3

s3_key="faiss_index"
s3_bucket="capleasemanager/lease"

#Read the documents
def read_documents_from_s3(bucket_name):
    # List objects in the bucket
    s3=boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name)
    documents = []
    
    for obj in response.get('Contents', []):
        # Read each document
        file_obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
        document = file_obj['Body'].read().decode('utf-8')
        documents.append(document)
    
    return documents

## This function will read the data and  split the documents into chunk
def data_splitter(documents):
    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    split_docs=text_splitter.split_documents(documents)
    return split_docs

   # Generate the FAISS vectors from the documents
def generate_faiss(split_docs):
    vectorstore_faiss = FAISS.from_documents(
        split_docs,
        bedrock_embeddings  # Ensure bedrock_embeddings is defined
    )
    return vectorstore_faiss

# Save the generated FAISS vectors in S3
def save_faiss_s3(faiss_index):
    s3 = boto3.client('s3')
    faiss_bytes = BytesIO()
    faiss_index.save_index(faiss_bytes)
    faiss_bytes.seek(0)

    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=faiss_bytes.getvalue())
        print("FAISS index saved to S3")
    except Exception as e:
        print(f"Error when saving the FAISS index to S3: {e}")


def lambda_handler(event, context):
    #read the documents
    documents = read_documents_from_s3(s3_bucket)
    #split document
    split_doc = data_splitter(documents)
    #generate faiss
    faiss_index = generate_faiss(split_docs)
    #Store faiss in S3
    save_faiss_s3(faiss_index)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }