## This will be a separate lambda which will be used to generate the vectors

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS

## This function will read the data and  split the documents into chunk
##@TODO: PyPDF may not give the best result. WE need to try calling llamaParser API and see if that works better
def data_splitter():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Generate the faiss vecotrs from the documents
def generate_faiss(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    return vectorstore_faiss

## save the generated faiss vectors in S3
def save_faiss_s3(s3_key,s3_bucket,faiss_embed):
    s3=boto3.client('s3')

    try:
        s3.put_object(Bucket = s3_bucket, Key = s3_key, Body =faiss_embed )
        print("faiss saved to s3")

    except Exception as e:
        print("Error when saving the code to s3")
