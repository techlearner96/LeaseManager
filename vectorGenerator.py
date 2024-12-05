## This will be a separate lambda which will be used to generate the vectors

import numpy as np
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.schema import Document
from io import BytesIO
import boto3
import os
import pytesseract
from PIL import Image
from langchain_community.embeddings import BedrockEmbeddings


bucket_name = 'capleasemanager'
prefix = 'lease/'
s3_faiss='faiss/'
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


#Get the documents
def get_documents_from_s3(s3_bucket, prefix):
    # @TODO  test with S3
    s3=boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name)
    files = response.get('Contents', [])

    text_list=[] 
    for file in files:
        file_key = file['Key']
        if file_key.endswith('.png'):
            # Download the file from S3
            obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            img_data = obj['Body'].read()

            # Open the image
            image = Image.open(BytesIO(img_data))

            # Perform OCR on the image
            text = pytesseract.image_to_string(image)

            # Append the extracted text to the list
            text_list.append(text)

    return text_list

    
    # # This code is for local testing only
    # folder_path = 'testdata/'
    # text_list = []

    # # Iterate over each file in the folder
    # for filename in os.listdir("testdata/"):
    #     if filename.endswith('.png'):
    #         # Open the image file
    #         img_path = os.path.join("testdata/", filename)
    #         img = Image.open(img_path)
            
    #         # Use pytesseract to extract text from the image
    #         text = pytesseract.image_to_string(img)
            
    #         # Append the extracted text to the list
    #         text_list.append(text)
    #         return text_list

    

# read the texts from the documents
def read_text_from_files(pngfiles):
    text_list=[]
    folder_path='testdata/'
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # Open the image file
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
        
            # Use pytesseract to extract text from the image
            text = pytesseract.image_to_string(img)
        
            # Append the extracted text to the list
            text_list.append(text)
            return text_list


## This function will read the data and  split the documents into chunk
def data_splitter(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    split_texts=[]
    for doc in documents:

        chunk=text_splitter.split_text(doc)
        split_texts.extend(chunk)
    return split_texts

   # Generate the FAISS vectors from the documents
def generate_faiss(split_docs):
    documents = [Document(page_content=text) for text in split_docs]
    vectorstore_faiss = FAISS.from_documents(
        split_docs,
        bedrock_embeddings  
    )
    return vectorstore_faiss

# Save the generated FAISS vectors in S3
def save_faiss_s3(faiss_index):
    s3 = boto3.client('s3')
    faiss_bytes = BytesIO()
    faiss_index.save_index(faiss_bytes)
    faiss_bytes.seek(0)

    try:
        s3.put_object(Bucket=bucket_name, Key=s3_faiss, Body=faiss_bytes.getvalue())
        print("FAISS index saved to S3")
    except Exception as e:
        print(f"Error when saving the FAISS index to S3: {e}")


def generate_vectors():
    list_text = get_documents_from_s3(bucket_name,prefix)
    
    # list_text=read_text_from_files(raw_png)
    splitted_text= data_splitter(list_text)
    vectorstore_faiss= generate_faiss(splitted_text)
    save_faiss_s3(vectorstore_faiss)
    
    

if __name__ == "__main__":
    generate_vectors()
    # # Example usage
    # bucket_name = 'capleasemanager'
    # prefix = 'lease/'
    # text_list = get_documents_from_s3(bucket_name, prefix)

    # # Print the list of extracted texts
    # for text in text_list:
    #     print(text)
    

#Commenting this function since I will not be using Lambda  anymore
###def lambda_handler(event, context):
    #read the documents
    #documents = read_documents_from_s3(s3_bucket)
    #split document
    #split_doc = data_splitter(documents)
    #generate faiss
    #faiss_index = generate_faiss(split_docs)
    #Store faiss in S3
    #save_faiss_s3(faiss_index)

   # return {
    #    'statusCode': 200,
     #   'body': json.dumps('Hello from Lambda!')
    #}#