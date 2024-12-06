## This file will contain the code realted with App's UI. It will need to deployed as application
### on EC2. Another option is to dockerize it and deploy it on ECS
import boto3
import streamlit as st
from botocore.exceptions import ClientError

bucket_name = 'capleasemanager'


##Only for local testing. We will using lambda to invoke the LLM via Bedrock
def main():
    st.set_page_config("Lease Manager")
    
    st.header("Lease Manager")

    user_question = st.text_input("Ask a Question Related With Leases")

    with st.sidebar:
        st.title("Add More Leases and update system:")
        uploaded_file = st.file_uploader("Choose lease pages to upload", type=["png", "jpeg"], accept_multiple_files=True)
        if st.button("Upload Lease Data"):
            with st.spinner("Processing..."):
                '''Storing the files to s3'''
                s3_client = boto3.client('s3')
                for file in uploaded_file:
                    try:
                        response = s3_client.upload_fileobj(file, bucket_name, file.name)
                    except ClientError as e:
                        raise Exception(f"error in uploading the file")
                '''
                  2. Call endpoint to generate vector from vectorGenerator.py
                  3. Here we need to update database but how?
                '''
                st.success("Done")


    if st.button("Output"):
        with st.spinner("Processing..."):
           #@TODO Add code to (1) invoke the getLeaseInfo API via APIGateway
            response = 'response of query' # call the api to get response for the query through getLeaseInfo
            st.success("Done")
        st.write(response)

if __name__ == "__main__":
    main()
