This RAG project aims to provide Lease Management services using Gen AI. Using this tool you will be able to :
1. Read the lease document provided in the png or jpg format
2. Ask Questions based on the provided lease documents
3. update database with vital information available in the lease document.
4. predict the lease prices over the years using the data available from previous years.


Code Structure (so far):
 We will be implementing the project on AWS and will use AWS sage maker and Bedrock for the prediction and invoking LLMs.
 The 2 lambda's created so far are :
 1. getLeaseInfo.py - this lambda will contain the code required to get the query from user, read the FAISS and then invoke the LLM and finally return the response to the user.
 2. vectorGenerator.py- this lambda will contain the code to read the document, split it and generate the FAISS vectors and store them in S3 bucket.
 3. leaseManagerUI.py - it will contain the streamlit based code which will be deployed on either ECS or on EC2 instance (TBD)
