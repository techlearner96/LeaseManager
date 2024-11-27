## This file will contain the code realted with App's UI. It will need to deployed as application
### on EC2. Another option is to dockerize it and deploy it on ECS


import streamlit as st

##Only for local testing. We will using lambda to invoke the LLM via Bedrock
def main():
    st.set_page_config("Lease Manager")
    
    st.header("Lease Manager")

    user_question = st.text_input("Ask a Question Related With Leases")

    with st.sidebar:
        st.title("Add More Leases and update system:")
        
        ## this button will not allow you to add the files rather it will update the vector store
        ## @TODO: Perhaps we should provide a button to add files into the S3 bucket here.
        if st.button("Update DB"):
            with st.spinner("Processing..."):
                #@TODO Add the code to invoke the vectorGenerator API via API Gateway
                st.success("Done")


    if st.button("Output"):
        with st.spinner("Processing..."):
           #@TODO Add code to (1) invoke the getLeaseInfo API via APIGateway

            st.success("Done")

if __name__ == "__main__":
    main()
