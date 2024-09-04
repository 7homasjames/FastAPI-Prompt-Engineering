import streamlit as st
import requests


def get_url(end_point):
    url = f"http://127.0.0.1:8000/{end_point}/"
    return url

# Backend URL
backend_url = get_url("process_samples")

st.title("AI Sample Processor")

st.write("This app allows you to process samples using different prompting techniques and view the results.")
no_of_samples_required = st.number_input("Enter the number of samples to process:", min_value=1, max_value=10, value=1)


if st.button("Process Samples"):
    with st.spinner("Processing..."):
        payload = {"no_of_samples_required": no_of_samples_required}
        
        # Send POST request to FastAPI backend
        response = requests.post(backend_url, json=payload)
        
        if response.status_code == 200:
            results = response.json()["samples"]
            # Display the results
            for idx, result in enumerate(results):

                st.subheader(f"Sample {idx + 1}")
                st.write("**Few Shot Samples:**")
                st.text(result['Few Shot Samples'])
                
                st.write("**Zero Shot Samples:**")
                st.text(result['Zero Shot Samples'])
                
                st.write("**Structured Samples:**")
                st.text(result['Structured Samples'])

                
                st.write("**Detailed Samples:**")
                st.text(result['Detailed Samples'])
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

    
        
