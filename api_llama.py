import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate, Document

import openai
from data import generated_data
from prompts_sample import few_shot_prompting, zero_shot_prompting, structured_prompting, detailed_prompting, claim_processing, policy_checking, policy_summarization
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError

# Load environment variables
_ = load_dotenv(find_dotenv())

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Azure Blob Storage account details
STORAGE_ACCOUNT_NAME = os.getenv('STORAGE_ACCOUNT_NAME')
STORAGE_ACCOUNT_KEY = os.getenv('STORAGE_ACCOUNT_KEY')
CONTAINER_NAME = os.getenv('CONTAINER_NAME')
FILE_NAME = os.getenv('FILE_NAME')
BLOB_NAME = 'text'
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
LOCAL_FILE_NAME = 'Sample'

# Initialize FastAPI app
app = FastAPI()


class SampleRequest(BaseModel):
    no_of_samples_required: int

class BlobJsonSample:
    def __init__(self):
        self.connection_string = AZURE_STORAGE_CONNECTION_STRING

    def upload_json(self, data):
        if self.connection_string is None:
            print("Missing required environment variable: AZURE_STORAGE_CONNECTION_STRING.")
            sys.exit(1)

        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        try:
            # Create the container if it doesn't exist
            container_client.create_container()
        except ResourceExistsError:
            print(f"Container '{CONTAINER_NAME}' already exists.")

        try:
            # Instantiate a BlobClient
            blob_client = container_client.get_blob_client(BLOB_NAME)

            # Upload the JSON data
            blob_client.upload_blob(json.dumps(data), blob_type="BlockBlob", overwrite=True)
            print(f"Uploaded data to {BLOB_NAME}")

        except Exception as ex:
            print(f"Error: {ex}")

    def download_json(self):
        if self.connection_string is None:
            print("Missing required environment variable: AZURE_STORAGE_CONNECTION_STRING.")
            sys.exit(1)

        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        try:
            blob_client = container_client.get_blob_client(BLOB_NAME)

            # Check if the blob exists
            if not blob_client.exists():
                raise FileNotFoundError(f"The blob '{BLOB_NAME}' does not exist.")

            # Download the JSON data
            download_stream = blob_client.download_blob()
            json_data = json.loads(download_stream.readall())
            print(json_data)

            with open(LOCAL_FILE_NAME, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
            print(f"Downloaded data to {LOCAL_FILE_NAME}")

        

        except FileNotFoundError as fnf_error:
            print(fnf_error)
        except Exception as ex:
            print(f"Error: {ex}")
        
        return json_data


def get_completion(prompt, model="gpt-3.5-turbo"):

    data = generated_data()
    documents = [Document(text= data)]
    index = VectorStoreIndex(documents)
    query_engine = index.as_query_engine()

    response = query_engine.query(prompt)
    #print(response)
    return response.response





# Function to process samples
def process_samples(no_of_samples_required: int):
    results = []
    for i in range(no_of_samples_required):
        few_shot_sample = get_completion(few_shot_prompting()) 
        zero_shot_sample = get_completion(zero_shot_prompting())
        structured_sample = get_completion(structured_prompting())
        detailed_sample = get_completion(detailed_prompting())
        
        sample_data = {
            "Few Shot Samples": few_shot_sample,
            "Zero Shot Samples": zero_shot_sample,
            "Structured Samples": structured_sample,
            "Detailed Samples": detailed_sample
        }
        
        results.append(sample_data)

        # Create a new dictionary with separated content and reference
        parsed_data = {}
        for key, value in sample_data.items():
            content, reference = parse_content_and_reference(value)
            parsed_data[key] = {
                "content": content,
                "reference": reference
            }

        print(parsed_data)
        
        # Append each result to the Azure Blob Storage
        sample = BlobJsonSample()
        sample.upload_json(parsed_data)
        sample.download_json()
    
    return {"samples": results}

def parse_content_and_reference(sample_text):
    # Split text at 'Reference:\n'
    parts = sample_text.split("\nReference:")
    content = parts[0].strip()
    reference = parts[1].strip() if len(parts) > 1 else ""
    return content, reference


@app.post("/process_samples/")
def process_samples_endpoint(request: SampleRequest):
    try:
        results = process_samples(request.no_of_samples_required)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read_blob_data/")
def read_blob_data():
    try:
        # Call the function to read data from the blob
        blob_data = sample.download_json()
        return {"blob_data": blob_data}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


