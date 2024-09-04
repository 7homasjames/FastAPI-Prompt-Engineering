import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv, find_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.contents.chat_history import ChatHistory
#from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, AppendBlobService
#from azure.storage.blob import  AppendBlobService
import openai
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

# Function to get completion using Semantic Kernel
async def get_completion_skernel(prompt, model="gpt-3.5-turbo"):
    kernel = Kernel()

    
    openai_service = OpenAIChatCompletion(
        api_key= os.getenv('OPENAI_API_KEY'),
        ai_model_id=model
    )
    kernel.add_service(openai_service)


    chat_execution_settings = OpenAIChatPromptExecutionSettings(
        ai_model_id=model,
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9
    )
    chat_completion_service = kernel.get_service(type=ChatCompletionClientBase)
    chat_history = ChatHistory()

    chat_prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="grounded_response",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="query_term", description="The user input", is_required=True),
        ],
        execution_settings=chat_execution_settings,
    )

    chat_function = kernel.add_function(
        function_name="ChatGPTFunc",
        plugin_name="chatGPTPlugin",
        prompt_template_config=chat_prompt_template_config
    )
    arguments = KernelArguments(query_term=prompt)

    # Invoke the function with the prompt
    result = await kernel.invoke(chat_function, arguments)
    return result.value[0].to_dict()["content"]

async def main(prompt):
    result = await get_completion_skernel(prompt)
    return result

# Function to process samples
async def process_samples(no_of_samples_required: int):
    results = []
    for i in range(no_of_samples_required):
        few_shot_sample = await main(few_shot_prompting()) 
        zero_shot_sample = await main(zero_shot_prompting())
        structured_sample = await main(structured_prompting())
        detailed_sample = await main(detailed_prompting())
        
        sample_data = {
            "Few Shot Samples": few_shot_sample,
            "Zero Shot Samples": zero_shot_sample,
            "Structured Samples": structured_sample,
            "Detailed Samples": detailed_sample
        }
        
        results.append(sample_data)

        
        
        # Append each result to the Azure Blob Storage
        sample = BlobJsonSample()
        sample.upload_json(sample_data)
        sample.download_json()
    
    return {"samples": results}

@app.post("/process_samples/")
async def process_samples_endpoint(request: SampleRequest):
    try:
        results = await process_samples(request.no_of_samples_required)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read_blob_data/")
async def read_blob_data():
    try:
        # Call the function to read data from the blob
        blob_data = sample.download_json()
        return {"blob_data": blob_data}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


