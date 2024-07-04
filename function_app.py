import azure.functions as func
import logging
import json
import asyncio
from openai import AsyncAzureOpenAI
import openai
import os

logging.basicConfig(level=logging.INFO)
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

async def vectorise_record(record: dict, tries_left=3) -> dict:
    """Vectorise the text in the record using the Azure OpenAI API.
    
    Args:
        record (dict): The record to vectorise.
        
    Returns:
        dict: The vectorised record."""
    
    compass_api_key = os.environ["COMPASS_API_KEY"]
    compass_endpoint = os.environ["COMPASS_ENDPOINT"]
    compass_api_version = os.environ["COMPASS_API_VERSION"]
    compass_embedding_model = os.environ["COMPASS_EMBEDDING_MODEL"]
    
    try:
        async with AsyncAzureOpenAI(
            api_key=compass_api_key,
            azure_endpoint=compass_endpoint,
            api_version=compass_api_version,
        ) as open_ai_client:
            embeddings = await open_ai_client.embeddings.create(
                model=compass_embedding_model, input=list(record["data"].values())
            )

            logging.debug("Embeddings: %s", embeddings)

    except openai.RateLimitError as e:
        logging.error("OpenAI Rate Limit Error: %s", e)
         
        if tries_left > 0:
            logging.info("Retrying vectorisation of record with %s tries left.", tries_left)
            remaining_tries = tries_left - 1
            backoff = 15 ** (3 - remaining_tries)
            await asyncio.sleep(backoff)
            return await vectorise_record(record, tries_left=remaining_tries)
        else:
            logging.info("Failed to vectorise record after 3 retries.")
            return {"recordId": record["recordId"], "data": {}, "errors": [{"message": "Failed to vectorise input records with CompassAPI. Check function app logs for more details of exact failure."}], "warnings": None}
        
    except (openai.OpenAIError, openai.APIConnectionError) as e:
        logging.error("OpenAI Error: %s", e)
        return {"recordId": record["recordId"], "data": {}, "errors": [{"message": "Failed to vectorise input records with CompassAPI. Check function app logs for more details of exact failure."}], "warnings": None}
    else:
        vectorised_record = {"recordId": record["recordId"], "data": {}, "errors": None, "warnings": None}
        for index, key in enumerate(record["data"].keys()):
            vectorised_record["data"][f"{key}_vector"] = embeddings.data[index].embedding

        return vectorised_record

@app.function_name(name="ai_search_2_compass")
@app.route(route="ai_search_2_compass", methods=[func.HttpMethod.POST])
async def ai_search_2_compass(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger for AI Search 2 Compass function.
    
    Args:
        req (func.HttpRequest): The HTTP request object.
        
    Returns:
        func.HttpResponse: The HTTP response object."""

    logging.info("Python HTTP trigger function processed a request.")

    try:
        req_body = req.get_json()
        values = req_body.get("values")
    except ValueError:
        return func.HttpResponse(
            "Please valid Custom Skill Payload in the request body", status_code=400
        )
    else:
        logging.debug("Input Values: %s", values)

        record_tasks = []

        for value in values:
            record_tasks.append(
                asyncio.create_task(vectorise_record(value))
            )

        results = await asyncio.gather(*record_tasks)
        logging.debug("Results: %s", results)
        vectorised_tasks = {"values": results}

        return func.HttpResponse(json.dumps(vectorised_tasks), status_code=200, mimetype="application/json")
