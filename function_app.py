import azure.functions as func
import logging
import json
import asyncio
from openai import AsyncAzureOpenAI
import os

logging.basicConfig(level=logging.DEBUG)
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

async def vectorise_record(record: dict) -> dict:
    """Vectorise the text in the record using the Azure OpenAI API."""

    api_key = os.environ["API_KEY"]
    async with AsyncAzureOpenAI(
        # This is the default and can be omitted
        api_key=api_key,
        azure_endpoint="https://api.core42.ai/",
        api_version="2023-03-15-preview",
    ) as open_ai_client:

        embeddings = await open_ai_client.embeddings.create(
            model="text-embedding-ada-002", input=list(record["data"].values())
        )

        logging.debug(f"Embeddings: {embeddings}")

    vectorised_record = {"recordId": record["recordId"], "data": {}, "errors": None, "warnings": None}
    for index, key in enumerate(record["data"]):

        vectorised_record["data"][f"{key}_vector"] = embeddings.data[index].embedding

    return vectorised_record

@app.route(route="ai_search_2_compass", methods=[func.HttpMethod.POST])
async def ai_search_2_compass(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger for AI Search 2 Compass function."""

    logging.info("Python HTTP trigger function processed a request.")

    try:
        req_body = req.get_json()
        values = req_body.get("values")
    except ValueError:
        return func.HttpResponse(
            "Please valid Custom Skill Payload in the request body", status_code=400
        )
    else:
        logging.debug(f"Input Values: {values}")

        record_tasks = []

        for value in values:
            record_tasks.append(
                asyncio.create_task(vectorise_record(value))
            )

        results = await asyncio.gather(*record_tasks)
        logging.debug(f"Results: {results}")
        vectorised_tasks = {"values": results}

        return func.HttpResponse(json.dumps(vectorised_tasks), status_code=200, mimetype="application/json")
