from openai import OpenAI
import cohere
import voyageai
import os

openai_models = ["text-embedding-3-small-768", "text-embedding-3-small-1536", "text-embedding-3-large-1536", "text-embedding-3-large-3072"]
cohere_models = ["embed-english-v3.0", "embed-multilingual-v3.0", "embed-english-light-v3.0", "embed-multilingual-light-v3.0"]
voyage_models = ["voyage-large-2", "voyage-law-2", "voyage-code-2"]

dimensionality = {
    "text-embedding-3-small-768": 768,
    "text-embedding-3-small-1536": 1536,
    "text-embedding-3-large-1536": 1536,
    "text-embedding-3-large-3072": 3072,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "voyage-large-2": 1536,
    "voyage-law-2": 1024,
    "voyage-code-2": 1536,
}

def get_embeddings(text: str or list[str], model: str, input_type: str = ""):
    """
    - input_type: "query" or "document"
    """
    if type(text) == str:
        input_format = "string"
    elif type(text) == list:
        input_format = "list"
    else:
        raise Exception("text must be a string or a list of strings")

    if model in openai_models:
        client = OpenAI()
        model, dimensions = split_model_and_dimensionality(model)
        response = client.embeddings.create(input=text, model=model, dimensions=int(dimensions))
        embeddings = [embedding_item.embedding for embedding_item in response.data]
        if input_format == "string":
            embeddings = embeddings[0]
        return embeddings
    elif model in cohere_models:
        if input_type == "query":
            input_type = "search_query"
        elif input_type == "document":
            input_type = "search_document"
        else:
            raise Exception("input_type must be 'query' or 'document'")
        if input_format == "string":
            text = [text]
        cohere_api_key = os.environ['COHERE_API_KEY']
        co = cohere.Client(f'{cohere_api_key}')
        response=co.embed(texts=text, input_type=input_type, model=model)
        if input_format == "string":
            return response.embeddings[0]
        else:
            return response.embeddings
    elif model in voyage_models:
        if input_format == "string":
            response = voyageai.get_embedding(text, model=model, input_type=input_type)
        else:
            response = voyageai.get_embeddings(text, model=model, input_type=input_type)
        return response
    
def split_model_and_dimensionality(model: str):
    """
    - model: model name with dimensionality at the end (only used for OpenAI models)
    """
    parts = model.rsplit("-", 1)  # Split the model from the last hyphen
    model_name = parts[0]         # The part before the last hyphen
    dimensionality = parts[1]     # The part after the last hyphen
    return model_name, dimensionality

def test__split_model_and_dimensionality():
    model, dimensionality = split_model_and_dimensionality("text-embedding-3-small-768")
    assert model == "text-embedding-3-small"
    assert dimensionality == "768"

    model, dimensionality = split_model_and_dimensionality("text-embedding-3-large-3072")
    assert model == "text-embedding-3-large"
    assert dimensionality == "3072"

def test__get_embeddings_openai():
    input_text = "Hello, world!"
    input_type = "query"
    model = "text-embedding-3-small-768"
    embedding = get_embeddings(input_text, model, input_type)
    assert len(embedding) == 768

def test__get_embeddings_cohere():
    input_text = "Hello, world!"
    input_type = "query"
    model = "embed-english-v3.0"
    embedding = get_embeddings(input_text, model, input_type)
    assert len(embedding) == 1024

def test__get_embeddings_voyage():
    input_text = "Hello, world!"
    input_type = "query"
    model = "voyage-large-2"
    embedding = get_embeddings(input_text, model, input_type)
    assert len(embedding) == 1536

#test__split_model_and_dimensionality()
#test__get_embeddings_openai()
#test__get_embeddings_cohere()
#test__get_embeddings_voyage()