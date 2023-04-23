# embedder.py
import openai
import json
import spacy


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


# nlp = spacy.load("en_core_web_md")
OPENAI_API_KEY = open_file("openai_api_key.txt")


def embed_data(text, api_key=OPENAI_API_KEY):
    # doc = nlp(text)
    # return doc.vector
    delay = 60  # Initial delay in seconds
    backoff_factor = 2  # Multiply the delay by this factor for each retry

    openai.api_key = api_key
    try:
        response = openai.Embedding.create(
            engine="text-embedding-ada-002",
            input=text
        )
        embedding = response['data'][0]['embedding']
        # embedding = json.loads(embedding_text)
        return embedding
    except openai.error.RateLimitError as e:
        print(
            f"API call failed with rate limit error: {e}. Retrying after {delay} seconds...")
        time.sleep(delay)
        delay *= backoff_factor
