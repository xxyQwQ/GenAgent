import os
import json
import yaml

from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    openai_config = config['openai']

OPENAI_BASE_URL = openai_config['base_url']
OPENAI_API_KEY = openai_config['api_key']
OPENAI_EMBEDDING_MODEL = openai_config['embedding_model']
OPENAI_COMPLETION_MODEL = openai_config['completion_model']


def retrieve_references(requirement, model='OpenAI', count=3):
    with open('./dataset/workflow/meta.json') as meta_file:
        metadata = json.load(meta_file)

    documents = []
    for name, path in metadata.items():
        with open(path['description'], 'r') as desc_file:
            desc = desc_file.read()
        meta = {
            'name': name,
            'description': path['description'],
            'workflow': path['workflow'],
            'code': path['code'],
            'markdown': path['markdown']
        }
        documents.append(Document(desc, metadata=meta))

    if model == 'OpenAI':
        embedding = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY
        )
    elif model == 'BERT':
        embedding = HuggingFaceEmbeddings(
            model_name='google-bert/bert-base-uncased'
        )
    elif model == 'MPNet':
        embedding = HuggingFaceEmbeddings(
            model_name='facebook/m2m100_418M'
        )
    elif model == 'BAAI':
        embedding = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en'
        )
    elif model == 'Qwen':
        embedding = HuggingFaceEmbeddings(
            model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct'
        )

    database = f'./dataset/workflow/db/{model}'
    if os.path.exists(database):
        vectors = Chroma(
            embedding_function=embedding,
            persist_directory=database
        )
    else:
        vectors = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=database
        )

    retriever = vectors.as_retriever(
        search_kwargs={"k": count}
    )

    references = retriever.invoke(requirement)
    return references


def invoke_completion(message):
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_COMPLETION_MODEL,
            messages=[{
                'role': 'user',
                'content': message
            }]
        )
        answer = response.choices[0].message
        usage = response.usage

    except Exception as error:
        answer = f'Error: {error}'
        usage = None

    return answer, usage
