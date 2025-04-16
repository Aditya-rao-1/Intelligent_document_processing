!pip install -U boto3 langchain langchain-pinecone langchain-community nltk sentence-transformers -U langchain-huggingface

# Imports
import boto3
import os
import time
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone as PineconeClient 
from langchain import PromptTemplate

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''

# AWS Textract client
client = boto3.client(
    'textract',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name='us-east-1'
)

# Start document text detection (multi-page PDF support)
response = client.start_document_text_detection(
    DocumentLocation={
        'S3Object': {
            'Bucket': '',
            'Name': 'CyberCrime.pdf'
        }
    }
)

job_id = response["JobId"]
print(f"Job started with Job ID: {job_id}")

# Polling for job completion
while True:
    result = client.get_document_text_detection(JobId=job_id)
    status = result["JobStatus"]

    if status in ["SUCCEEDED", "FAILED"]:
        break

    print("Processing...")
    time.sleep(5)

if status == "FAILED":
    raise Exception("Textract job failed!")

print("Processing completed!")

# Extract text from response
extracted_text = []

while True:
    if "Blocks" in result:
        for block in result["Blocks"]:
            if block["BlockType"] == "LINE" and "Text" in block:
                extracted_text.append(block["Text"])

    if "NextToken" in result:
        result = client.get_document_text_detection(JobId=job_id, NextToken=result["NextToken"])
    else:
        break

# Combine extracted text into a single string
full_text = "\n".join(extracted_text)
# Save extracted text to a file
output_file_name = "demo_rag_on_image.txt"
with open(output_file_name, "w") as output_file_io:
    output_file_io.write(full_text)

# NLP Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Stopword removal
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

preprocessed_text = preprocess_text(full_text)

# Prepare document for embedding
docs = [Document(page_content=preprocessed_text)]

# Split document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=250, separator="\n")
split_docs = text_splitter.split_documents(docs)

# Use multi-qa-mpnet-base-cos-v1 embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = 'pcsk_Ctqom_MAJhXtKwybY6xfHyQSQhgQEXpVgwHSkKj4TaBxLjKQE8VM22dJ3HDCRoa5PcJ5C'
index_name = "textract"

docsearch = PineconeVectorStore.from_documents(split_docs, embedding_model, index_name=index_name)

print("Processing complete!")
# Conversation history
chat_history = []

# Run conversation loop
while True:
    human_input = input("\nAsk a question (or type 'exit' to quit): ")
    if human_input.lower() == 'exit':
        break

    query_embedding = embedding_model.embed_query(human_input)
    search_results = docsearch.similarity_search(human_input, k=5)

    # Create context from retrieved documents
    MAX_CONTEXT_LENGTH = 6000
    context_string = '\n\n'.join(
        [f'Document {ind+1}: ' + i.page_content[:MAX_CONTEXT_LENGTH] for ind, i in enumerate(search_results)]
    )

    # Build conversation history
    formatted_history = ""
    for turn in chat_history:
        formatted_history += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    # Prompt template
    RAG_PROMPT_TEMPLATE = '''
You are a helpful and knowledgeable AI assistant having a conversation with a user.
Use the context and conversation history to answer the question.

Context:
{context}
You are a helpful and knowledgeable AI assistant. Use the provided context to answer the question.

If the context is insufficient, rely on your own knowledge to provide the best possible response.

Conversation History:
{history}

Question: {human_input}

Answer:
'''
    PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    prompt_data = PROMPT.format(
        human_input=human_input,
        context=context_string,
        history=formatted_history
    )

    # Bedrock model
    boto3_bedrock = boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    body_part = json.dumps({
        'inputText': prompt_data,
        'textGenerationConfig': {
            'maxTokenCount': 8192,
            'stopSequences': [],
            'temperature': 0.7,
            'topP': 1
        }
    })

    response = boto3_bedrock.invoke_model(
        body=body_part,
        contentType="application/json",
        accept="application/json",
        modelId='amazon.titan-text-express-v1'
    )

    output_text = json.loads(response['body'].read())['results'][0]['outputText']
    output_text = output_text.replace(". ", ".\n")
    print(f"\nAnswer:\n{output_text.strip()}")

    # Save to chat history
    chat_history.append({
        "question": human_input,
        "answer": output_text.strip()
    })