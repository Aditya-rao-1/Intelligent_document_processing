!pip install -U boto3 langchain langchain-pinecone langchain-community nltk sentence-transformers -U langchain-huggingface
# Imports
import boto3
import os
import time
import json
import nltk
from sentence_transformers import SentenceTransformer, util
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

# AWS Credentials
AWS_ACCESS_KEY_ID=''  # Replace with your AWS Access Key ID
AWS_SECRET_ACCESS_KEY=''  # Replace with your AWS Secret Access Key

# AWS Configuration
AWS_REGION=''  # Set the AWS region (default: us-east-1)

# Pinecone API Key and Index
PINECONE_API_KEY=''  # Replace with your Pinecone API Key
PINECONE_INDEX=''  # Replace with your Pinecone Index Name

# Amazon S3 Bucket and File Details
S3_BUCKET_NAME=''  # Replace with your S3 bucket name where the document is stored
PDF_FILE_NAME=''  # Replace with the filename of the document to process

# AWS Textract client
client = boto3.client(
    'textract',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Start document text detection
response = client.start_document_text_detection(
    DocumentLocation={"S3Object": {"Bucket": S3_BUCKET_NAME, "Name": PDF_FILE_NAME}}
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

# Extract Text from Response
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
output_file_name = "extracted_text.txt"
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

# Prepare Document for Embedding
docs = [Document(page_content=preprocessed_text)]

# Split document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=250, separator="\n")
split_docs = text_splitter.split_documents(docs)

# Use Embeddings for Text Processing
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

docsearch = PineconeVectorStore.from_documents(split_docs, embedding_model, index_name=PINECONE_INDEX)

print("Processing complete!")

# Conversation history
chat_history = []

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

# Bedrock model
boto3_bedrock = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Sentence transformer model for evaluation
eval_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_response(query, response, context):
    q_emb = eval_model.encode(query, convert_to_tensor=True)
    r_emb = eval_model.encode(response, convert_to_tensor=True)
    c_emb = eval_model.encode(context, convert_to_tensor=True)
    fluency = len([w for w in word_tokenize(response) if w.isalpha()])
    return {
        "query_similarity": float(util.cos_sim(q_emb, r_emb)),
        "context_similarity": float(util.cos_sim(c_emb, r_emb)),
        "fluency": "High" if fluency > 10 else "Low",
        "fluency_score": fluency / 100
    }

def scoring_fn(metrics):
    return 0.4 * metrics["query_similarity"] + 0.4 * metrics["context_similarity"] + 0.2 * metrics["fluency_score"]

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

    prompt_data = PROMPT.format(
        human_input=human_input,
        context=context_string,
        history=formatted_history
    )

    # Prepare body for both models
    body_part = json.dumps({
        'inputText': prompt_data,
        'textGenerationConfig': {
            'maxTokenCount': 3072,
            'stopSequences': [],
            'temperature': 0.7,
            'topP': 1
        }
    })

    express_response = boto3_bedrock.invoke_model(
        body=body_part,
        contentType="application/json",
        accept="application/json",
        modelId='amazon.titan-text-express-v1'
    )
    express_text = json.loads(express_response['body'].read())['results'][0]['outputText'].strip()

    premier_response = boto3_bedrock.invoke_model(
        body=body_part,
        contentType="application/json",
        accept="application/json",
        modelId='amazon.titan-text-premier-v1:0'
    )
    premier_text = json.loads(premier_response['body'].read())['results'][0]['outputText'].strip()

    # Evaluate and choose best
    eval_express = evaluate_response(human_input, express_text, context_string)
    eval_premier = evaluate_response(human_input, premier_text, context_string)

    score_express = scoring_fn(eval_express)
    score_premier = scoring_fn(eval_premier)

    best_model = "Express" if score_express >= score_premier else "Premier"
    best_text = express_text if score_express >= score_premier else premier_text
    best_text = best_text.replace(". ", ".\n")
    best_eval = eval_express if score_express >= score_premier else eval_premier

    print(f"\nAnswer from {best_model}:\n{best_text}")
    print(f"\n[Evaluation Summary]")
    print(f"Query Similarity: {best_eval['query_similarity']:.3f}")
    print(f"Context Similarity: {best_eval['context_similarity']:.3f}")
    print(f"Fluency: {best_eval['fluency']} ({best_eval['fluency_score']:.3f})")
    print(f"Final Score: {max(score_express, score_premier):.3f}")

    # Save to chat history
    chat_history.append({
        "question": human_input,
        "answer": best_text
    })
    
     
