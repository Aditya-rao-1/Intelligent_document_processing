# Install required packages
!pip install -U boto3 langchain langchain-pinecone langchain-community nltk sentence-transformers langchain-huggingface

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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# AWS credentials (REDACTED)
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""

# AWS Textract client
client = boto3.client(
    'textract',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name='us-east-1'
)

# Start document text detection
response = client.start_document_text_detection(
    DocumentLocation={
        'S3Object': {
            'Bucket': '',
            'Name': 'Cyber.pdf'
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

# Extract text
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

# Combine and save text
full_text = "\n".join(extracted_text)
with open("demo_rag_on_image.txt", "w") as f:
    f.write(full_text)

# NLP Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

preprocessed_text = preprocess_text(full_text)
docs = [Document(page_content=preprocessed_text)]

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=250, separator="\n")
split_docs = text_splitter.split_documents(docs)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")

# Pinecone setup
os.environ["PINECONE_API_KEY"] = ""
index_name = "textract"
docsearch = PineconeVectorStore.from_documents(split_docs, embedding_model, index_name=index_name)

print("Processing complete!")

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

# Bedrock client (REDACTED)
boto3_bedrock = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Evaluation model
eval_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_response(query, response, context):
    q_emb = eval_model.encode(query, convert_to_tensor=True)
    r_emb = eval_model.encode(response, convert_to_tensor=True)
    c_emb = eval_model.encode(context, convert_to_tensor=True)
    query_sim = float(util.cos_sim(q_emb, r_emb)[0][0])
    context_sim = float(util.cos_sim(c_emb, r_emb)[0][0])
    words = [w for w in word_tokenize(response) if w.isalpha()]
    fluency_len = len(words)
    fluency_score = min(fluency_len / 20, 1.0)
    fluency = "High" if fluency_len > 10 else "Low"
    return {
        "query_similarity": query_sim,
        "context_similarity": context_sim,
        "fluency": fluency,
        "fluency_score": fluency_score
    }

def scoring_fn(metrics, gen_time, max_gen_time):
    norm_time = gen_time / max_gen_time if max_gen_time > 0 else 1.0
    time_penalty = 1.0 - norm_time
    return (
        0.35 * metrics["query_similarity"]
        + 0.35 * metrics["context_similarity"]
        + 0.2 * metrics["fluency_score"]
        + 0.1 * time_penalty
    )

# Score tracking
express_scores = []
premier_scores = []
lite_scores = []
interaction_ids = []
interaction_count = 0
MAX_HISTORY_SIZE = 8
express_gen_times = []
premier_gen_times = []
lite_gen_times = []
total_times_express = []
total_times_premier = []
total_times_lite = []
chat_history = []

# Main loop
while True:
    if len(chat_history) >= MAX_HISTORY_SIZE:
        chat_history.clear()

    print(f"\n--- Interaction #{interaction_count + 1} ---")
    human_input = input("\nAsk a question (or type 'exit' to quit): ")
    if human_input.lower() == 'exit':
        break

    start = time.time()
    query_embedding = embedding_model.embed_query(human_input)
    embedding_time = time.time() - start

    start = time.time()
    search_results = docsearch.similarity_search(human_input, k=5)
    retrieval_time = time.time() - start

    context_string = '\n\n'.join(
        [f'Document {ind+1}: ' + i.page_content[:6000] for ind, i in enumerate(search_results)]
    )
    lite_context_string = '\n\n'.join(
        [f'Document {ind+1}: ' + i.page_content[:2000] for ind, i in enumerate(search_results)]
    )

    formatted_history = ""
    for turn in chat_history:
        formatted_history += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt_data = PROMPT.format(human_input=human_input, context=context_string, history=formatted_history)
    lite_prompt_data = PROMPT.format(human_input=human_input, context=lite_context_string, history="")

    body_part = json.dumps({
        'inputText': prompt_data,
        'textGenerationConfig': {
            'maxTokenCount': 1024,
            'stopSequences': [],
            'temperature': 0.7,
            'topP': 0.9
        }
    })
    lite_body_part = json.dumps({
        'inputText': lite_prompt_data,
        'textGenerationConfig': {
            'maxTokenCount': 1024,
            'stopSequences': [],
            'temperature': 0.7,
            'topP': 0.9
        }
    })

    start = time.time()
    express_text = json.loads(
        boto3_bedrock.invoke_model(body=body_part, contentType="application/json", accept="application/json", modelId='amazon.titan-text-express-v1')['body'].read()
    )['results'][0]['outputText'].strip()
    express_gen_time = time.time() - start

    start = time.time()
    premier_text = json.loads(
        boto3_bedrock.invoke_model(body=body_part, contentType="application/json", accept="application/json", modelId='amazon.titan-text-premier-v1:0')['body'].read()
    )['results'][0]['outputText'].strip()
    premier_gen_time = time.time() - start

    start = time.time()
    lite_text = json.loads(
        boto3_bedrock.invoke_model(body=lite_body_part, contentType="application/json", accept="application/json", modelId='amazon.titan-text-lite-v1')['body'].read()
    )['results'][0]['outputText'].strip()
    lite_gen_time = time.time() - start

    eval_express = evaluate_response(human_input, express_text, context_string)
    eval_premier = evaluate_response(human_input, premier_text, context_string)
    eval_lite = evaluate_response(human_input, lite_text, context_string)
    max_gen_time = max(express_gen_time, premier_gen_time, lite_gen_time)

    score_express = scoring_fn(eval_express, express_gen_time, max_gen_time)
    score_premier = scoring_fn(eval_premier, premier_gen_time, max_gen_time)
    score_lite = scoring_fn(eval_lite, lite_gen_time, max_gen_time)

    gen_times = {"Express": express_gen_time, "Premier": premier_gen_time, "Lite": lite_gen_time}
    total_times = {m: embedding_time + retrieval_time + gen_times[m] for m in gen_times}

    express_gen_times.append(express_gen_time)
    premier_gen_times.append(premier_gen_time)
    lite_gen_times.append(lite_gen_time)
    total_times_express.append(total_times["Express"])
    total_times_premier.append(total_times["Premier"])
    total_times_lite.append(total_times["Lite"])

    scores = {"Express": score_express, "Premier": score_premier, "Lite": score_lite}
    best_model = max(scores, key=scores.get)
    best_eval = {"Express": eval_express, "Premier": eval_premier, "Lite": eval_lite}[best_model]
    best_text = {"Express": express_text, "Premier": premier_text, "Lite": lite_text}[best_model].replace(". ", ".\n")

    interaction_count += 1
    interaction_ids.append(interaction_count)
    express_scores.append(eval_express)
    premier_scores.append(eval_premier)
    lite_scores.append(eval_lite)

    avg_throughput = (interaction_count * 3) / sum(total_times_express + total_times_premier + total_times_lite)

    print(f"\nAnswer from {best_model}:\n{best_text}")
    print(f"\n--- Model Generation Time ---\nEmbedding: {embedding_time:.2f}s | Retrieval: {retrieval_time:.2f}s | Generation: {gen_times[best_model]:.2f}s | Total: {total_times[best_model]:.2f}s")
    print(f"Throughput: {avg_throughput:.3f} responses/s")

    print(f"\n--- Model Score ---\nQuery Similarity: {best_eval['query_similarity']:.3f}\nContext Similarity: {best_eval['context_similarity']:.3f}\nFluency: {best_eval['fluency']} ({best_eval['fluency_score']:.3f})\nFinal Score: {scores[best_model]:.3f}")

    chat_history.append({"question": human_input, "answer": best_text})

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# Calculate final scores list for each model
express_final_scores = [
    scoring_fn(eval_express, gen_time, max(express_gen_times[i], premier_gen_times[i], lite_gen_times[i]))
    for i, (eval_express, gen_time) in enumerate(zip(express_scores, express_gen_times))
]

premier_final_scores = [
    scoring_fn(eval_premier, gen_time, max(express_gen_times[i], premier_gen_times[i], lite_gen_times[i]))
    for i, (eval_premier, gen_time) in enumerate(zip(premier_scores, premier_gen_times))
]

lite_final_scores = [
    scoring_fn(eval_lite, gen_time, max(express_gen_times[i], premier_gen_times[i], lite_gen_times[i]))
    for i, (eval_lite, gen_time) in enumerate(zip(lite_scores, lite_gen_times))
]

# Plot 1: Scores
plt.subplot(1, 2, 1)
plt.plot(interaction_ids, express_final_scores, label='Express Score')
plt.plot(interaction_ids, premier_final_scores, label='Premier Score')
plt.plot(interaction_ids, lite_final_scores, label='Lite Score')
plt.xlabel('Interaction #')
plt.ylabel('Final Score')
plt.title('Model Scores Over Interactions')
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True)

# Plot 2: Times
plt.subplot(1, 2, 2)
plt.plot(interaction_ids, total_times_express, label='Express Total Time')
plt.plot(interaction_ids, total_times_premier, label='Premier Total Time')
plt.plot(interaction_ids, total_times_lite, label='Lite Total Time')
plt.xlabel('Interaction #')
plt.ylabel('Total Time (seconds)')
plt.title('Total Time Over Interactions')
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np

# Compute average scores and average total times per model
avg_express_score = np.mean([
    scoring_fn(eval_express, gen_time, max(express_gen_times[i], premier_gen_times[i], lite_gen_times[i]))
    for i, (eval_express, gen_time) in enumerate(zip(express_scores, express_gen_times))
])
avg_premier_score = np.mean([
    scoring_fn(eval_premier, gen_time, max(express_gen_times[i], premier_gen_times[i], lite_gen_times[i]))
    for i, (eval_premier, gen_time) in enumerate(zip(premier_scores, premier_gen_times))
])
avg_lite_score = np.mean([
    scoring_fn(eval_lite, gen_time, max(express_gen_times[i], premier_gen_times[i], lite_gen_times[i]))
    for i, (eval_lite, gen_time) in enumerate(zip(lite_scores, lite_gen_times))
])

avg_express_time = np.mean(total_times_express)
avg_premier_time = np.mean(total_times_premier)
avg_lite_time = np.mean(total_times_lite)

models = ['Express', 'Premier', 'Lite']
avg_scores = [avg_express_score, avg_premier_score, avg_lite_score]
avg_times = [avg_express_time, avg_premier_time, avg_lite_time]

plt.figure(figsize=(16, 6))

# Bar chart for average scores
plt.subplot(1, 3, 1)
bars = plt.bar(models, avg_scores, color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Average Model Scores')
plt.ylabel('Score')
plt.ylim(0, 1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center')

# Bar chart for average total times
plt.subplot(1, 3, 2)
bars = plt.bar(models, avg_times, color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Average Total Time (sec)')
plt.ylabel('Seconds')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center')
plt.tight_layout()
plt.show()
