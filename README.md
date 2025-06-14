
# 🧠 Intelligent Document Processing with AI-Powered Agent

An AI-powered intelligent agent that extracts, processes, evaluates, and understands the content of PDF documents from Amazon S3. Leveraging **AWS Textract**, **NLTK**, **Pinecone**, **Hugging Face embeddings**, and **Amazon Titan LLMs**, it provides context-aware question answering with multi-turn memory and automatic model evaluation to select the best response.

---

## 🚀 Features

* 📄 **Text Extraction** from PDFs in **Amazon S3** using **AWS Textract**.
* 🧹 **Text Preprocessing** with **NLTK**: tokenization, stopword removal, and lemmatization.
* 🔍 **Embeddings** via Hugging Face `multi-qa-mpnet-base-cos-v1`.
* 📦 **Vector Store** using **Pinecone** for document retrieval.
* 🧠 **Retrieval-Augmented Generation (RAG)** using:
  - **Amazon Titan Text Express**
  - **Amazon Titan Text Premier**
  - **Amazon Titan Text Lite**
* 🗣️ **Multi-Turn Conversations** with conversation history memory.
* 📊 **Response Evaluation** using:
  - Cosine similarity to query and context
  - Fluency analysis
  - Generation time penalties
* 🏆 **Automated Best Response Selection** from three LLM outputs.
* 📈 **Benchmarking & Visualization**:
  - Score trends over time
  - Generation time per model
  - Comparative bar charts for performance and efficiency

---

## 📦 Installation

### 1. Install Required Packages

```bash
pip install -U boto3 langchain langchain-pinecone langchain-community nltk sentence-transformers langchain-huggingface
````

### 2. Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🔐 Setup Credentials

Configure the following variables in the script:

```python
AWS_ACCESS_KEY_ID = 'your-aws-access-key-id'
AWS_SECRET_ACCESS_KEY = 'your-aws-secret-access-key'
AWS_REGION = 'us-east-1'

PINECONE_API_KEY = 'your-pinecone-api-key'
PINECONE_INDEX = 'your-pinecone-index-name'

S3_BUCKET_NAME = 'your-s3-bucket-name'
PDF_FILE_NAME = 'your-document.pdf'
```

---

## 💡 Usage

1. Upload your PDF to your S3 bucket.

2. Run the script:

   * Extracts text with **Textract**
   * Cleans & lemmatizes content
   * Splits text into chunks
   * Embeds and stores vectors in Pinecone
   * Enters chat loop for Q\&A

3. Ask questions interactively—answers are generated by all three models and scored automatically.

---

## 🧠 AI Agent Behavior

The agent uses Retrieval-Augmented Generation (RAG) with context retrieved from document embeddings. Responses from **Titan Express**, **Premier**, and **Lite** are:

* Generated with both full and truncated context windows
* Scored for relevance, coherence, and efficiency
* Ranked using a custom scoring function

The best response is selected **automatically**.

---

### Sample Interaction

```
User: What was the net revenue in 2023?
AI: Acme Corp reported $5.2 million in revenue for 2023.

User: What drove that growth?
AI: Primarily North American expansion and a new analytics platform.

User: Any setbacks?
AI: The hardware division saw declines due to supply chain disruptions.
```

---

## 📊 Response Evaluation Logic

Each response is scored by:

* 🔁 **Query Similarity** (cosine similarity to question)
* 📚 **Context Similarity** (cosine similarity to retrieved chunks)
* ✍️ **Fluency Score** (based on word count)
* ⏱️ **Time Penalty** (penalizes slower models)

Final score:

```text
score = 0.35 * query_sim + 0.35 * context_sim + 0.2 * fluency_score + 0.1 * (1 - norm_gen_time)
```

---

## 📉 Visual Benchmarking

After interacting:

* 📈 Line charts show score and time trends
* 📊 Bar charts compare average scores and times per model
* 🧮 Throughput (responses/sec) is printed after each round

---

## 💼 Example Use Cases

Ask questions like:

* “Summarize key financials for 2023”
* “What products launched last year?”
* “Mention challenges in Q4”

Ideal for:

* Annual reports
* Financial disclosures
* Legal contracts
* Research whitepapers

---

## 📬 Contributions & Feedback

Have ideas or found a bug? Contributions are welcome—submit issues, feature requests, or PRs to improve this project.

---

