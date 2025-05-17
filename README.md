
# 🧠 Intelligent Document Processing with AI-Powered Agent

An AI-powered intelligent agent that extracts, processes, evaluates, and understands the content of PDF documents from Amazon S3. Leveraging AWS Textract, NLP, Pinecone, Hugging Face embeddings, and **Amazon Titan LLMs**, it provides **context-aware question answering** with **multi-turn memory**, and **automated model evaluation** for the best responses.

---

## 🚀 Features

* 📄 **Text Extraction** from PDF documents stored in **Amazon S3** using **AWS Textract**.
* 🧹 **Text Preprocessing** with **NLTK** (tokenization, stopword removal, lemmatization).
* 🔍 **Embedding Generation** via **Hugging Face** `multi-qa-mpnet-base-cos-v1`.
* 📦 **Vector Storage & Retrieval** using **Pinecone**.
* 🧠 **Retrieval-Augmented Generation (RAG)** using **Amazon Titan Text Express** and **Premier LLMs**.
* 🗣️ **Multi-Turn Conversations** with persistent memory.
* 🤖 **Model Evaluation & Scoring** with **semantic similarity and fluency metrics**.
* 🏆 **Automatic Best Response Selection** from two LLM outputs using scoring logic.

---

## 📦 Installation

### 1. Install Required Packages

```bash
pip install -U boto3 langchain langchain-pinecone langchain-community nltk sentence-transformers langchain-huggingface
```

### 2. Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🔐 Setup Credentials

Update the script with your credentials and configuration:

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

   * Extracts text using **AWS Textract**.
   * Cleans and preprocesses the text.
   * Splits and embeds the document.
   * Stores vectors in **Pinecone**.
3. Enter the interactive conversation loop and ask questions.

---

## 🧠 AI Agent Behavior

The system uses retrieval-augmented generation and memory to enable deep, context-aware responses. It compares responses from **Titan Express** and **Titan Premier**, then selects the most relevant answer using:

* **Query relevance**
* **Context alignment**
* **Fluency scoring**

### Sample Interaction

```
User: What was the net revenue of Acme Corp in 2023?
AI: Acme Corp reported a net revenue of $5.2 million in 2023.

User: What contributed to that growth?
AI: Growth was driven by increased sales in North America and the launch of a new analytics platform.

User: Were there any setbacks?
AI: Yes, the hardware division experienced a decline due to supply chain issues.
```

---

## 📊 Response Evaluation Logic

The system scores LLM responses using:

* 🔁 **Cosine Similarity** to the original query and the retrieved context.
* ✍️ **Fluency Score** based on token analysis.
* 🏅 **Weighted Final Score** to choose the best model's response automatically.

---

## 💼 Example Use Cases

Ask your document anything:

* "Summarize key financials for 2023."
* "List product launches last year."
* "What challenges were mentioned in Q4?"

Works great with annual reports, legal documents, research papers, and more.

---

## 🌱 Future Enhancements

* 🖥️ Web UI or REST API for easier interaction.
* 📚 Multi-document RAG and chunk indexing.
* 🔐 Role-based access and audit trail for enterprise use.

---

## 📬 Contributions & Feedback

Have ideas or found a bug? Contributions are welcome—submit issues, feature requests, or PRs to improve this project.

---

