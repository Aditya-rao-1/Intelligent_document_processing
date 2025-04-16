
# 🧠 Intelligent Document Processing with AI-Powered Agent

An AI-powered intelligent agent that extracts, processes, and understands the content of PDF documents from Amazon S3. Using AWS Textract, NLP techniques, Pinecone vector store, and Amazon Titan LLM, it enables context-aware question-answering with multi-turn memory.

---

## 🚀 Features

- 📄 **Text Extraction** from PDF documents stored in **Amazon S3** using **AWS Textract**.
- 🧹 **Text Preprocessing** using **NLTK** (tokenization, stopword removal, lemmatization).
- 🔍 **Embedding Generation** with Hugging Face's `multi-qa-mpnet-base-cos-v1` model.
- 📦 **Vector Storage and Retrieval** via **Pinecone**.
- 🧠 **Contextual Q&A** using **Amazon Titan LLM** in a **Retrieval-Augmented Generation (RAG)** workflow.
- 🗣️ **Multi-Turn Conversation Support** with memory for follow-up questions.

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

Update your script with the following configuration:

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

1. Upload your PDF to the specified S3 bucket.
2. Run the script to:
   - Extract text with **AWS Textract**.
   - Preprocess with **NLTK**.
   - Generate and store embeddings in **Pinecone**.
3. Start the interactive conversation loop and ask context-based questions.

---

## 🧠 AI Agent Behavior

This system maintains conversational memory for context-aware responses. Example interaction:

```
User: What was the net revenue of Acme Corp in 2023?
AI: Acme Corp reported a net revenue of $5.2 million in 2023.

User: What contributed to that growth?
AI: The main contributors were increased product sales in North America and the launch of their AI-driven analytics platform.

User: Was there any decline in other segments?
AI: Yes, there was a slight decline in the hardware division due to supply chain issues.
```

---

## 💼 Example Use Case

If your document is a company’s annual report, you can ask:

- "What was the profit in 2022?"
- "Which sectors performed best this year?"
- "How does this year’s performance compare to last year?"

The AI agent uses the document and memory to provide accurate, contextual responses.

---

## 🌱 Future Enhancements

- 🖥️ Develop a GUI or REST API for user-friendly access.
- 📚 Support simultaneous processing of multiple documents.
- 🏢 Integrate with enterprise document processing pipelines.

---

## 📬 Contributions & Feedback

Feel free to submit issues, feature requests, or pull requests to help improve this project!

---

Let me know if you want a version with badges, a license, or contribution guidelines added.
