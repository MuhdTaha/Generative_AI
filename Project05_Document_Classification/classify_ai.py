import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------
# Configuration
# ---------------------------
load_dotenv()

# Define the possible document types for classification
DOCUMENT_TYPES = [
    "Invoice",
    "Resume",
    "Contract",
    "Scientific Paper",
    "News Article",
    "Legal Document",
    "Financial Report",
    "Marketing Material",
    "Research Report",
    "Lecture Notes",
    "Presentation",
    "Technical Document",
    "Other"
]

# ---------------------------
# Document Loading
# ---------------------------
def load_document(file_path: str) -> str:
    """
    Loads a document from the given file path and returns its text content.
    Supports PDF, TXT, and DOCX files.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    try:
        documents = loader.load()
        # Combine the content of all pages/parts into a single string
        full_text = "\n".join([doc.page_content for doc in documents])
        return full_text
    except Exception as e:
        print(f"Error loading document: {e}")
        return ""

# ---------------------------
# AI-Powered Classification
# ---------------------------
def classify_document(file_path: str, categories: List[str]) -> Dict[str, Any]:
    """
    Classifies a document into one of the given categories using Gemini.
    Always returns a well-structured dictionary.
    """
    # Load the document
    document_text = load_document(file_path)
    if not document_text.strip():
        return {
            "predicted_class": "Unknown",
            "confidence_score": 0.0,
            "explanation": "No readable text found in document.",
            "text": ""
        }

    truncated_text = document_text[:8000]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = f"""
You are an expert document classifier.

Classify the following document into **one** of these categories:
{", ".join(categories)}

Return ONLY a JSON object with this structure:
{{
  "predicted_class": "<category>",
  "confidence_score": <float between 0.0 and 1.0>,
  "explanation": "<1-2 sentence reasoning>"
}}

Document text:
---
{truncated_text}
---
"""

    try:
        response = llm.invoke(prompt)
        text_response = response.content.strip()

        # Try parsing JSON
        try:
            parsed = json.loads(text_response)
        except json.JSONDecodeError:
            # If Gemini added extra text, extract JSON safely
            import re
            match = re.search(r'\{.*\}', text_response, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}

        # Return clean result
        return {
            "predicted_class": parsed.get("predicted_class", "Unknown"),
            "confidence_score": parsed.get("confidence_score", 0.0),
            "explanation": parsed.get("explanation", "No explanation provided."),
            "text": truncated_text
        }

    except Exception as e:
        return {
            "predicted_class": "Error",
            "confidence_score": 0.0,
            "explanation": f"AI classification failed: {str(e)}",
            "text": truncated_text
        }
