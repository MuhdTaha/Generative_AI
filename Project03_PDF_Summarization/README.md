# Project 03: Reflection-Critic Agents for Efficient PDF Summarization

This project is an interactive Streamlit web application that uses a multi-agent system built with LangGraph to generate high-quality summaries of PDF documents. It features a "Summarizer" agent that creates an initial summary and a "Critic" agent that provides feedback for revision, ensuring a refined and accurate final output.

## ✨ Features

- **Interactive UI**: A clean and user-friendly interface built with Streamlit.
- **PDF Upload**: Supports uploading any PDF document for summarization.
- **OCR for Scanned PDFs**: Automatically performs Optical Character Recognition (OCR) on scanned or image-based PDFs to extract text.
- **Reflection-Critic Agent System**: Utilizes a LangGraph-powered loop where a summary is generated, critiqued, and revised to improve quality.
- **Token-Aware Processing**: Intelligently chunks and batches the document to handle large files without exceeding the context limits of the language model.
- **Secure Configuration**: Manages API keys and local file paths securely using an environment file.
- **Customizable Settings**: Allows users to adjust advanced parameters like reflection loops, batch size, and chunk size directly from the UI.

## 🤖 How It Works

The application follows a multi-step process to generate a summary:

1. **PDF Processing & OCR**: When a PDF is uploaded, it is first processed page by page. If a page contains no machine-readable text (i.e., it's an image or a scan), Google's Tesseract OCR engine is used to extract the text.

2. **Batching & Chunking**: The extracted text is divided into smaller, manageable chunks and grouped into batches to ensure reliable processing and avoid token limits.

3. **Summarization Loop (LangGraph)**: Each chunk goes through a reflection-critic loop:
   - **Summarizer Agent**: Generates an initial summary of the text chunk.
   - **Critic Agent**: Critiques the summary, identifying potential flaws, missing details, or areas for improvement.
   - **Revision**: The Summarizer agent receives the critique and generates a revised, improved summary. This loop can run multiple times for enhanced quality.

4. **Final Combination**: After all chunks have been summarized, the individual summaries are combined and passed through a final summarization step to create a single, cohesive, and comprehensive summary of the entire document.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **Streamlit**: For building the interactive web interface.
- **Google Gemini**: The language model used for summarization and critique.
- **LangGraph**: To create and manage the multi-agent reflection-critic system.
- **LangChain**: For document loading and prompt management.
- **Tesseract & Poppler**: For OCR and PDF-to-image conversion.
- **Dotenv**: For environment variable management.

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

You must have the following installed on your system:

- **Python 3.8+**
- **Poppler**: Required for converting PDF pages to images.
  - Download and installation instructions for Windows available [here](https://github.com/oschwartz10612/poppler-windows).
  - For Mac/Linux, you can use Homebrew (`brew install poppler`) or your system's package manager.
- **Google Tesseract OCR**: Required for extracting text from scanned documents.
  - Download and installation instructions available [here](https://github.com/tesseract-ocr/tesseract).

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create an Environment File
Create a file named .env in the root of your project directory and add the following variables:

```bash
GEMINI_API_KEY="your_google_api_key_here"
POPPLER_PATH="C:/path/to/your/poppler/bin"
```

- Replace "your_google_api_key_here" with your actual Google AI Studio API key.
- Update the POPPLER_PATH to point to the bin directory of your Poppler installation.

## 🚀 Usage 
Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```

The application will open in your web browser. From there, you can:
- Upload a PDF file using the file uploader in the sidebar.
- Adjust any advanced settings if needed.
- Click the "Summarize PDF" button to start the process.
