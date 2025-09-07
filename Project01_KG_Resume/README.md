# Neo4j Knowledge Graph for Automated Resume Screening

This project automates the process of screening multiple resumes by extracting key information from PDF files and structuring it into a query-able Neo4j knowledge graph. It leverages the power of Google's Gemini AI model to intelligently identify entities (like names, skills, and companies) and their relationships, providing a powerful tool for recruiters and HR professionals to quickly find and analyze candidate profiles.

## Project Overview

Recruiting often involves sifting through hundreds of resumes to find the right candidate. This manual process is time-consuming and prone to overlooking qualified applicants. This project tackles that challenge by building an automated pipeline that:

- **Reads PDF Resumes**: It processes a folder containing multiple resumes in PDF format.
- **Extracts Information with AI**: It sends the text content of each resume to the Google Gemini AI model to extract structured data, including entities (people, universities, skills) and triples (the relationships between them, e.g., "Ada Lovelace" HAS_SKILL "JavaScript").
- **Builds a Knowledge Graph**: It takes the structured data from the AI and populates a Neo4j graph database, creating a network of interconnected information that represents the collective qualifications of all candidates.

The result is a rich, queryable knowledge graph that allows for complex questions like, "Find all candidates who have experience with Python and have worked at Google," or "Show me the skills most commonly associated with a Data Scientist role."

## Features

- **Batch Processing**: Automatically processes all PDF files located in a `resumes/` directory.
- **Intelligent Extraction**: Uses Google's Gemini Flash model for high-quality entity and relationship extraction from unstructured text.
- **Structured Knowledge Base**: Populates a Neo4j database, creating a robust and scalable knowledge graph.
- **Automated Data Linking**: Ensures all extracted information is connected to a central "root" entity (the candidate), creating a cohesive profile.
- **Dynamic File Naming**: Automatically generates file nodes in the graph based on the resume's filename.

## Technology Stack

- **Language**: Python 3.x
- **AI Model**: Google Gemini 2.5 Flash
- **Database**: Neo4j Graph Database (can be run locally or on a cloud service like Neo4j Aura)
- **Python Libraries**:
  - `google-generativeai`: To interact with the Gemini API.
  - `neo4j`: The official Python driver for Neo4j.
  - `python-dotenv`: For managing environment variables.
  - `PyPDF2`: For extracting text from PDF files.

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

- Python 3.7 or higher.
- A running Neo4j instance (e.g., Neo4j Desktop or a free Neo4j AuraDB instance).
- A Google Gemini API key. You can get one from the [Google AI Studio](https://makersuite.google.com/app/apikey).

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 3. Set up the Python Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the root directory of the project and add your credentials in the following format:

```env
# .env

# Neo4j Credentials
NEO4J_URI="bolt://your-neo4j-instance-uri:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your-neo4j-password"

# Google Gemini API Key
GEMINI_API_KEY="your-google-ai-api-key"
```

## How to Use

1. **Add Resumes**: Inside the folder named `resumes` in the project's root directory, place all the PDF resume files you want to process.

2. **Run the Script**: Execute the main Python script from your terminal:

```bash
python createKG.py
```

3. **Monitor the Output**: The script will print its progress to the console, indicating when it connects to Neo4j and which file it is currently processing.

```
✅ Connected to Neo4j

Processing 'Ada_Lovelace_Resume.pdf' -> File Name: 'Ada Lovelace Resume'
✅ Processed 'Ada_Lovelace_Resume.pdf' successfully. Entities: 15, Relationships: 14

Processing 'Charles_Babbage_Resume.pdf' -> File Name: 'Charles Babbage Resume'
✅ Processed 'Charles_Babbage_Resume.pdf' successfully. Entities: 12, Relationships: 11

✅ All resumes processed.
🔒 Neo4j connection closed
```

4. **Explore the Graph**: Once the script finishes, you can explore the knowledge graph using the Neo4j Browser. Open your Neo4j instance and run a Cypher query to visualize the data, for example:

```cypher
MATCH (n) RETURN n LIMIT 50;
```

## Project Structure

```
.
├── resumes/                  # Directory for your PDF resume files
│   ├── Ada_Resume.pdf
│   ├── Teresa_Resume.pdf
│   ├── Terrence_Resume.pdf
│   └── Taha_Resume.pdf
├── createKG.py               # The main Python script for the project
├── .env                      # File for storing environment variables (credentials)
└── README.md                 # This file
```