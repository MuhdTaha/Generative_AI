# Project 01: AI-Powered Knowledge Graph for Automated Resume Screening

This project is a Streamlit web application that automates the process of resume screening by leveraging AI and a Neo4j graph database. It transforms unstructured PDF resumes into a structured, interactive knowledge graph, allowing for powerful and intuitive querying of candidate information.


## ✨ Features

- **Interactive Web Interface**: A user-friendly front-end built with Streamlit.
- **Bulk Resume Processing**: Upload and process multiple PDF resumes at once.
- **AI-Powered Entity Extraction**: Uses Google's Gemini AI to intelligently parse resumes and extract key entities (e.g., names, skills, companies) and their relationships.
- **Automated Knowledge Graph Construction**: Automatically populates a Neo4j database with the extracted data.
- **Interactive Graph Visualization**: Displays the entire resume knowledge graph directly in the app, with interactive nodes that can be dragged, zoomed, and explored.
- **Dynamic & Scalable**: The graph grows and becomes more powerful with each resume added.

## ⚙️ How It Works

1. **Upload**: The user uploads one or more PDF resumes via the Streamlit interface.
2. **Process**: When the "Build Knowledge Graph" button is clicked, the application iterates through each PDF.
3. **Extract**: The text content of each resume is extracted.
4. **Analyze**: The extracted text is sent to the Gemini AI model with a carefully crafted prompt, asking it to return a structured JSON object containing entities and relationships.
5. **Ingest**: The application parses the AI's response and writes the data to the Neo4j database using parameterized Cypher queries, creating Entity, File, and Resume nodes and the relationships between them.
6. **Visualize**: The "Show/Refresh Graph" button queries the Neo4j database and uses the streamlit-agraph component to render an interactive visualization of the knowledge graph.

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.8+
- A running Neo4j AuraDB instance or local installation.
- A Google AI API key for the Gemini model.

### Steps

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Set up your environment variables:
   Create a file named `.env` in the root of the project and add your credentials:

```
NEO4J_URI="neo4j+s://your-database-id.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your-neo4j-password"
GOOGLE_API_KEY="your-google-ai-api-key"
```

## 🚀 Usage

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

Your web browser will open with the application running. You can then upload your resumes and build your knowledge graph.

## 💻 Technologies Used

- **Frontend**: Streamlit
- **AI / LLM**: Google Gemini
- **Database**: Neo4j
- **PDF Processing**: PyPDF2
- **Graph Visualization**: streamlit-agraph
- **Core Language**: Python

---

**Developed by Muhammad Taha**