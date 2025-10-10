# Project 02: GraphRAG-Powered Resume Q&A System

This project is a Streamlit web application that serves as an intelligent Q&A system for resumes. It leverages a knowledge graph populated with resume data [(from Project 1)](https://github.com/MuhdTaha/Generative_AI/tree/main/Project01_KG_Resume) and uses a Retrieval-Augmented Generation (RAG) pipeline powered by Google's Gemini AI to answer natural language questions about the candidates.

## 🚀 Features

- **Database Connectivity**: Securely connect to a Neo4j graph database.
- **Candidate Listing**: Automatically lists all available candidates found in the knowledge graph.
- **Natural Language Q&A**: Ask complex questions about a single candidate or compare multiple candidates in plain English.
- **AI-Powered Answers**: Receives answers synthesized by the Gemini AI, based on the specific data retrieved from the graph.
- **Transparent Process**: Offers a "behind-the-scenes" look at the RAG process, showing the identified subjects, the generated Cypher query, and the raw data returned from the database.
- **Conversational History**: Keeps a running history of the current Q&A session, which resets on page refresh.

## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers.

1. **User Question**: The user asks a question through the Streamlit interface (e.g., "What skills do Ada and Alan have in common?").
2. **Subject Identification**: The question and a list of all known entities are sent to the Gemini AI, which identifies the main subjects of the query (e.g., ["Ada Lovelace", "Alan Turing"]).
3. **Cypher Query Generation**: The subjects, question, and graph schema are passed to Gemini a second time, instructing it to generate a precise Cypher query to fetch the relevant information from the Neo4j database.
4. **Graph Retrieval**: The generated Cypher query is executed against the Neo4j database, retrieving the raw data.
5. **Answer Synthesis**: The original question and the retrieved data are sent to Gemini a final time, which synthesizes a clear, human-readable answer.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Language**: Python
- **AI Model**: Google Gemini 2.5 Flash
- **Database**: Neo4j
- **Libraries**:
  - `google` (for Gemini AI)
  - `langchain-community` (for Neo4j graph interaction)
  - `neo4j` (Python driver)
  - `python-dotenv` (for Environmental Variables)

## ✅ Prerequisites

- Python 3.8 or newer.
- A running Neo4j database instance.
- The Neo4j database must be populated with data from the Project 1: Knowledge Graph Creation script. This Q&A system reads from that graph.
- A Google Gemini API key.

## 📦 Setup & Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Create an environment file:
   Create a file named `.env` in the root of the project directory and add your credentials:

```env
# .env file
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
NEO4J_URI="bolt://your-neo4j-host:7687"
NEO4J_USERNAME="your-neo4j-username"
NEO4J_PASSWORD="your-neo4j-password"
```

## ▶️ How to Run the Application

1. Open your terminal in the project directory.
2. Run the following command:

```bash
streamlit run app.py
```

3. The application will open automatically in a new browser tab.
4. Use the sidebar to connect to your database, and then start asking questions!

---

**Developed by Muhammad Taha**
