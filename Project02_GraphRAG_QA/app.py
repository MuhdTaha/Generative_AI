import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
import json
import time

# Load environment variables from .env file
load_dotenv()

# --- Gemini API Configuration ---
# Configure the Gemini API key
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --------------------------------------
# Gemini LLM Call
# --------------------------------------
def call_gemini(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Calls the Google Gemini API to generate content based on a prompt.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # Clean up the response text, removing markdown formatting
        if response.parts:
            return response.text.strip().replace("```cypher", "").replace("```json", "").replace("```", "").strip()
        else:
            print("Warning: Gemini API returned an empty response.")
            return ""
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {str(e)}")
        return ""

# --- Neo4j Helper Functions ---

@st.cache_resource(show_spinner=False)
def init_neo4j_connection():
    """Initialize connection to Neo4j database and cache it."""
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", ""),
            username=os.getenv("NEO4J_USERNAME", ""),
            password=os.getenv("NEO4J_PASSWORD", "")
        )
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", ""),
            auth=(os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", ""))
        )
        driver.verify_connectivity()
        return graph, driver
    except Exception as e:
        st.error(f"Failed to initialize Neo4j connection: {str(e)}")
        return None, None

def get_schema(_graph):
    """Get the graph schema from the Neo4jGraph object."""
    try:
        return _graph.get_schema if _graph else ""
    except Exception as e:
        print(f"Error retrieving schema: {str(e)}")
        return ""

def get_all_node_names(driver):
    """Fetch all unique node names from the graph."""
    if not driver:
        return []
    try:
        with driver.session() as session:
            query = "MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name"
            result = session.run(query)
            return [record["name"] for record in result]
    except Exception as e:
        print(f"Error in get_all_node_names: {str(e)}")
        return []

def get_all_relationship_types(driver):
    """Fetch all unique relationship types from the graph."""
    if not driver:
        return []
    try:
        with driver.session() as session:
            query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            result = session.run(query)
            return [record["relationshipType"] for record in result]
    except Exception as e:
        print(f"Error in get_all_relationship_types: {str(e)}")
        return []

def get_candidates(driver):
    """Fetch all candidate names from the graph."""
    if not driver:
        return []
    try:
        with driver.session() as session:
            query = "MATCH (p:Person) RETURN p.name AS name ORDER BY name"
            result = session.run(query)
            return [record["name"] for record in result]
    except Exception as e:
        st.error(f"Error fetching candidates from the graph: {str(e)}")
        return []

# --- Core RAG Logic using Gemini ---

def get_subject_nodes(query, nodes):
    """Select the primary subject nodes from a user query using Gemini."""
    if not query or not nodes:
        return []
    prompt = f"""
    You are an expert at identifying all the main subjects in a user's question for a knowledge graph query.
    Your task is to select all relevant entities from the 'List of nodes' that were **explicitly mentioned** in the 'Query'.

    **Instructions:**
    1.  Read the user's 'Query' carefully.
    2.  Identify all distinct entities **explicitly mentioned** in the query that are also present in the 'List of nodes'. Do not infer subjects that are not mentioned.
    3.  Prioritize matching people's names.
    4.  Return the results as a valid JSON list of strings. For example: ["Entity 1", "Entity 2"].
    5.  If no relevant entities from the list are found in the query, return an empty list `[]`.

    **Query:**
    {query}

    **List of nodes:**
    {nodes}

    **Relevant nodes from the list (JSON format):**
    """
    response_text = call_gemini(prompt)
    try:
        # The AI might return a list that includes nodes not in our master list, so we filter them out.
        identified_nodes = json.loads(response_text)
        # Filter to ensure all returned nodes are valid
        valid_nodes = [node for node in identified_nodes if node in nodes]
        return valid_nodes
    except (json.JSONDecodeError, TypeError):
        # If parsing fails or the response is not a list, return an empty list.
        return []


def generate_cypher_query(question, schema, subject_nodes, relationship_types):
    """Generates a Cypher query using the Gemini LLM for one or more subjects."""
    prompt = f"""You are an expert Neo4j Cypher query writer.
    Your task is to write a single, complete Cypher query to answer a user's question, which may involve one or more subjects.

    **Instructions:**
    - The 'Primary Subjects' list contains the names of the main entities to query.
    - **IMPORTANT**: When matching node properties like names, use case-insensitive comparisons. For example, use `WHERE toLower(p.name) = 'some name'` instead of `WHERE p.name = 'Some Name'`.
    - If the 'User Question' implies a comparison or intersection (e.g., "common skills"), write a query to find the common nodes.
    - Otherwise, write a query that retrieves the requested information for all subjects in the list.
    - **CRITICAL:** When using `UNION`, you **must** use the same aliases for the returned columns in all parts of the query. For example, `RETURN p.name AS personName, s.name AS skillName` must be used consistently.
    - To preserve context, your query must return the names of the subject node(s) and the name(s) of the target node(s).
    - You must only use the relationship types provided in the 'Available Relationship Types' list.

    **Graph Schema:**
    {schema}

    **Available Relationship Types:**
    {relationship_types}

    **User Question:**
    {question}

    **Primary Subjects:**
    {subject_nodes}

    **Cypher Statement:**"""
    return call_gemini(prompt)

def generate_final_response(question, query_result):
    """Generate a polished, final response from the raw query data using Gemini."""
    prompt = f"""
    You are a helpful AI assistant. You will be provided with a user's question and the raw data retrieved from a knowledge graph that should answer it.
    Your task is to synthesize this information into a clear, natural-language answer. If the data is empty or irrelevant, state that you couldn't find the information.

    User's Question: {question}
    Retrieved Data: {str(query_result)}
    Final Answer:
    """
    final_answer = call_gemini(prompt)
    return final_answer or "Sorry, I was unable to formulate a final answer."

def process_query(question: str, graph, driver):
    """Handle the complete query processing workflow."""
    if not all([question, graph, driver]):
        return {"error": "Missing required parameters.", "status": "error"}

    try:
        schema = get_schema(graph)
        nodes_list = get_all_node_names(driver)
        relationship_types = get_all_relationship_types(driver)

        subject_nodes = get_subject_nodes(question, nodes_list)
        
        # If no specific subjects are found, check for general queries about all candidates.
        if not subject_nodes:
            general_keywords = ["all candidates", "everyone", "any candidate", "common skills", "most common"]
            if any(keyword in question.lower() for keyword in general_keywords):
                # This is a general query, so we'll use all candidates as the subjects.
                all_candidates = get_candidates(driver)
                if all_candidates:
                    subject_nodes = all_candidates
                else:
                    return {"error": "No candidates found in the database to query.", "status": "error"}

        if not subject_nodes:
            return {"error": "Failed to identify any subject nodes in your question. Please be more specific.", "status": "error"}
        
        cypher_query = generate_cypher_query(question, schema, subject_nodes, relationship_types)
        if not cypher_query:
            return {"error": "The AI failed to generate a Cypher query for your question.", "status": "error"}
        
        raw_data = graph.query(cypher_query)

        if raw_data:
            final_answer = generate_final_response(question, raw_data)
            return {
                "status": "success",
                "subject_nodes": subject_nodes,
                "result": final_answer,
                "intermediate_steps": {"cypher_query": cypher_query, "raw_data": raw_data}
            }
        else:
            return {
                "status": "partial",
                "subject_nodes": subject_nodes,
                "result": "I found the subject(s) in the graph, but couldn't find the specific information you asked for.",
                "intermediate_steps": {"cypher_query": cypher_query, "raw_data": []}
            }

    except Exception as e:
        print(f"An error occurred in process_query: {str(e)}")
        return {"error": str(e), "status": "error"}

# --- Streamlit Frontend ---
def main():
    st.set_page_config(page_title="Resume Q&A with GraphRAG", page_icon="📄", layout="wide")

    st.title("📄 GraphRAG-Powered Resume Q&A System")

    # Initialize session state
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'connecting' not in st.session_state:
        st.session_state.connecting = False
    if 'disconnecting' not in st.session_state:
        st.session_state.disconnecting = False
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'driver' not in st.session_state:
        st.session_state.driver = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # --- Connection Sidebar ---
    with st.sidebar:
        st.header("Database Connection")
        
        if st.session_state.db_connected:
            st.success("Connected to Neo4j Database.")
            if st.button("Disconnect from Database"):
                st.session_state.disconnecting = True
                st.rerun()
        else:
            st.warning("Not connected to Neo4j.")
            if st.button("Connect to Database"):
                st.session_state.connecting = True
                st.rerun()

        # Footer
        footer_html = """
        <div class="sidebar-footer">
            <hr>
            <p>Developed by <strong>Muhammad Taha</strong></p>
            <a href="https://www.linkedin.com/in/muhdtaha/" target="_blank" style="margin: 0 10px;">
                <img src="https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png" alt="LinkedIn" width="50" height="50">
            </a>
            <a href="https://github.com/MuhdTaha" target="_blank" style="margin: 0 10px;">
                <img src="https://img.icons8.com/m_sharp/512/FFFFFF/github.png" alt="GitHub" width="50" height="50">
            </a>
        </div>
        """
        st.markdown(footer_html, unsafe_allow_html=True)

    # --- Main App ---

    # Handle connection logic triggered from sidebar
    if st.session_state.connecting:
        with st.spinner('Connecting...'):
            time.sleep(1) # Delay for UX
            graph, driver = init_neo4j_connection()
            if graph and driver:
                st.session_state.graph = graph
                st.session_state.driver = driver
                st.session_state.db_connected = True
                st.session_state.candidates = get_candidates(driver)
            else:
                st.session_state.db_connected = False
            st.session_state.connecting = False
        st.rerun()

    # Handle disconnection logic triggered from sidebar
    if st.session_state.disconnecting:
        with st.spinner('Disconnecting...'):
            time.sleep(1) # Delay for UX
            # Clear session state
            st.session_state.db_connected = False
            st.session_state.graph = None
            st.session_state.driver = None
            st.session_state.candidates = []
            st.session_state.chat_history = []
            init_neo4j_connection.clear()
            st.session_state.disconnecting = False
        st.rerun()


    if not st.session_state.db_connected:
        st.info("Please connect to the Neo4j database using the button in the sidebar to begin.")
        st.stop()

    st.markdown("Ask any question about a resume, and the system will query the knowledge graph to find the answer.")

    if st.session_state.candidates:
        with st.expander("Available Resumes in the Knowledge Graph", expanded=True):
            st.info("You can ask questions about the following people:")
            cols = st.columns(3)
            for i, name in enumerate(st.session_state.candidates):
                cols[i % 3].markdown(f"- {name}")
    else:
        st.warning("Could not find any resumes (nodes with label 'Person') in the database.")
    
    st.markdown("---")

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if "details" in entry:
                 with st.expander("See the behind-the-scenes process"):
                    st.code(f"Identified Subjects: {', '.join(entry['details'].get('subject_nodes', []))}", language="text")
                    st.code(entry['details']['intermediate_steps'].get('cypher_query'), language="cypher")
                    st.json(entry['details']['intermediate_steps'].get('raw_data', ""))


    question = st.chat_input("Ask your question here...")

    if question:
        # Add user question to history and display it
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner('Thinking...'):
            response = process_query(question, st.session_state.graph, st.session_state.driver)
        
        answer_entry = {"question": question}
        
        with st.chat_message("assistant"):
            if response and response.get("status") in ["success", "partial"]:
                answer = response.get("result", "No result found.")
                st.markdown(answer)
                answer_entry["answer"] = answer

                # Store and display the details
                details = {
                    "subject_nodes": response.get("subject_nodes", []),
                    "intermediate_steps": response.get("intermediate_steps", {})
                }
                answer_entry["details"] = details
                with st.expander("See the behind-the-scenes process"):
                    st.code(f"Identified Subjects: {', '.join(details.get('subject_nodes', []))}", language="text")
                    st.code(details['intermediate_steps'].get('cypher_query'), language="cypher")
                    st.json(details['intermediate_steps'].get('raw_data', ""))

            else:
                error_message = response.get("error", "An unknown error occurred.")
                st.error(error_message)
                answer_entry["answer"] = error_message

        st.session_state.chat_history.append(answer_entry)
    
# --- Main Execution Block ---
if __name__ == "__main__":
    main()

