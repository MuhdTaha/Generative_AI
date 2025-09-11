# --------------------------------------
# Author: Muhammad Taha
# Project 1: Knowledge Graph for Automated Resume Screening (Streamlit UI)
# Video Link: https://1drv.ms/v/c/017e2386548f0457/ESIGXTBICO9FnReOEIR5-fIB2opsQy1ybt2i_v_G2GjTug
# --------------------------------------

import os
import PyPDF2
from google import genai
from dotenv import load_dotenv
from neo4j import GraphDatabase
import json
import streamlit as st
import tempfile

from streamlit_agraph import agraph, Node, Edge, Config

# --------------------------------------
# Load environment variables
# --------------------------------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- Define a list of valid node labels for security and consistency ---
VALID_LABELS = ["Person", "Skill", "Company", "University", "Project", "Certification", "Contact_Info", "Location", "Degree", "Job_Title"]

# --------------------------------------
# Helper function for label truncation
# --------------------------------------
def truncate_label(label, length=15):
    """Truncates a string to a given length and adds an ellipsis if it's too long."""
    if len(label) > length:
        return f"{label[:length]}..."
    return label

# --------------------------------------
# Neo4j Connection Wrapper
# --------------------------------------
class Neo4jConnection:
    def __init__(self, uri, username, password, database="neo4j"):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.database = database
            st.session_state.neo4j_connected = True
            print("✅ Connected to Neo4j")
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}", icon="🚨")
            st.session_state.neo4j_connected = False

    def close(self):
        if self.driver:
            self.driver.close()
            st.session_state.neo4j_connected = False
            print("🔒 Neo4j connection closed")

    def write_transaction(self, query, parameters=None):
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, parameters or {}))

    def get_all_relationship_types(self):
        query = "CALL db.relationshipTypes()"
        with self.driver.session() as session:
            results = session.run(query)
            return [record["relationshipType"] for record in results]
    
    def fetch_graph_data(self):
        """Fetches all nodes and relationships for visualization with improved styling based on labels."""
        query = "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m"
        nodes = {}
        edges = set() 

        label_colors = {
            "Person": "#FF6347",      # Tomato Red
            "Company": "#4682B4",     # Steel Blue
            "Skill": "#32CD32",       # Lime Green
            "University": "#FFD700",  # Gold
            "Project": "#6A5ACD",      # Slate Blue
            "Job_Title": "#40E0D0",   # Turquoise
            "File": "#D3D3D3",        # Light Grey
            "DEFAULT": "#808080"      # Grey
        }

        with self.driver.session(database=self.database) as session:
            results = session.run(query)
            for record in results:
                for node_type in ["n", "m"]:
                    node_record = record[node_type]
                    if node_record and node_record.element_id not in nodes:
                        full_name = node_record.get("name", "Unknown")
                        short_name = truncate_label(full_name)
                        node_labels = list(node_record.labels)
                        
                        # Determine primary label for coloring and sizing
                        primary_label = next((label for label in VALID_LABELS if label in node_labels), 
                                             next(iter(node_labels), "DEFAULT"))

                        color = label_colors.get(primary_label, label_colors["DEFAULT"])
                        size = 25 if primary_label == "Person" else 15

                        nodes[node_record.element_id] = Node(id=node_record.element_id, 
                                                            label=short_name, 
                                                            title=full_name,
                                                            size=size,
                                                            color=color,
                                                            font={'color': 'white', 'size': 12})
                
                if record["r"] is not None and record["m"] is not None:
                    source_id = record["n"].element_id
                    target_id = record["m"].element_id
                    rel_type = record["r"].type
                    edge_tuple = (source_id, target_id, rel_type)
                    edges.add(edge_tuple)

        edge_objects = [Edge(source=s, target=t, label=l, font={'color': 'white', 'size': 8}) for s, t, l in edges]
        return list(nodes.values()), edge_objects

# --------------------------------------
# PDF & AI Utilities
# --------------------------------------
def extract_text_from_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def call_gemini(prompt: str) -> str:
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip().replace("```json", "").replace("```", "")
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}", icon="🤖")
        return None

# --------------------------------------
# Core Processing Logic
# --------------------------------------
def process_and_ingest_resume(neo4j_conn, full_text, filename, all_relationships):
    root_file_name = os.path.splitext(filename)[0].replace("_", " ")

    template = """
    IMPORTANT: Ensure your output is valid JSON. All keys and string values must be in double quotes.
    You are an expert data extractor specializing in resumes. Your task is to analyze the resume text and convert it into a structured knowledge graph.

    - Document Text: {text}
    - Existing Relationship Types: {all_relationships}

    Instructions:
    1.  Identify the single main person (the candidate) in the resume. This is your root entity.
    2.  Extract all other relevant entities, such as skills, companies, job titles, universities, degrees, etc.
    3.  **Crucially, assign a label to each entity.** Use one of the following valid labels: {valid_labels}.
    4.  Generate relationships (triples) connecting these entities. All entities must eventually connect back to the root 'Person' entity.

    Return the output in this exact JSON format:
    {{
      "root_entity_name": "Full Name of the Candidate",
      "entities": [
        {{"name": "Entity Name 1", "label": "EntityLabel1"}},
        {{"name": "Entity Name 2", "label": "EntityLabel2"}},
        ...
      ],
      "triples": [
        ["source_entity_name", "RELATIONSHIP_TYPE", "target_entity_name"],
        ...
      ]
    }}
    """

    final_prompt = template.format(text=full_text, all_relationships=all_relationships, valid_labels=VALID_LABELS)
    gemini_output = call_gemini(final_prompt)

    if not gemini_output:
        st.warning(f"AI model did not return a response for {filename}. Skipping.", icon="⚠️")
        return

    try:
        response_dict = json.loads(gemini_output)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI output for {filename}: {e}", icon="❌")
        st.code(gemini_output)
        return

    entities = response_dict.get("entities", [])
    triples = response_dict.get("triples", [])
    root_entity_name = response_dict.get("root_entity_name")

    if not root_entity_name or not entities or not triples:
        st.warning(f"AI output for {filename} was missing key information (root entity, entities, or triples). Skipping.", icon="⚠️")
        return
    
    # Ingest all entities with their specific labels
    for entity in entities:
        name = entity.get("name")
        label = entity.get("label")
        # Security check: Only use labels from our predefined valid list
        if name and label in VALID_LABELS:
            # The root entity is special - we ensure it is labeled as 'Person'
            if name == root_entity_name:
                query = "MERGE (p:Person {name: $name})"
            else:
                query = f"MERGE (e:{label} {{name: $name}})"
            neo4j_conn.write_transaction(query, {"name": name})

    # Ingest all relationships
    for source, relationship, target in triples:
        sanitized_relationship = ''.join(filter(str.isalnum, relationship.replace(" ", "_"))).upper()
        if not sanitized_relationship: 
            sanitized_relationship = "RELATED_TO"
        
        query = f"""
        MATCH (a {{name: $source}})
        MATCH (b {{name: $target}})
        MERGE (a)-[:{sanitized_relationship}]->(b)
        """
        neo4j_conn.write_transaction(query, {"source": source, "target": target})

    # Link the Person node to a File node representing this resume
    neo4j_conn.write_transaction("MERGE (f:File {name: $file_name})", {"file_name": root_file_name})
    root_entity_query = """
    MATCH (p:Person {name: $root_name})
    MATCH (f:File {name: $file_name})
    MERGE (p)-[:HAS_FILE]->(f)
    """
    neo4j_conn.write_transaction(root_entity_query, {"root_name": root_entity_name, "file_name": root_file_name})

    st.success(f"Processed '{filename}'. Added {len(entities)} entities and {len(triples)} relationships.", icon="✅")

# --------------------------------------
# Streamlit UI
# --------------------------------------
st.set_page_config(page_title="Resume Knowledge Graph Builder", page_icon="🧠", layout="wide")

st.title("🧠 AI-Powered Resume Knowledge Graph Builder")
st.markdown("""
This tool uses Google's Gemini AI to read PDF resumes, extract key information, and build an interconnected knowledge graph in Neo4j.
1.  **Upload** one or more resumes below.
2.  **Click** the 'Build Knowledge Graph' button to process them.
3.  **Click** the 'Show/Refresh Graph' button to visualize the database.
""")

uploaded_files = st.file_uploader("Choose resume PDF files", type="pdf", accept_multiple_files=True)

if st.button("Build Knowledge Graph", type="primary", use_container_width=True, disabled=(not uploaded_files)):
    with st.spinner("Connecting to database and processing resumes... Please wait."):
        neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        if st.session_state.get("neo4j_connected", False):
            all_relationships = neo4j_conn.get_all_relationship_types()

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                full_text = extract_text_from_pdf(tmp_file_path)
                process_and_ingest_resume(neo4j_conn, full_text, uploaded_file.name, all_relationships)
                os.remove(tmp_file_path)

            st.success("🎉 All resumes have been processed and added to the knowledge graph!")
            neo4j_conn.close()
        else:
            st.error("Could not process files because the database connection failed.", icon="💔")

st.markdown("---")

if st.button("Show/Refresh Knowledge Graph", use_container_width=True):
    st.subheader("Interactive Knowledge Graph")
    with st.spinner("Connecting to database and fetching graph data..."):
        neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        if st.session_state.get("neo4j_connected", False):
            nodes, edges = neo4j_conn.fetch_graph_data()
            if nodes:
                config = Config(width=1200, 
                                height=800, 
                                directed=True, 
                                interaction={
                                  "hover": True,
                                  "navigationButtons": True,
                                  "tooltipDelay": 300,
                                  "hideEdgesOnDrag": True,
                                  "hideNodesOnDrag": False
                                },
                                physics={
                                    "enabled": True,
                                    "forceAtlas2Based": {
                                        "gravitationalConstant": -100,
                                        "centralGravity": 0.01,
                                        "springLength": 400,
                                        "springConstant": 0.18,
                                        "avoidOverlap": 1
                                    },
                                    "minVelocity": 0.75,
                                    "solver": "forceAtlas2Based",
                                    "stabilization": {
                                        "enabled": True,
                                        "iterations": 400,
                                        "fit": True
                                    }
                                })
                
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.warning("No data found in the graph. Please process some resumes first!", icon="🕸️")
            neo4j_conn.close()
        else:
            st.error("Database connection failed. Cannot display graph.", icon="💔")

# --------------------------------------
# Footer
# --------------------------------------
st.markdown("---")
footer_html = """
<div style="text-align: center; margin-top: 20px;">
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