# --------------------------------------
# Author: Muhammad Taha
# Project 1: Knowledge Graph for Automated Resume Screening (Streamlit UI)
# Video Link: https://1drv.ms/v/c/017e2386548f0457/ESIGXTBICO9FnReOEIR5-fIB2opsQy1ybt2i_v_G2GjTug
# --------------------------------------

import os
import PyPDF2
from google import genai
from enum import Enum
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
        """Fetches all nodes and relationships for visualization with improved styling."""
        query = "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m"
        nodes = {}
        edges = set() # Use a set to avoid duplicate edges

        # Define colors for different node labels for better visual distinction
        label_colors = {
            "File": "#FF6347",    # Tomato Red
            "Resume": "#4682B4",  # Steel Blue
            "Entity": "#32CD32",  # Lime Green
            "DEFAULT": "#808080"  # Grey
        }

        with self.driver.session(database=self.database) as session:
            results = session.run(query)
            for record in results:
                # Process the source node
                source_node = record["n"]
                if source_node.element_id not in nodes:
                    full_name = source_node.get("name", "Unknown")
                    short_name = truncate_label(full_name)
                    node_labels = list(source_node.labels)
                    primary_label = next((label for label in ["File", "Resume"] if label in node_labels), "Entity")
                    color = label_colors.get(primary_label, label_colors["DEFAULT"])
                    size = 25 if primary_label in ["File", "Resume"] else 15

                    nodes[source_node.element_id] = Node(id=source_node.element_id, 
                                                         label=short_name, 
                                                         title=full_name,
                                                         size=size,
                                                         color=color,
                                                         font={'color': 'white', 'size': 12})

                # Process relationship and target node if they exist
                if record["r"] is not None and record["m"] is not None:
                    target_node = record["m"]
                    relationship = record["r"]

                    if target_node.element_id not in nodes:
                        full_name = target_node.get("name", "Unknown")
                        short_name = truncate_label(full_name)
                        node_labels = list(target_node.labels)
                        primary_label = next((label for label in ["File", "Resume"] if label in node_labels), "Entity")
                        color = label_colors.get(primary_label, label_colors["DEFAULT"])
                        size = 25 if primary_label in ["File", "Resume"] else 15
                        
                        nodes[target_node.element_id] = Node(id=target_node.element_id, 
                                                             label=short_name,
                                                             title=full_name,
                                                             size=size,
                                                             color=color,
                                                             font={'color': 'white', 'size': 12})
                    
                    # Add a unique edge tuple to the set
                    edge_tuple = (source_node.element_id, target_node.element_id, relationship.type)
                    edges.add(edge_tuple)

        # Convert the set of edge tuples to Edge objects with styling
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
            text += page.extract_text()
    return text

def call_gemini(prompt: str) -> str:
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="models/gemini-1.5-flash",
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
    # (This function remains the same as your previous version)
    root_file_name = os.path.splitext(filename)[0].replace("_", " ")

    template = """
    IMPORTANT: Ensure your output is valid JSON with all keys and string values enclosed in double quotes.
    You are a document parser specialized in resumes. Your task is to analyze the given resume and extract key information 
    to construct a knowledge graph representing the person’s profile.

    Your output must **always** include the person’s name as the root entity, and all entities must be connected directly
    or indirectly to this root entity. If a relationship is missing, create a generic connection to the root entity.

    - Document Text: {text}
    - Existing Relationships: {all_relationships}

    Follow this schema:
    - Entities: Person Name, Contact Info, Education, Work Experience, Skills, Projects, etc.
    - Triples: Use clear relationships like HAS_EDUCATION, HAS_SKILL, WORKED_AT, etc.
    
    Return the output in this exact JSON format:
    {{
        "entities": ["Entity 1", "Entity 2", ...],
        "triples": [
            ["source_entity", "RELATIONSHIP_TYPE", "target_entity"],
            ...
        ],
        "root_entity_name": "Person Name"
    }}
    """

    final_prompt = template.format(text=full_text, all_relationships=all_relationships)
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

    combined_entities = set(response_dict.get("entities", []))
    combined_triples = response_dict.get("triples", [])
    root_entity_node = response_dict.get("root_entity_name") or root_file_name

    connected_entities = {root_entity_node}
    for s, _, t in combined_triples:
        connected_entities.add(s)
        connected_entities.add(t)
    
    for entity in combined_entities:
        if entity not in connected_entities:
            combined_triples.append([root_entity_node, "RELATED_TO", entity])

    neo4j_conn.write_transaction("MERGE (f:File {name: $file_name})", {"file_name": root_file_name})
    belongs_query = """
    MERGE (r:Resume {name: 'Resume'})
    MERGE (f:File {name: $file_name})
    MERGE (f)-[:BELONGS_TO]->(r)
    """
    neo4j_conn.write_transaction(belongs_query, {"file_name": root_file_name})

    for entity in combined_entities:
        neo4j_conn.write_transaction("MERGE (e:Entity {name: $name})", {"name": entity})

    for source, relationship, target in combined_triples:
        sanitized_relationship = ''.join(filter(str.isalnum, relationship.replace(" ", "_"))).upper()
        if not sanitized_relationship: sanitized_relationship = "RELATED_TO"
        query = f"MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}}) MERGE (a)-[r:{sanitized_relationship}]->(b)"
        neo4j_conn.write_transaction(query, {"source": source, "target": target})

    root_entity_query = "MATCH (b:Entity {name: $root_name}), (f:File {name: $file_name}) MERGE (b)-[:HAS_FILE]->(f)"
    neo4j_conn.write_transaction(root_entity_query, {"root_name": root_entity_node, "file_name": root_file_name})

    st.success(f"Processed '{filename}'. Added {len(combined_entities)} entities and {len(combined_triples)} relationships.", icon="✅")


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
        <img src="https://static.vecteezy.com/system/resources/previews/018/930/480/non_2x/linkedin-logo-linkedin-icon-transparent-free-png.png" alt="LinkedIn" width="50" height="50">
    </a>
    <a href="https://github.com/MuhdTaha" target="_blank" style="margin: 0 10px;">
        <img src="https://freepnglogo.com/images/all_img/github-logo-white-stroke-2a6c.png" alt="GitHub" width="50" height="50">
    </a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

