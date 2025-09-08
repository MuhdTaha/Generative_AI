# --------------------------------------
# Author: Muhammad Taha
# Project 1: Knowledge Graph for Automated Resume Screening
# --------------------------------------

import os
import PyPDF2
from google import genai
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from typing import List, Tuple
import json

# --------------------------------------
# Load environment variables
# --------------------------------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --------------------------------------
# Enums & Schema
# --------------------------------------
class DocClass(Enum):
    RESUME = "resume"
    SCIENCE_ARTICLE = "science_article"

class ContentSchema(BaseModel):
    entities: List[str] = Field(..., description="Extracted unique entities")
    triples: List[Tuple[str, str, str]] = Field(..., description="List of (source, relationship, target) triples")
    root_entity_name: str = Field(..., description="Root entity name")

# --------------------------------------
# Neo4j Connection Wrapper
# --------------------------------------
class Neo4jConnection:
    def __init__(self, uri, username, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        print("✅ Connected to Neo4j")

    def close(self):
        if self.driver:
            self.driver.close()
            print("🔒 Neo4j connection closed")

    def write_transaction(self, query, parameters=None):
        with self.driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, parameters or {}))

    def query(self, query, parameters=None):
        with self.driver.session(database=self.database) as session:
            return list(session.run(query, parameters or {}))

# --------------------------------------
# PDF Utilities
# --------------------------------------
def extract_text_from_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# --------------------------------------
# Neo4j Utilities
# --------------------------------------
def get_all_relationship_types(driver):
    query = "CALL db.relationshipTypes()"
    with driver.session() as session:
        results = session.run(query)
        return [record["relationshipType"] for record in results]

# --------------------------------------
# Gemini LLM Call
# --------------------------------------
def call_gemini(prompt: str) -> str:
    client = genai.Client()
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip().replace("```json", "").replace("```", "")

# --------------------------------------
# Main Pipeline
# --------------------------------------
def main():
    # 1. Connect to Neo4j
    neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    # 2. Folder containing resumes
    resumes_folder = "resumes"
    doc_class = DocClass.RESUME

    # 3. Fetch existing graph context once
    all_relationships = get_all_relationship_types(neo4j_conn.driver)

    # 4. Loop through all PDF files in the folder
    for filename in os.listdir(resumes_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(resumes_folder, filename)
            full_text = extract_text_from_pdf(pdf_path)

            # Generate root_file_name automatically from filename
            root_file_name = os.path.splitext(filename)[0].replace("_", " ")

            print(f"\nProcessing '{filename}' -> File Name: '{root_file_name}'")

            # --- Gemini LLM Call ---
            template = """
            IMPORTANT: Ensure your output is valid JSON with all keys and string values enclosed in double quotes.
            You are a document parser specialized in resumes. Your task is to analyze the given resume and extract key information 
            to construct a knowledge graph representing the person’s profile.

            Your output must **always** include the person’s name as the root entity, and all entities must be connected directly
            or indirectly to this root entity. If a relationship is missing, create a generic connection to the root entity.

            - Document Text: {text}
            - Document Class: {doc_class} (always 'resume').
            - Existing Relationships: {all_relationships}

            Follow this schema:
            - Entities: Person Name, Contact Info, Education (Degree, Major, Minor, University), Work Experience (Company, Role), Skills, Projects, Certifications, Awards, Locations.
            - Triples: Use clear relationships, e.g., 
                Person -> HAS_EDUCATION -> Degree
                Degree -> HAS_MAJOR -> Major
                Degree -> HAS_MINOR -> Minor
                Degree -> AT_UNIVERSITY -> University
                Person -> HAS_EXPERIENCE -> Company/Role
                Person -> HAS_SKILLS -> Skill
                Person -> HAS_CONTACT_INFO -> Contact Info
                Person -> AWARDED -> Award/Recognition
                Person -> HAS_PROJECTS -> Project/Research
                Person -> LIVES_IN -> Location
                Company/Project -> USED_TECH -> Skill/Tool/Technology

            Make sure **all entities** are connected to the root entity directly or indirectly.
            Return the output in this exact JSON format:
            {{
                "entities": [...],
                "triples": [
                    ["source_entity", "RELATIONSHIP_TYPE", "target_entity"],
                    ...
                ],
                "root_entity_name": "Person Name"
            }}
            """

            final_prompt = template.format(text=full_text, all_relationships=all_relationships, doc_class=doc_class.value)
            gemini_output = call_gemini(final_prompt)

            # --- Parse JSON output ---
            try:
                response_dict = json.loads(gemini_output)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse Gemini output for {filename}: {e}")
                continue

            # Extract entities
            raw_entities = response_dict.get("entities", [])
            combined_entities = set()
            for ent in raw_entities:
                if isinstance(ent, dict) and "name" in ent:
                    combined_entities.add(ent["name"])
                elif isinstance(ent, str):
                    combined_entities.add(ent)

            # Extract triples
            combined_triples = response_dict.get("triples", [])

            # Determine root entity
            root_entity_node = response_dict.get("root_entity_name") or root_file_name

            # Ensure all entities are connected to root
            for entity in combined_entities:
                if entity != root_entity_node and not any(
                    entity in (t[0], t[2]) for t in combined_triples
                ):
                    combined_triples.append([root_entity_node, "RELATED_TO", entity])

            # --- Insert into Neo4j ---
            # Insert File node
            neo4j_conn.write_transaction("MERGE (f:File {name: $file_name})", {"file_name": root_file_name})

            # Link File to Document Class
            belongs_query = """
            MERGE (r:resume {name: 'resume'})
            MERGE (f:File {name: $file_name})
            MERGE (f)-[:BELONGS_TO]->(r)
            """
            neo4j_conn.write_transaction(belongs_query, {"file_name": root_file_name})

            # Insert Entities
            for entity in combined_entities:
                neo4j_conn.write_transaction("MERGE (e:Entity {name: $name})", {"name": entity})

            # Insert Relationships
            for source, relationship, target in combined_triples:
                sanitized_relationship = ''.join(filter(str.isalnum, relationship.replace(" ", "_"))).upper()
                if not sanitized_relationship:
                    sanitized_relationship = "RELATED_TO"
                
                query = f"""
                MATCH (a:Entity {{name: $source}}), (b:Entity {{name: $target}})
                MERGE (a)-[r:{sanitized_relationship}]->(b)
                """
                neo4j_conn.write_transaction(query, {"source": source, "target": target})

            # Link Root Entity to File
            root_entity_query = """
            MATCH (b:Entity {name: $root_name}), (f:File {name: $file_name})
            MERGE (b)-[:HAS_FILE]->(f)
            """
            neo4j_conn.write_transaction(root_entity_query, {"root_name": root_entity_node, "file_name": root_file_name})

            print(f"✅ Processed '{filename}' successfully. Entities: {len(combined_entities)}, Relationships: {len(combined_triples)}")

    # Cleanup
    neo4j_conn.close()
    print("\n✅ All resumes processed.")


if __name__ == "__main__":
    main()
