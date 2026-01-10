from neo4j import GraphDatabase
import pandas as pd

#ontology and data from AISeck paper: https://par.nsf.gov/servlets/purl/10401616 

uri = "bolt://localhost:7687"
username = "neo4j"
password = "LewisAndClark"
csv_path = "langchain/all_entity_info.csv"
driver = GraphDatabase.driver(uri, auth=(username, password)) #connect to db 

#define ontology, note: under the top level categories: concept, application, role
CONCEPT_TYPES = {"feature", "function", "attack", "vulnerability", "technique", "data"}
APPLICATION_TYPES = {"tool", "system", "app"}
ROLE_TYPES = {"attacker", "securityTeam", "user"}

df = pd.read_csv(csv_path)

def get_category(type_):
    if pd.isna(type_): return "Unknown"
    t = type_.lower()
    if t in CONCEPT_TYPES: return "Concept" #if within this group we assign it to the top level category 
    if t in APPLICATION_TYPES: return "Application"
    if t in ROLE_TYPES: return "Role"
    return "Unknown"

def create_entity(tx, label, name, entity_type, category, description):
    query = f"""
    MERGE (n:{label} {{name: $name}})
    SET n.type = $entity_type,
        n.category = $category,
        n.description = $description
    """
    tx.run(query, name=name, entity_type=entity_type, category=category, description=description or "")

with driver.session() as session:
    for _, row in df.iterrows():
        name = row["entityName"]
        entity_type = row["entityType"]
        category = row["entityCategory"]
        description = row.get("entityDescription", "")
        # Only process valid rows
        if pd.notna(name) and pd.notna(category):
            label = category.strip().capitalize()
            session.execute_write(create_entity, label, name, entity_type, category, description)

with driver.session() as session:
    session.run("DROP INDEX search_index IF EXISTS")
    session.run("""
        CREATE FULLTEXT INDEX search_index IF NOT EXISTS
        FOR (n:Concept|Application|Role)
        ON EACH [n.name]
    """)
driver.close() 