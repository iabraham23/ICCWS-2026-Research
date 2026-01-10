import os
from langchain_neo4j import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LewisAndClark"

graph = Neo4jGraph(enhanced_schema=True)
graph.refresh_schema()
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
BATCH_SIZE = 500

def format_node_text(record):
    parts = []
    for key in ["name", "type", "category", "description"]:
        value = record[key]
        if value is not None:
            parts.append(str(value))
    return " | ".join(parts)

def format_relationship_text(record):
    return f"{record['start_name']} | {record['rel_type']} | {record['end_name']}"

def embed_relationships():
    result = graph.query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.name IS NOT NULL AND b.name IS NOT NULL AND r.embedding IS NULL
        RETURN id(r) AS rel_id, type(r) AS rel_type, a.name AS start_name, b.name AS end_name
        """
    )
    relationships = [record for record in result]
    total = len(relationships)
    print(total)
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding Relationships"):
        batch_rels = relationships[i:i + BATCH_SIZE]
        texts = [format_relationship_text(r) for r in batch_rels]
        embeddings = embedding_model.embed_documents(texts)

        for rel, embedding in tqdm(zip(batch_rels, embeddings), desc="graph insert"):
            graph.query(
                """
                MATCH ()-[r]->() WHERE id(r) = $rel_id
                SET r.embedding = $embedding
                """,
                {"rel_id": rel["rel_id"], "embedding": embedding}
            )


def embed_nodes():
    result = graph.query(
        """
        MATCH (n)
        WHERE n.name IS NOT NULL AND n.embedding IS NULL
        RETURN n.name AS name, n.type AS type, n.category AS category, n.description AS description
        """
    )
    nodes = [record for record in result]
    total = len(nodes)
    print(total)
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding"):
        batch_nodes = nodes[i:i + BATCH_SIZE]
        texts = [format_node_text(n) for n in batch_nodes]
        embeddings = embedding_model.embed_documents(texts)  # returns list of vectors

        for node, embedding in tqdm(zip(batch_nodes, embeddings), desc="graph insert"):
            graph.query( 
                """
                MATCH (n) WHERE n.name = $name
                SET n.embedding = $embedding
                """,
                {"name": node["name"], "embedding": embedding}
            )

embed_nodes()
embed_relationships()
