import pandas as pd
from langchain_neo4j import Neo4jGraph
import os
from sentence_transformers import SentenceTransformer

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LewisAndClark"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
graph = Neo4jGraph(enhanced_schema=True)

#from https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663 
def generate_full_text_query(input: str) -> str:  
    """Generate a fuzzy-matching query for full-text search."""
    def remove_lucene_chars(text):
        return ''.join(c for c in text if c not in r'+-=&|><!(){}[]^"~*?:\\/')
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return input
    fuzzy = ' AND '.join([f'{word}~2' for word in words])
    return fuzzy

def resolve_entity(name: str) -> str:
    global resolved_count, new_count

    name = name.lower().strip()
    if name in resolved_entities_cache: 
        return resolved_entities_cache[name]
    query = generate_full_text_query(name)
    cypher = """
    CALL db.index.fulltext.queryNodes("search_index", $query) YIELD node, score
    RETURN node.name AS name, score
    ORDER BY score DESC LIMIT 1
    """
    results = graph.query(cypher, {"query": query})
    if results and results[0]["score"] > 1.50:
        entity_name = results[0]["name"]
        resolved_entities_cache[name] = entity_name
        resolved_count += 1
        print(f"[RESOLVED] '{name}' → '{entity_name}' (score: {results[0]['score']:.2f})")
        return entity_name

    # No match found, insert placeholder node to be updated
    label = "Unknown"
    entity_type = "unspecified"
    category = "Unknown"
    embeding = embedding_model.encode(name)

    cypher_create = f"""
    MERGE (n:{label} {{name: $name}})
    SET n.type = $entity_type,
        n.category = $category,
        n.embedding = $embedding

    """
    graph.query(cypher_create, {
        "name": name,
        "entity_type": entity_type,
        "category": category, 
        "embedding": embeding
    })

    resolved_entities_cache[name] = name
    new_count += 1
    print(f"[NEW ENTITY] '{name}' added as new (no close match found)")
    return name

def get_label_and_feature(name: str):
    result = graph.query(
        "MATCH (n {name: $name}) RETURN labels(n)[0] AS label, n.type AS type LIMIT 1",
        {"name": name}
    )
    if result:
        return result[0]["label"], result[0]["type"]
    return None, None

def insert_triple(e1, rel, e2):
    rel = rel.lower().strip()
    e1_resolved = resolve_entity(e1)
    e2_resolved = resolve_entity(e2)
    
    e1_label, e1_feature = get_label_and_feature(e1_resolved)
    e2_label, e2_feature = get_label_and_feature(e2_resolved)

    if not e1_label or not e2_label:
        print(f"[SKIPPED] One of the entities labels not found in graph: {e1_resolved}, {e2_resolved}")
        return
    triples_key = (e1_resolved, rel, e2_resolved)
    if triples_key in seen_triples:
        print(f"[SKIPPED] Duplicate triple: {triples_key}")
        return
    seen_triples.add(triples_key) 

    cypher = f"""
    MERGE (a:{e1_label} {{name: $e1}})
    SET a.type = $e1_feature, a.category = $e1_label
    MERGE (b:{e2_label} {{name: $e2}})
    SET b.type = $e2_feature, b.category = $e2_label
    MERGE (a)-[r:`{rel.replace(" ", "_")}`]->(b)
    """
    graph.query(cypher, {
        "e1": e1_resolved, "e2": e2_resolved,
        "e1_feature": e1_feature or "",
        "e2_feature": e2_feature or "", 
        "e1_label": e1_label, "e2_label": e2_label
    }) #pass in parameter dictionary



seen_triples = set()
resolved_entities_cache = {}
resolved_count = 0
new_count = 0
row_count = 0
if __name__ == "__main__":
    
    df = pd.read_csv('langchain/new_clean_triples.csv')
    for _, row in df.iterrows():
        row_count +=1
        insert_triple(row['e1'], row['r'], row['e2'])

    print(f"\DEBUG:")
    print(f"row count: {row_count}")
    print(f"Resolved: {resolved_count}")
    print(f"New Entities: {new_count}")
    print(f"Unique relationships inserted: {len(seen_triples)}")
