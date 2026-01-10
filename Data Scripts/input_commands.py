import os
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph
import pandas as pd

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LewisAndClark"

graph = Neo4jGraph(enhanced_schema=True)
graph.refresh_schema()

#creates an index for our new commands
graph.query("""
CREATE INDEX IF NOT EXISTS FOR (n:nl_bash) ON (n.name)
""")

BATCH_SIZE = 500

#must have APOC installed, which we do 
def batch_insert_triples(triples_batch, dry_run = False):
    if not triples_batch:
        return

    cypher = """
UNWIND $batch AS row
MERGE (nl:nl_bash {name: row.nl})
SET nl.type = row.nl_type
SET nl.category = "natural language"
MERGE (bash:nl_bash {name: row.bash})
SET bash.type = row.bash_type
SET bash.category = "bash command"
WITH nl, bash, row
CALL apoc.merge.relationship(nl, row.relation, {}, {}, bash) YIELD rel
RETURN count(rel)
"""
    if dry_run:
        print("\n--- DRY RUN ---")
        print("Cypher Query:\n", cypher)
        print("Sample Parameters:")
        for r in triples_batch[:3]:
            print(r)
        print("--- END DRY RUN ---\n")
        return
    graph.query(cypher, {"batch": triples_batch})



df = pd.read_csv('langchain/Refined_Bash_Command_Classification.csv')
batch = []
seen_triples = set()
skipped = 0

for _, row in df.iterrows():
    triple_key = (row["nl"].strip().lower().replace(" ", "_"), row["relation"].strip().lower().replace(" ", "_"), row["bash"].strip().lower().replace(" ", "_"))
    if triple_key in seen_triples:
        skipped +=1
        continue
    seen_triples.add(triple_key)

    batch.append({
        "nl": row["nl"].strip().lower().replace(" ", "_"),
        "bash": row["bash"].strip().lower().replace(" ", "_"),
        "nl_type": row["nl_type"].strip().lower().replace(" ", "_"),
        "bash_type": row["bash_type"].strip().lower().replace(" ", "_"),
        "relation": row["relation"].strip().lower().replace(" ", "_")
    })

    if len(batch) >= BATCH_SIZE:
        batch_insert_triples(batch)
        batch = []

#any remaining         
if batch:
    batch_insert_triples(batch)

print(f"Unique relationships inserted: {len(seen_triples)}")
print(f"amount of duplicates: {skipped}")
