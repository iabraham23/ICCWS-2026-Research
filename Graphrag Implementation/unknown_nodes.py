import os
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
import math
import json
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LewisAndClark"

graph = Neo4jGraph(enhanced_schema=True)
graph.refresh_schema()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#define ontology, note: under the top level categories: concept, application, role
#interestingly anamoly seems like a good addition to the concept types list 
CONCEPT_TYPES = ["feature", "function", "attack", "vulnerability", "technique", "data"]
APPLICATION_TYPES = ["tool", "system", "app"]
ROLE_TYPES = ["attacker", "securityTeam", "user"]



def get_unknown_nodes(graph):
    cypher = """
    MATCH (n:Unknown)
    RETURN n.name AS name, n.type AS type, n.category AS category 
    """
    return(graph.query(cypher))
#print(unknown_nodes) #returns a list of dicts, keys: "name", "type", "category"
#print(len(unknown_nodes))

#note: we use {{}} to escape the normal variable behavior of {} 
FEW_SHOT = """
Example classifications:
- Input: "ls" : {{"name": "ls", "category": "Concept", "type": "function"}} 
- Input: "nmap" : {{"name": "nmap", "category": "Application", "type": "tool"}}
- Input: "phishing" : {{"name": "phishing", "category": "Concept", "type": "attack"}}
- Input: "admin" : {{"name": "admin", "category": "Role", "type": "user"}}
- Input: "firewall" : {{"name": "firewall", "category": "Application", "type": "system"}}

Ontology labels:
- Concept: {concepts}
- Application: {applications}
- Role: {roles}
"""


batched_prompt = ChatPromptTemplate.from_messages([
    ("system", FEW_SHOT+"\n You are a cybersecurity ontology classifier. Classify each of the following terms" ), 
    ("human",
    """
    Classify each term into a dictionary with:
    - "name": the term
    - "category": one of ["Concept", "Application", "Role"]
    - "type": one of the corresponding subtypes

    Ontology labels:
    - Concept: {concepts}
    - Application: {applications}
    - Role: {roles}

    Terms:
    {terms}

    Return the results as a **valid JSON array of dictionaries**. Do not include explanations or comments. The JSON must be directly parsable with `json.loads()`.

    """ ) ])

def format_terms(terms_ls):
    return "\n".join(f"- {term}" for term in terms_ls)

def full_prompt(formatted_terms):
    prompt_text = batched_prompt.format_messages(
    terms=formatted_terms,
    concepts=CONCEPT_TYPES,
    applications=APPLICATION_TYPES,
    roles=ROLE_TYPES
    )
    return prompt_text


def run_update(graph):
    unknown_nodes = get_unknown_nodes(graph=graph)
    batch_list = [[] for _ in range(math.ceil(len(unknown_nodes)/10))] #batches of 10 nodes max 
    if not unknown_nodes:
        return 
    for i,node in enumerate(unknown_nodes):
        index = i // 10
        batch_list[index].append(node["name"])

    response_list = []
    for batch in batch_list:
        terms = format_terms(batch)
        response = llm.invoke(full_prompt(terms))
        response_list.append(response)

    for batch_response in response_list:
        try:
            parsed = json.loads(batch_response.content)
            print(parsed)
            for node in parsed:
                label = node["category"]
                node_embedding = embedding_model.encode(node["name"])
                graph.query(
            f"""
            MATCH (n)
            WHERE toLower(n.name) = toLower($name)
            REMOVE n:Unknown
            REMOVE n:Concept
            REMOVE n:Application
            REMOVE n:Role
            SET n:`{label}`
            SET n.type = $type
            SET n.category = $category
            SET n.embedding = $embedding
            """,
            params={"name": node["name"], "type": node["type"], "category": node["category"], "embedding": node_embedding}
        )
        except json.JSONDecodeError as err:
            print("Failed to parse JSON:", batch_response.content)
    

if __name__ == "__main__":
    graph = Neo4jGraph(enhanced_schema=True)
    graph.refresh_schema()
    run_update(graph)


