#NON QandA version of the hybrid rag, made to work as a mimic to EDUHints

from dotenv import load_dotenv
load_dotenv()
import os 
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_community.llms.llamacpp import LlamaCpp
from graph_utils import * 
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from unknown_nodes import run_update
from sklearn.neighbors import NearestNeighbors
import time

def clean_input(inp):
    return unicodedata.normalize("NFKC", inp).strip().lower().strip('"\'')

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LewisAndClark"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
entity_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

graph = Neo4jGraph(enhanced_schema=True)
graph.refresh_schema()

s = time.time()
all_nodes = graph.query("MATCH (n) WHERE n.embedding IS NOT NULL RETURN n.name AS name, n.embedding AS embedding")
e = time.time()
print(f"DEBUG graph finished querying: {e-s}") #loading this all_nodes takes a lot of time which we don't want to reload  

llm = LlamaCpp(
    model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    temperature=0.0,
    max_tokens=512,
    n_ctx=4096,
    n_threads=8,
    use_mmap=True,
    use_mlock=True,
    verbose=False,
)


#from https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663  
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )
parser = PydanticOutputParser(pydantic_object = Entities)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting cybersecurity related entities (commands, concepts, key phrases, etc...) from a students most recent bash commands and questions"),
        ("human", "Use the given format to extract information from the following input: {question}"),
    ]
)

extract_entities_llm = prompt | entity_llm.with_structured_output(Entities) #we need structured output for this to work


def extract_entities_with_input(input_dict):
    bash_commands = input_dict["bash_commands"]
    chat_messages = input_dict["chat_messages"]
    question = f"A students bash commands: {bash_commands}\n A students chat messages: {chat_messages}"
    entities = extract_entities_llm.invoke({"question": question}) #call our chain to get the entities from question
    entities.names = [name.lower().strip().replace(" ", "_") for name in entities.names]
    for q in chat_messages: #attach all chat messages as entities themselves 
        entities.names.append("_".join(q.split()))
    print(f"DEBUG extracted entities: {entities}")
    return {"entities": entities, "bash_commands": bash_commands, "chat_messages": chat_messages}

extract_entities = RunnableLambda(extract_entities_with_input)

def top_embedded_entities(entity_terms, top_k=5):
    names = [node['name'] for node in all_nodes]
    vectors = np.array([node['embedding'] for node in all_nodes])
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(vectors)
    results = []
    for term in entity_terms:
        vec = embedding_model.encode(term).reshape(1, -1)
        distances, indices = nn_model.kneighbors(vec, n_neighbors=top_k)
        results.extend([names[i] for i in indices[0]])
    return results
def path_signature(path):
    return tuple(
        (element.get("id", element.get("name", "NONE")), element.get("type", "NONE"))
        if isinstance(element, dict) else element
        for element in path
    )

def fuzzy_search(names):
    cypher_exact = """
   MATCH path = (n)-[*1..2]->(m)
WHERE 
  n.name IN $resolved_names 
  OR 
  m.name in $resolved_names
  AND any(label IN labels(n) WHERE label IN ['Concept', 'Application', 'Role', 'nl_bash', 'Unknown']) 
  AND any(label IN labels(m) WHERE label IN ['Concept', 'Application', 'Role', 'nl_bash', 'Unknown'])
RETURN DISTINCT path
LIMIT 20
    """
    exact_results = graph.query(cypher_exact, {"resolved_names": names})
    return exact_results


def mean_pool_path(path): #pool the nodes of a path to come up with a single meaningful vector 
    node_embeddings = []
    for i in range(0, len(path)): 
        entity = path[i] #could be node or relationship 
        if "embedding" in entity and entity["embedding"] is not None:
            node_embeddings.append(np.array(entity["embedding"]))
    if node_embeddings:
        return np.mean(node_embeddings, axis=0)
    return None

#most important part of the code, where we actually get the information from the graph 
def query_neo4j(input_dict): 
    c={} #keep a cache 
    entities = input_dict["entities"]
    entity_terms = entities.names
    resolved_names = [resolve_entity(name,c, graph, allow_updates=True) for name in entity_terms]
    #classify unknown nodes right here, on the fly classification before retrieval 
    #run_update(graph=graph)
    # resolved_id_map = resolve_entity_ids(entity_terms, graph)
    # node_ids = list(resolved_id_map.values())
    
    graph_results = [
    {**res, "source": "initial_connection"} for res in find_connections_between_names(resolved_names, graph)
    ]

    exact_results = []
    if len(graph_results)<5: #look at lexical AND cosine similarity
        print(f"DEBUG: node connection returned {len(graph_results)} results: falling back to fuzzy and semantic search...")
        exact_results = fuzzy_search(resolved_names)
        exact_results = [{**res, "source": "fuzzy"} for res in exact_results]
        print(f"DONE WITH EXACT RESULTS... len:{len(exact_results)}")
        semantic_nodes = top_embedded_entities(entity_terms)
        print(f"DONE WITH FINDING TOP SEMANTIC NODES. len: {len(semantic_nodes)}")
        semantic_results = fuzzy_search(semantic_nodes)
        semantic_results = [{**res, "source": "semantic"} for res in semantic_results]

    seen_signatures = set(path_signature(res["path"]) for res in graph_results)
    for res in exact_results + semantic_results:
        sig = path_signature(res["path"])
        if sig not in seen_signatures:
            graph_results.append(res)
            seen_signatures.add(sig)

    chat_embedding = [embedding_model.encode(q) for q in input_dict["chat_messages"]]
    scored_paths = []
    for result in graph_results:
        vec = mean_pool_path(result["path"])
        if vec is not None:
            similarities = cosine_similarity([chat_embedding],[vec])
            highest_score = max(similarities[:,0]) #choose highest score 
            scored_paths.append((highest_score, result))

    scored_paths.sort(key=lambda x: x[0], reverse=True) 
    top_paths = [res for _, res in scored_paths[:20]] #ignoring the score 

    #print(f"DEBUG -- entities: {entities} result in {len(top_paths)}: {top_paths}")
    return {"graph_results": top_paths, "bash_commands": input_dict["bash_commands"], "chat_messages": input_dict["chat_messages"]}


query_graph = RunnableLambda(query_neo4j)

#mess around here
summary_prompt = PromptTemplate.from_template(
    "A student is completing a cyber-security scenario, review the scenario's summary, their bash, chat and question/answer history, along with relevant graph data and provide them a single concise hint on what to do next. The hint must not exceed two sentences in length."
    "The student's recent bash commands: {bash_commands}\n The student's recent chat messages: {chat_messages}\n Relevant graph data: {data}"
)

summarize_chain = summary_prompt | llm

def format_graph_results(graph_results):
    formatted_paths = []
    for result in graph_results:
        path = result.get("path", [])
        path_parts = []
        for element in path:
            if isinstance(element, dict):
                name = element.get("name", "UNKNOWN")
                type_ = element.get("type", "UNKNOWN")
                category = element.get("category", "")
                node_repr = f"[{name} | {type_}, {category}]"
                path_parts.append(node_repr)
            elif isinstance(element, str):
                path_parts.append(f"--{element.replace('_', ' ')}-->")
        formatted_paths.append(" ".join(path_parts))
    return "\n".join(f"- {p}" for p in formatted_paths)

def format_for_summary(x): #need to pass in dict with graph_result and question as keys
    graph_results = x["graph_results"]
    data = format_graph_results(graph_results)
    print(data)
    print("\n--- DEBUG Sources of top paths ---")
    for i, result in enumerate(graph_results, 1):
        source = result.get("source", "unknown")
        print(f"{i}. source: {source}")
    return {
        "data": data,
        "bash_commands": x["bash_commands"],
        "chat_messages": x["chat_messages"]
    } #could also unpack using **x but still need to get rid of graph_results or I could keep it along 

def make_input_dict(prompt: str) -> dict: #must take a dict with bash_commands and chat_messages as keys with lists as values
    if isinstance(prompt, dict) and type(prompt["bash_commands"]) == List and type(prompt["chat_messages"]) == List:
        return prompt 
    raise TypeError 
make_input_dict = RunnableLambda(make_input_dict)

graph_chain = RunnableSequence(make_input_dict, extract_entities, query_graph, format_for_summary, summarize_chain)