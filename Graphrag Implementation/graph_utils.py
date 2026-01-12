import numpy as np

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

def resolve_entity(name, cache, graph, allow_updates = True) -> str:
    name = name.lower().strip()
    if name in cache: 
        return cache[name]
    query = generate_full_text_query(name)
    cypher = """
    CALL db.index.fulltext.queryNodes("search_index", $query) YIELD node, score
    RETURN node.name AS name, score
    ORDER BY score DESC LIMIT 1
    """
    results = graph.query(cypher, {"query": query})
    if results and results[0]["score"] > 2.00:
        entity_name = results[0]["name"]
        cache[name] = entity_name
        print(f"[RESOLVED] '{name}' → '{entity_name}' (score: {results[0]['score']:.2f})")
        return entity_name

    if allow_updates:
        # No match found, insert placeholder node to be updated
        label = "Unknown"
        entity_type = "unspecified"
        category = "Unknown"

        cypher_create = f"""
        MERGE (n:{label} {{name: $name}})
        ON CREATE SET n.type = $entity_type,
            n.category = $category
        RETURN n.type IS NOT NULL AS created
        """
        res = graph.query(cypher_create, {
            "name": name,
            "entity_type": entity_type,
            "category": category
        })

        cache[name] = name
        if res and res[0].get('created', False):
            print(f"[NEW ENTITY] '{name}' added as new (no close match found)")
    print(f"no close match found and no update for '{name}'")
    return name

def resolve_entity_ids(terms: list[str], graph) -> dict: #deprecated, no need for this anymore
    resolved = {}
    import re
    import unicodedata 
    def clean_input(inp):
        return unicodedata.normalize("NFKC", inp).strip().lower().strip('"\'')
    def clean_for_lucene(term):
        term = clean_input(term)
        return re.sub(r'[^a-zA-Z0-9_-]', '', term)
    for term in terms:
        sanitized_term = clean_for_lucene(term)
        if not sanitized_term or len(sanitized_term) < 3:
            print(f"[WARN] Skipping unsafe or too-short term: {repr(term)}")
            continue
        cypher = """
        CALL db.index.fulltext.queryNodes("search_index", $query)
        YIELD node, score
        WHERE score > 1.5
        RETURN id(node) AS id, node.name AS name, score
        ORDER BY score DESC
        LIMIT 1
        """
        query_str = f"{sanitized_term}~2"
        res = graph.query(cypher, {"query": query_str})
        if res:
            resolved[term] = res[0]["id"]
    return resolved

def find_connections_between_names(node_names: list[str], graph) -> list:
    paths = []
    for i, name1 in enumerate(node_names):
        for j, name2 in enumerate(node_names):
            if i < j:
                cypher = """
                MATCH path = (a)-[*1..3]-(b)
                WHERE a.name = $name1 AND b.name = $name2
                RETURN DISTINCT path
                LIMIT 5
                """
                result = graph.query(cypher, {"name1": name1, "name2": name2})
                for record in result:
                    paths.append({"path": record["path"]})
    return paths


def path_signature(path):
    def normalize(element):
        if isinstance(element, dict):
            name = element.get("name", "").strip().lower()
            type_ = element.get("type", "").strip().lower()
            category = element.get("category", "").strip().lower()
            return f"NODE({name}|{type_}|{category})"
        elif isinstance(element, str):
            rel = element.strip().lower().replace("_", " ")
            return f"REL({rel})"
        else:
            return f"UNKNOWN({str(element)})"

    signature_parts = [normalize(el) for el in path]
    return "->".join(signature_parts)
    
def mean_pool_path(path): #pool the nodes and relationships of a path to come up with a single meaningful vector 
        node_embeddings = []
        for p in path: #could be node or relationship 
            if "embedding" in p and p["embedding"] is not None:
                node_embeddings.append(np.array(entity["embedding"]))
         if node_embeddings:
            return np.mean(node_embeddings, axis=0)
        return None

def format_graph_results(graph_results):
    formatted_paths = []
    def clean(text):
        return text.replace("_", " ").strip()
    for i, result in enumerate(graph_results, 1):
        path = result.get("path", [])
        parts = []
        for element in path:
            if isinstance(element, dict):
                name = clean(element.get("name", "UNKNOWN"))
                type_ = element.get("type", "UNKNOWN")
                cat = element.get("category", "")
                parts.append(f"\"{name}\" ({type_}, {cat})")
            elif isinstance(element, str):
                rel = clean(element)
                parts.append(f"-> {rel} ->")
        formatted_paths.append(f"{i}. " + " ".join(parts))
    return "\n".join(formatted_paths)


import re, ast

def get_bash_chat_v2(user_prompt: str):  
    """
    Parse one EDUHints-style sample (single row) and return:
      - bash_commands: list[str]
      - chat_messages: list[str]
      - chat_attempts: list[str]
      - background: str
    """

    def extract_array(label: str):
        # Grab the smallest [...] after label:
        m = re.search(
            rf"{re.escape(label)}\s*:\s*(\[[\s\S]*?\])",
            user_prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return []
        raw = m.group(1)
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []

    def extract_background(label: str):
        # Capture everything after "Background scenario context:" to the end
        m = re.search(
            rf"{re.escape(label)}\s*:\s*(.*)\s*$",
            user_prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return (m.group(1).strip() if m else "")

    bash_entries    = extract_array("MOST RECENT COMMANDS (analyze these first)")
    chat_entries    = extract_array("RECENT QUESTIONS/DISCUSSION")
    attempt_entries = extract_array("Previous attempts")
    background      = extract_background("Background scenario context")

    bash_commands = [e.get("bashEntry")      for e in bash_entries    if isinstance(e, dict) and "bashEntry"      in e]
    chat_messages = [e.get("chatEntry")      for e in chat_entries    if isinstance(e, dict) and "chatEntry"      in e]
    chat_attempts = [e.get("responsesEntry") for e in attempt_entries if isinstance(e, dict) and "responsesEntry" in e]

    return {
        "bash_commands": bash_commands,
        "chat_messages": chat_messages,
        "chat_answers": chat_attempts,
        "background": background,
    }

