from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from graph_utils import * 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from unknown_nodes import run_update
from sklearn.neighbors import NearestNeighbors
import time

class GraphRAGPipeline:
    def __init__(self, llm, graph, all_nodes):
        self.llm = llm
        self.graph = graph
        self.all_nodes = all_nodes
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.entity_llm = llm 
        class Entities(BaseModel):
            """Identifying information about entities."""
            names: List[str] = Field(
                ...,
                description="All the person, organization, or business entities that appear in the text",
            )

        # prompt and parser setup
        self.parser = PydanticOutputParser(pydantic_object=Entities)
        self.prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract cybersecurity-related entities as JSON ONLY. Do not include any other text. Output must match this format exactly: {{\"names\": [\"entity1\", \"entity2\"]}}."),
            ("human", "Input: {question}")
        ]
)
        self.extract_entities_llm = self.prompt | self.entity_llm | self.parser 
        self.graph_chain = self._build_graph_chain()
        self.cache = {}
        self.query_database_duration = None


    def _build_graph_chain(self):
        return RunnableSequence(
            RunnableLambda(self._make_input_dict),
            RunnableLambda(self._extract_entities),
            RunnableLambda(self._query_graph),
            RunnableLambda(self._format_for_summary),
            self._build_summarize_chain()  # already a chain
        )

    def _make_input_dict(self, prompt: str) -> dict:
        if isinstance(prompt, dict) and isinstance(prompt["bash_commands"], list) and isinstance(prompt["chat_messages"], list) and isinstance(prompt["chat_answers"], list):
            return prompt 
        raise TypeError("Input must be a dict with 'bash_commands', 'chat_messages', and 'chat_answers' as lists.")

    def _extract_entities(self, input_dict):
        bash = input_dict["bash_commands"]
        chat = input_dict["chat_messages"]
        answers = input_dict["chat_answers"]
        question = f"A students bash commands: {bash}\n A students chat messages: {chat} A students chat answers: {answers}"
        raw_output = (self.prompt | self.entity_llm).invoke({"question": question})
        print("=== RAW ENTITY LLM OUTPUT ===")
        print(repr(raw_output))  # use repr to see things like newlines, whitespace, etc.
        
        #entities = self.extract_entities_llm.invoke({"question": question})
        try:
            # Try parsing
            entities = self.parser.invoke(raw_output)
        except Exception as e:
            print("=== PARSING FAILED ===")
            print(str(e))
            return {
            "__return_direct__": True,
            "output": f"[ERROR] LLM did not return a valid result: {raw_output}"
        }
            

        entities.names = [name.lower().strip().replace(" ", "_") for name in entities.names]
        for q in chat:
            entities.names.append("_".join(q.split()))
        return {
                "__return_direct__": True,
                "output": f"DEBUG: extracted entities: {entities}, chat messages: {chat}, bash commands: {bash}"
            }

        print(f"DEBUG extracted entities: {entities}")
        return {"entities": entities, "bash_commands": bash, "chat_messages": chat, "chat_answers": answers}
    

    def _top_embedded_entities(self, entity_terms, top_k=5):
        names = [node['name'] for node in self.all_nodes]
        vectors = np.array([node['embedding'] for node in self.all_nodes])
        nn_model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(vectors)
        results = []
        print(f"entity terms: {entity_terms}")
        for term in entity_terms:
            vec = self.embedding_model.encode(term, convert_to_tensor=True).cpu().numpy().reshape(1,-1)
            distances, indices = nn_model.kneighbors(vec, n_neighbors=top_k)
            results.extend([names[i] for i in indices[0]])
        return results
    
    #path signuture in utils

    def _fuzzy_search(self, names):
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
        exact_results = self.graph.query(cypher_exact, {"resolved_names": names})
        return exact_results


    #mean pool path in utils

    def _query_graph(self, input_dict):
        self.query_database_duration = None
        start_time = time.perf_counter()
        entities = input_dict["entities"]
        entity_terms = entities.names
        resolved_names = [resolve_entity(name,self.cache, self.graph, allow_updates=False) for name in entity_terms]
        #run_update(graph=self.graph)
        raw_results = [
        {**res, "source": "initial_connection"} for res in find_connections_between_names(resolved_names, self.graph)
        ]

        exact_results = []
        if len(raw_results)<5: #look at lexical AND cosine similarity
            print(f"DEBUG: node connection returned {len(raw_results)} results: falling back to fuzzy and semantic search...")
            exact_results = self._fuzzy_search(resolved_names)
            exact_results = [{**res, "source": "fuzzy"} for res in exact_results]
            print(f"DONE WITH EXACT RESULTS... len:{len(exact_results)}")
            semantic_nodes = self._top_embedded_entities(entity_terms)
            print(f"DONE WITH FINDING TOP SEMANTIC NODES. len: {len(semantic_nodes)}")
            semantic_results = self._fuzzy_search(semantic_nodes)
            semantic_results = [{**res, "source": "semantic"} for res in semantic_results]

            raw_results += exact_results + semantic_results
        
        seen_signatures = set()
        graph_results = []
        for res in raw_results:
            sig = path_signature(res["path"])
            if sig not in seen_signatures:
                graph_results.append(res)
                seen_signatures.add(sig)
        print(f"→ Total raw paths: {len(raw_results)}")
        print(f"→ Unique paths after deduplication: {len(graph_results)}")

        if len(graph_results)<5:
            return {
                "__return_direct__": True,
                "output": "Sorry I don't have enough information given the provided questions. Can you ask more specifically what you need help with?"
            }
        list_of_terms = [q for q in input_dict["chat_messages"]]
        vectors = self.embedding_model.encode(list_of_terms, convert_to_tensor=True)
        chat_embedding = vectors.cpu().numpy()
        scored_paths = []
        for result in graph_results:
            vec = mean_pool_path(result["path"])
            if vec is not None:
                similarities = cosine_similarity(chat_embedding, vec.reshape(1, -1))
                highest_score = max(similarities[:,0]) #choose highest score 
                scored_paths.append((highest_score, result))
        scored_paths.sort(key=lambda x: x[0], reverse=True) 
        top_paths = [res for _, res in scored_paths[:3]] #ignoring the score 
        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)
        self.query_database_duration = duration
        return {"graph_results": top_paths, 
                "bash_commands": input_dict["bash_commands"], 
                "chat_messages": input_dict["chat_messages"], 
                "chat_answers": input_dict["chat_answers"]
                }

    def _format_for_summary(self, x):
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
            "chat_messages": x["chat_messages"], 
            "chat_answers": x["chat_answers"]
        } #could also unpack using **x but still need to get rid of graph_results or I could keep it along 

    def _build_summarize_chain(self): #for now just copy and past getting started in
        summary_prompt = PromptTemplate.from_template(
            "The scenario summary: The tasks involve pwd, ls, man, cd, mv, cp, chmod, md5sum, grep, find and others. Key objectives include understanding command syntax, file manipulation, and permission settings, using man for command documentation, and composing commands with redirection and filters such as less and sort. They explore options for ls to display hidden files. The student must rename and copy files using mv and cp, adjust file permissions with chmod, and use pattern matching with both regular expressions and glob patterns."
            "The student's recent bash commands: {bash_commands}, The student's recent chat messages: {chat_messages}, The students recent chat answers: {chat_answers}"
            "Ordered graph data: <start data> \n{data}\n <end data>"
            "The graph data is ordered with the most relevant at the top, it is meant to give you example information about themes from the students questions and commands. Simplify and adjust commands to the students needs don't just copy them directly. The data comes in the form 'entity1' (entity1 properties) ->relationsip to-> 'entity2 (entity2 properties)"
            "A student is completing a cyber-security scenario. Review the scenario's summary, their bash, chat and question/answer history, along with the relevant graph data and provide them a single concise hint on what to do next. The hint MUST not exceed two sentences in length.\n"
        )
        return summary_prompt | self.llm

    def run(self, input_dict: dict):
        start_time = time.perf_counter()
        generated_hint = self.graph_chain.invoke(input_dict)
        stop_time = time.perf_counter()
        duration = round(stop_time - start_time, 2)
        return generated_hint, duration, self.query_database_duration
    


class GraphRAGPipeline_No_Context_File:
    def __init__(self, llm, graph, all_nodes):
        self.llm = llm
        self.graph = graph
        self.all_nodes = all_nodes
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.entity_llm = llm 
        class Entities(BaseModel):
            """Identifying information about entities."""
            names: List[str] = Field(
                ...,
                description="All the person, organization, or business entities that appear in the text",
            )

        # prompt and parser setup
        self.parser = PydanticOutputParser(pydantic_object=Entities)
        self.prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting cybersecurity related entities (commands, concepts, key phrases, etc...) from a students most recent bash commands and questions/answers"),
        ("human", "Use the given format to extract information from the following input: {question}"),
    ]
)
        self.extract_entities_llm = self.prompt | self.entity_llm.with_structured_output(Entities)
        self.graph_chain = self._build_graph_chain()
        self.cache = {}
        self.query_database_duration = None


    def _build_graph_chain(self):
        return RunnableSequence(
            RunnableLambda(self._make_input_dict),
            RunnableLambda(self._extract_entities),
            RunnableLambda(self._query_graph),
            RunnableLambda(self._format_for_summary),
            self._build_summarize_chain()  # already a chain
        )

    def _make_input_dict(self, prompt: str) -> dict:
        if isinstance(prompt, dict) and isinstance(prompt["bash_commands"], list) and isinstance(prompt["chat_messages"], list) and isinstance(prompt["chat_answers"], list):
            return prompt 
        raise TypeError("Input must be a dict with 'bash_commands', 'chat_messages', and 'chat_answers' as lists.")

    def _extract_entities(self, input_dict):
        bash = input_dict["bash_commands"]
        chat = input_dict["chat_messages"]
        answers = input_dict["chat_answers"]
        question = f"A students bash commands: {bash}\n A students chat messages: {chat} A students chat answers: {answers}"
        entities = self.extract_entities_llm.invoke({"question": question})
        entities.names = [name.lower().strip().replace(" ", "_") for name in entities.names]
        for q in chat:
            entities.names.append("_".join(q.split()))
        print(f"DEBUG extracted entities: {entities}")
        return {"entities": entities, "bash_commands": bash, "chat_messages": chat, "chat_answers": answers}
    

    def _top_embedded_entities(self, entity_terms, top_k=5):
        names = [node['name'] for node in self.all_nodes]
        vectors = np.array([node['embedding'] for node in self.all_nodes])
        nn_model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(vectors)
        results = []
        print(f"entity terms: {entity_terms}")
        for term in entity_terms:
            vec = self.embedding_model.encode(term, convert_to_tensor=True).cpu().numpy().reshape(1,-1)
            distances, indices = nn_model.kneighbors(vec, n_neighbors=top_k)
            results.extend([names[i] for i in indices[0]])
        return results
    
    #path signuture in utils

    def _fuzzy_search(self, names):
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
        exact_results = self.graph.query(cypher_exact, {"resolved_names": names})
        return exact_results


    #mean pool path in utils

    def _query_graph(self, input_dict):
        self.query_database_duration = None
        start_time = time.perf_counter()
        entities = input_dict["entities"]
        entity_terms = entities.names
        resolved_names = [resolve_entity(name,self.cache, self.graph, allow_updates=False) for name in entity_terms]
        #run_update(graph=self.graph)
        raw_results = [
        {**res, "source": "initial_connection"} for res in find_connections_between_names(resolved_names, self.graph)
        ]

        exact_results = []
        if len(raw_results)<5: #look at lexical AND cosine similarity
            print(f"DEBUG: node connection returned {len(raw_results)} results: falling back to fuzzy and semantic search...")
            exact_results = self._fuzzy_search(resolved_names)
            exact_results = [{**res, "source": "fuzzy"} for res in exact_results]
            print(f"DONE WITH EXACT RESULTS... len:{len(exact_results)}")
            semantic_nodes = self._top_embedded_entities(entity_terms)
            print(f"DONE WITH FINDING TOP SEMANTIC NODES. len: {len(semantic_nodes)}")
            semantic_results = self._fuzzy_search(semantic_nodes)
            semantic_results = [{**res, "source": "semantic"} for res in semantic_results]

            raw_results += exact_results + semantic_results
        
        seen_signatures = set()
        graph_results = []
        for res in raw_results:
            sig = path_signature(res["path"])
            print(sig)
            if sig not in seen_signatures:
                graph_results.append(res)
                seen_signatures.add(sig)
        print(f"→ Total raw paths: {len(raw_results)}")
        print(f"→ Unique paths after deduplication: {len(graph_results)}")

        if len(graph_results<5):
            return {
                "__return_direct__": True,
                "output": "Sorry I don't have enough information given the provided questions. Can you ask more specifically what you need help with?"
            }
        
        list_of_terms = [q for q in input_dict["chat_messages"]]
        vectors = self.embedding_model.encode(list_of_terms, convert_to_tensor=True)
        chat_embedding = vectors.cpu().numpy()
        scored_paths = []
        for result in graph_results:
            vec = mean_pool_path(result["path"])
            if vec is not None:
                similarities = cosine_similarity(chat_embedding, vec.reshape(1, -1))
                highest_score = max(similarities[:,0]) #choose highest score 
                scored_paths.append((highest_score, result))
        scored_paths.sort(key=lambda x: x[0], reverse=True) 
        top_paths = [res for _, res in scored_paths[:3]] #ignoring the score [TAKE TOP 3 results]
        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)
        self.query_database_duration = duration
        return {"graph_results": top_paths, 
                "bash_commands": input_dict["bash_commands"], 
                "chat_messages": input_dict["chat_messages"], 
                "chat_answers": input_dict["chat_answers"]
                }

    def _format_for_summary(self, x):
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
            "chat_messages": x["chat_messages"], 
            "chat_answers": x["chat_answers"]
        } #could also unpack using **x but still need to get rid of graph_results or I could keep it along 

    def _build_summarize_chain(self): #for now just copy and past getting started in
        summary_prompt = PromptTemplate.from_template(
            "The student's recent bash commands: {bash_commands}, The student's recent chat messages: {chat_messages}, The students recent chat answers: {chat_answers}"
            "Ordered graph data: <start data> \n{data}\n <end data>"
            "The graph data is ordered with the most relevant at the top, it is meant to give you example information about themes from the students questions and commands. Simplify and adjust commands to the students needs don't just copy them directly. The data comes in the form 'entity1' (entity1 properties) ->relationsip to-> 'entity2 (entity2 properties)"
            "A student is completing a cyber-security scenario, look at their bash, chat and question/answer history and provide them a single concise hint on what to do next. The hint must not exceed two sentences in length."
        )
        return summary_prompt | self.llm

    def run(self, input_dict: dict):
        start_time = time.perf_counter()
        generated_hint = self.graph_chain.invoke(input_dict)
        stop_time = time.perf_counter()
        duration = round(stop_time - start_time, 2)
        return generated_hint, duration, self.query_database_duration
