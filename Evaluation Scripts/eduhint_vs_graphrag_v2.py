#NEW VERSION, September 2025 data  

from dotenv import load_dotenv
load_dotenv()
import os 
from langchain_neo4j import Neo4jGraph
from hybridrag_class import GraphRAGPipeline, GraphRAGPipeline_No_Context_File
from eduhints_basemodel import generate_hint_with_prompts
from langchain_community.llms.llamacpp import LlamaCpp
import time
from log_into_user_sys import get_system_user_prompts
import re 
import ast
import pandas as pd
from tqdm import tqdm
from graph_utils import get_bash_chat_v2


def compare_to_base_eduhint(df, chain, out_path):
    rows = []
    columns = ["System_Prompt", "User_Prompt", "Bash_Commands", "Recent_Responses", "Chat_Answers", "EDUHint", "GraphRag", "EDUHint_Hint_Duration", "GraphRag_Hint_Duration", "GraphRag_DB_Query_Duration"]
    for index, r in tqdm(enumerate(df), desc="Appending to rows..."):
        print(f'ROW: {index}')
        user_prompt = r["user_prompt"]
        system_prompt = r["system_prompt"]
        bash_chat_dict = get_bash_chat_v2(user_prompt)
        tqdm.write("getting eduhint...")
        eduhint = generate_hint_with_prompts(system_prompt, user_prompt)
        tqdm.write("finished edhuint, getting graphhint...")
        graphhint = chain.run(bash_chat_dict)
        tqdm.write("finished graphhint, getting eduhint...")
        rows.append({"System_Prompt": system_prompt,
                    "User_Prompt": user_prompt,
                    "Bash_Commands": bash_chat_dict["bash_commands"], 
                    "Recent_Responses": bash_chat_dict["chat_messages"], 
                    "Chat_Answers": bash_chat_dict["chat_messages"], 
                    "EDUHint": eduhint[0], 
                    "GraphRag": graphhint[0], 
                    "EDUHint_Hint_Duration": eduhint[1], 
                    "GraphRag_Hint_Duration": graphhint[1], 
                    "GraphRag_DB_Query_Duration": graphhint[2]})
    pd_df = pd.DataFrame(rows, columns=columns)
    pd_df.to_csv(out_path, index=False)
    print("DONE: saved to evaluations folder")

def compare_with_set_eduhint(df, chain, out_path): #this is to be used when we don't have the exact edhint system to test with but we have its data 
    rows = []
    columns = ["System_Prompt", "User_Prompt", "Bash_Commands", "Recent_Responses", "Chat_Answers", "EDUHint", "GraphRag", "EDUHint_Hint_Duration", "GraphRag_Hint_Duration", "GraphRag_DB_Query_Duration"]
    for index, r in tqdm(enumerate(df), desc="Appending to rows..."):
        print(f'ROW: {index}')
        user_prompt = r["user_prompt"]
        system_prompt = r["system_prompt"]
        eduhint = r["eduhint"]
        eduhint_duration = r["eduhint_duration"]
        bash_chat_dict = get_bash_chat_v2(user_prompt)
        graphhint = chain.run(bash_chat_dict)
        tqdm.write("finished graphhint")
        rows.append({"System_Prompt": system_prompt,
                    "User_Prompt": user_prompt,
                    "Bash_Commands": bash_chat_dict["bash_commands"], 
                    "Recent_Responses": bash_chat_dict["chat_messages"], 
                    "Chat_Answers": bash_chat_dict["chat_answers"], 
                    "EDUHint": eduhint, 
                    "GraphRag": graphhint[0], 
                    "EDUHint_Hint_Duration": eduhint_duration, 
                    "GraphRag_Hint_Duration": graphhint[1], 
                    "GraphRag_DB_Query_Duration": graphhint[2]})
    pd_df = pd.DataFrame(rows, columns=columns)
    pd_df.to_csv(out_path, index=False)
    print("DONE: saved to evaluations folder")


if __name__ == "__main__":
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "LewisAndClark"

    graph = Neo4jGraph(enhanced_schema=True)
    llm = LlamaCpp(
        model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx=4096,
        n_threads=8,
        use_mmap=True,
        use_mlock=True,
        verbose=False
    )

    print("ran LlamaCpp from eduhints vs graphrag v2")

    start = time.perf_counter()
    all_nodes = graph.query("MATCH (n) WHERE n.embedding IS NOT NULL RETURN n.name AS name, n.embedding AS embedding")
    end = time.perf_counter()
    print(f"time to process all nodes: {round(end-start, 3)}")

    in_path = 'langchain/logs/cleaned_sept_qwen_logs.csv'
    df = get_system_user_prompts(in_path)
    #df = df[:18] + df[20:]
    

    """
    MAKE SURE WE USING THE RIGHT CHAIN, WITH OR WITHOUT CONTEXT FILE
    For this testing we will temporarily use the older context file although the hints we compare with use a newer one
    """
    chain = GraphRAGPipeline(llm=llm, graph=graph, all_nodes=all_nodes) 


    #in this testing we don't need to regenerate eduhints as they are already generated and tracked with times
    #however, it should be noted that times won't be a fair comparison as they were generated on a different system
    #we will combat this be regenerating base eduhints, seeing the time difference compared to when they did it, then apply that percent to the rest of the hints

    out = "Data/printing.csv"
    compare_with_set_eduhint(df=df, chain=chain, out_path=out)

