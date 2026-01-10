from full_local_graphrag import GraphRAGPipeline, GraphRAGPipeline_No_Context_File
from langchain_community.llms.llamacpp import LlamaCpp
from log_into_user_sys import get_system_user_prompts
from eduhints_vs_graphrag import get_bash_and_chat, extract_array
from langchain_neo4j import Neo4jGraph
import os 
from tqdm import tqdm 

path = 'langchain/logs/Fully_Cleaned_Prompt_Data.csv'
df = get_system_user_prompts(path)
#df = df[:11] + [df[-1]] #with context file
#df = df[11:-1] #no context file
df = df[:2]

all_nodes = []
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

chain = GraphRAGPipeline(llm=llm, graph=graph, all_nodes=all_nodes)

for index, r in tqdm(enumerate(df), desc="Appending to rows..."):
    user_prompt = r["user_prompt"]
    system_prompt = r["system_prompt"]
    bash_chat_dict = get_bash_and_chat(user_prompt)
    graphhint = chain.run(bash_chat_dict)
    print(graphhint)

