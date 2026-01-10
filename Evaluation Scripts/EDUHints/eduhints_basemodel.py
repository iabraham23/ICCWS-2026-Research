from llama_cpp import Llama
import time

llm = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,
    n_threads=8,
    use_mmap=True,
    use_mlock=True,
    verbose=False
)
print("ran Llama from basemodel")

def load_context_file_contents(context_file_type, scenario_name):
    file_path = f"langchain/{context_file_type}/{scenario_name}.txt"
    try: 
        with open(file_path, 'r', encoding='utf-8') as file:
                context_file_content = file.read()
        return context_file_content

    except Exception as e:
        raise Exception (f"ERROR: Failed to load context file contents: {e}")


def generate_hint(language_model_object_llama, generation_parameters, logs_dict):
        start_time = time.perf_counter()
        try:
            temperature = float(generation_parameters.get('temperature'))
            scenario_name = generation_parameters.get('scenario_name')
            disable_scenario_context = generation_parameters.get('disable_scenario_context')

        except Exception as e:
            raise Exception (f"ERROR: Failed to load items from Redis cache: [{e}]")

        if disable_scenario_context:

            try:
                finalized_system_prompt = "##A student is completing a cyber-security scenario, look at their bash, chat and question/answer history and provide them a single concise hint on what to do next. The hint must not exceed two sentences in length."
                finalized_user_prompt = f" The student's Recent bash commands: {logs_dict['bash']}. The student's recent chat messages: {logs_dict['chat']}. The student's recent answers: {logs_dict['responses']}. "
            
            except Exception as e:
                raise Exception (f"ERROR: Failed to initialize prompts: [{e}]")
        else:

            try:
                scenario_summary = load_context_file_contents('scenario_summaries', scenario_name)

            except Exception as e:
                raise Exception (f"ERROR: 'load_context_file_contents()' failed: [{e}]")

            try:
                finalized_system_prompt = "##A student is completing a cyber-security scenario, review the scenario's summary along with their bash, chat and question/answer history and provide them a single concise hint on what to do next. The hint must not exceed two sentences in length."
                finalized_user_prompt = f" The scenario's summary: {scenario_summary}. The student's recent bash commands: {logs_dict['bash']}. The student's recent chat messages: {logs_dict['chat']}. The student's recent answers: {logs_dict['responses']}. "
            
            except Exception as e:
                raise Exception (f"ERROR: Failed to initialize prompts: [{e}]")


        try:
            result = language_model_object_llama(
                f"<|system|>{finalized_system_prompt}<|end|>\n<|user|>\n{finalized_user_prompt}<|end|>\n<|assistant|> ",
                max_tokens=-1,
                stop=["<|end|>"], 
                echo=False, 
                temperature=temperature,
            )

        except Exception as e:
            raise Exception (f"ERROR: Failed to generate results: [{e}]")
            
        generated_hint = result["choices"][0]["text"]

        stop_time = time.perf_counter()
        duration = round(stop_time - start_time, 2)

        return generated_hint, logs_dict, duration

def generate_hint_with_prompts(sys_prompt, user_prompt):
    start_time = time.perf_counter()
    try:
        temperature = 0.0
    except Exception as e:
        raise Exception (f"ERROR: Failed to load items from Redis cache: [{e}]")

    try:
        result = llm.create_completion(
            f"<|system|>{sys_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|> ",
            max_tokens=-1,
            stop=["<|end|>"], 
            echo=False, 
            temperature=temperature,
        )

    except Exception as e:
        raise Exception (f"ERROR: Failed to generate results: [{e}]")
        
    generated_hint = result["choices"][0]["text"]

    stop_time = time.perf_counter()
    duration = round(stop_time - start_time, 2)

    return generated_hint, duration

