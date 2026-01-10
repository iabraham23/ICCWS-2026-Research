import re

import pandas as pd

def clean_log_csv(log_path: str) -> pd.DataFrame:
    """
    Reads a log CSV file with embedded labels and multiline fields.
    Cleans labels like '*SYSTEM PROMPT*:' from all relevant fields.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with original structure preserved.
    """
    # Load CSV with multiline support
    df = pd.read_csv(log_path, engine="python", quotechar='"')

    # Strip labels from each relevant field
    df["scenario_name"] = df["scenario_name"].str.replace(r"^\*SCENARIO NAME\*:\s*", "", regex=True)
    df["disable_scenario_context_file"] = df["disable_scenario_context_file"].str.replace(
        r"^\*BOOL DISABLE SCENARIO CONTEXT FILE\*:\s*", "", regex=True)
    df["system_prompt"] = df["system_prompt"].str.replace(r"^\*SYSTEM PROMPT\*:\s*", "", regex=True)
    df["user_prompt"] = df["user_prompt"].str.replace(r"^\*USER PROMPT\*:\s*", "", regex=True)
    df["generation_duration"] = df["generation_duration"].str.replace(
        r"^\*HINT GENERATION DURATION\*:\s*", "", regex=True)

    return df

def clean_log_csv_v2(log_path: str) -> pd.DataFrame: #for september data 
    """
    Reads a log CSV file with embedded labels and multiline fields.
    Cleans labels like '*SYSTEM PROMPT*:' from all relevant fields.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with original structure preserved.
    """
    # Load CSV with multiline support
    df = pd.read_csv(log_path, engine="python", quotechar='"')

    # Strip labels from each relevant field
    df["scenario_name"] = df["scenario_name"].str.replace(r"^\*SCENARIO NAME\*:\s*", "", regex=True)
    df["disable_scenario_context_file"] = df["disable_scenario_context_file"].str.replace(
        r"^\*BOOL DISABLE SCENARIO CONTEXT FILE\*:\s*", "", regex=True)
    df["system_prompt"] = df["system_prompt"].str.replace(r"^\*SYSTEM PROMPT\*:\s*", "", regex=True)
    df["user_prompt"] = df["user_prompt"].str.replace(r"^\*USER PROMPT\*:\s*", "", regex=True)
    df["generated_hint"] = df["generated_hint"].str.replace(r"^\*GENERATED HINT\*:\s*(?:EDUHINT:\s*)?","",regex=True)
    df["eduhint_duration"] = df["eduhint_duration"].str.replace(r"^\*HINT GENERATION DURATION\*:\s*", "", regex=True)

    return df



def get_system_user_prompts(log_path) -> list[dict]:
    arr = []
    logs = pd.read_csv(log_path)
    for _,row in logs.iterrows():
        arr.append({'system_prompt': row['system_prompt'], 
                    'user_prompt': row['user_prompt'], 
                    'eduhint': row["generated_hint"], 
                    'eduhint_duration': row["eduhint_duration"]})
    return arr

if __name__ == '__main__':
    path = 'langchain/logs/sept_qwen_logs.csv'
    df = clean_log_csv_v2(path)
    df.to_csv("langchain/logs/cleaned_sept_qwen_logs.csv")

    # inputs = get_system_user_prompts(path)
    # print(inputs[0]['system_prompt'])