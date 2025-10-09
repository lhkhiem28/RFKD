from source.models.llm import *

load_model = {
    'llm': BaselineLLM,
}

# Replace the following with the model paths
get_llm_path = {
    'gemma-2-2b'    : 'google/gemma-2-2b-it',
    'granite-3.3-2b': 'ibm-granite/granite-3.3-2b-instruct',
}