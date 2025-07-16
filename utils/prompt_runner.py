from dotenv import load_dotenv
from langchain_openai import OpenAI
import os

# Load environment variables
load_dotenv()

def load_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        return f.read()

def run_prompt(prompt_text):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    llm = OpenAI()
    response = llm.invoke(prompt_text)
    return response

if __name__ == "__main__":
    prompt_path = "../prompts/zero_few_shot_examples.md"
    prompt_text = load_prompt(prompt_path)
    output = run_prompt(prompt_text)
    print("Prompt:\n", prompt_text)
    print("\nLLM Output:\n", output)