import os
import argparse

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.sequential import SequentialChain

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="return a list of numbers.")
parser.add_argument("--language", type=str, default="python")
args = parser.parse_args()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# provided that the OpenAI API key is named OPENAI_API_KEY in the .env file, the above 2 lines can be replaced with:
# llm = ChatOpenAI(model="gpt-4o-mini")

code_prompt: PromptTemplate = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}. Return the code only.",
)
test_prompt: PromptTemplate = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}",
)

code_chain: LLMChain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code",
)
test_chain: LLMChain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
    verbose=True,
)

result = chain.invoke(
    {
        "language": args.language,
        "task": args.task,
    }
)

print("Generated Code:\n", result["code"])
print("\nGenerated Test:\n", result["test"])
