import os
import argparse

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

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

code_chain: RunnableSequence = RunnableSequence(
    code_prompt,
    llm,
    lambda response: response.content,
)
test_chain: RunnableSequence = RunnableSequence(
    test_prompt,
    llm,
    lambda response: response.content,
)

chain = RunnableSequence(
    RunnablePassthrough.assign(code=lambda x: code_chain.invoke(x)),
    RunnablePassthrough.assign(test=lambda x: test_chain.invoke(x)),
)

result = chain.invoke({"language": args.language, "task": args.task})

print("Generated Code:\n", result["code"])
print("\nGenerated Test:\n", result["test"])
