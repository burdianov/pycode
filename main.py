import os
import argparse

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="return a list of numbers")
parser.add_argument("--language", type=str, default="python")
args = parser.parse_args()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# provided that the OpenAI API key is named OPENAI_API_KEY in the .env file, the above 2 lines can be replaced with:
# llm = ChatOpenAI(model="gpt-4o-mini")

code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very sort {language} functon that will {task}",
)

code_chain = RunnableSequence(code_prompt, llm)

result = code_chain.invoke({"language": args.language, "task": args.task})
print(result.content)
