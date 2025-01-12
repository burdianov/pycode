import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very sort {language} functon that will {task}",
)

code_chain = RunnableSequence(code_prompt, llm)

result = code_chain.invoke({"language": "python", "task": "return a list of numbers"})
print(result.content)
