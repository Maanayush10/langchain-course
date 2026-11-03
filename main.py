from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


# custom search tool
# tavily= TavilyClient()

# @tool
# def search(query:str)->str:
#     '''
#     Tool that searches over the internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     '''
#     print(f"Searching for query : {query}")
#     return tavily.search(query=query)


# tavily search tool added in the tools

class Source(BaseModel):
    """Scehma for the reource used by the agent"""

    url: str = Field(description="The url of the source used by the agent")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources."""
    answer: str= Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello world from langchain course, This is AYUSHMAN THE GREAT.")
    # result= agent.invoke({"messages":"What is the weather in Tokyo"})
    result = agent.invoke(
        {
            "messages": "search for 3 job postings for an ai engineer using langchain in Bengaluru, India on linkedin and list their details?"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
