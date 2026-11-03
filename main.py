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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


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
