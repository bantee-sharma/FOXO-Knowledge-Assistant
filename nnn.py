from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

print(search.invoke("Trending news in india"))