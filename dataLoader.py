import os
import openai
import sys
sys.path.append('../..')

# openai api
openai.api_key  = os.environ['OPENAI_API_KEY']

# load env
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()


len(pages)
page = pages[0]
print(page.page_content[0:500])
page.metadata

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")


docs = loader.load()

from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()

# Document Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

chunk_size =26
chunk_overlap = 4

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "\. ", " ", ""]
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
    separators=["\n\n", "\n", "\. ", " ", ""]
)
