# encoding:utf8
import os

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.vectorstores import LanceDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import lancedb

# 读取文档
txt_loader = TextLoader("demo.txt", encoding="utf8")
txt_content = txt_loader.load()
txt_content_list = [i.page_content for i in txt_content]
print("文本内容列表：",txt_content_list)
# separator: 分隔符
# chunk_size: 切割长度
# chunk_overlap: 切割后文本可重复的长度
txt_splitter = CharacterTextSplitter(separator="\n",chunk_size=50, chunk_overlap=15)
docs = txt_splitter.create_documents(txt_content_list)
print("切割后的文本内容：",docs)
for i in docs:
    print("=====",i)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# 半结构化转向量
db = lancedb.connect(os.path.join(os.path.dirname(os.getcwd()), "lancedb"))

vectordb = LanceDB.from_texts(
    texts=[i.page_content for i in docs],
    embedding=embeddings,
    connection=db,
    metadatas=[{"source": str(i)} for i in range(len(docs))],
    table_name='langchain_test'
)
print("数据已存储到Lancedb向量库中")

