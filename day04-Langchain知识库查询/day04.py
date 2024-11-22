# encoding:utf8
import os

import lancedb
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = lancedb.connect(os.path.join(os.path.dirname(os.getcwd()), "lancedb"))
vectordb = LanceDB(connection=db, embedding=embeddings, table_name='langchain_test')


user_question = "Cobra创建子命令"
# 实例化LLM
llm = ChatOllama(base_url="http://localhost:11434", model="qwen2.5:0.5b")

# 声明系统提示模板,接受一个变量 <language>,目标语言
system_template = """
# 设定
你是一个知识库管理员，掌握以下内容
{content}
# 要求
- 1. 必须根据文档内容回答用户提出的问题
- 2. 若无结果直接回复未找到相关方案
"""
question = "python是什么"
sim_docs = vectordb.similarity_search(question, k=3)
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content}", end="\n--------------\n")
# 声明用户提示模板，接受一个变量<text>，翻译内容
user_template = "{question}"
# 创建提示模板
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", user_template)]
)
print("========提示词模板========")
# 使用LCEL组装一个链，将模板、LLM和输出解析器三者连接起来
chain = prompt_template | llm | StrOutputParser()
print("========封装调用链========")
# 调用链
chain_rs = chain.invoke({"question": question, "content": "\n".join([i.page_content for i in sim_docs])})
# 打印结果
print("========调用链执行结果========\n", chain_rs)
