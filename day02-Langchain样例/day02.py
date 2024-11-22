from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 实例化
llm = ChatOllama(base_url="http://localhost:11434", model="qwen2.5:0.5b")

# 声明提示词模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个专业的翻译官，可以将用户提供的内容翻译为英文"),
        ("user", "{question}")
    ]
)
# 创建链
chain = prompt_template | llm | StrOutputParser()

# 调用
result = chain.invoke({"question": "我爱写代码"})

print(result)
