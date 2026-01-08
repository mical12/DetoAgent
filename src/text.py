import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 设置数据库路径和嵌入模型
persist_directory = '/mnt/d/Ubantu_run/MetaOpenFOAM/database/openfoam_tutorials_summary'
embeddings = HuggingFaceEmbeddings(model_name="/mnt/d/Ubantu_run/MetaOpenFOAM/Qwen3-Embedding-0.6B")

# 加载已保存的向量数据库
vectordb = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)

# 方法1: 相似性搜索 - 返回最相似的文档
query =  '\n        Find the OpenFOAM case that most closely matches the following case:\n        \n\ncase name: Buoyant_Cavity  \ncase domain: heatTransfer  \ncase category: RAS  \ncase solver: buoyantFoam  \n\nNote: The category is set to RAS (Reynolds-Averaged Simulation) since this is explicitly a RANS simulation. The domain remains heatTransfer as this involves natural convection analysis, consistent with other buoyancy-driven cases in the context.\n        where case domain, case category and case solver should be matched with the highest priority\n    '
k = 5  # 返回前5个最相似的结果

# 基本相似性搜索
docs = vectordb.similarity_search(query, k=k)
print("=== 相似性搜索结果 ===")
for i, doc in enumerate(docs):
    print(f"结果 {i+1}:")
    print(f"内容: {doc.page_content[:200]}...")  # 显示前200个字符
    print(f"元数据: {doc.metadata}")
    print("-" * 50)
