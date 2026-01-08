
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import re
#import config_path
import os
import uuid
from langchain.docstore.document import Document  # 使用内置的 Document 类
from langchain_huggingface import HuggingFaceEmbeddings

# class Document:
#     def __init__(self, page_content, metadata=None):
#         self.page_content = page_content
#         self.metadata = metadata

#     def __repr__(self):
#         return f"Document(page_content={self.page_content[:30]}..., metadata={self.metadata})"
    
database_allrun_path = f'/mnt/d/Ubantu_run/MetaOpenFOAM/database/openfoam_allrun.txt'
loader = TextLoader(database_allrun_path)
pages = loader.load()

pattern = re.compile(r"```input_file_begin:(.*?)input_file_end.```", re.DOTALL)
matches = pattern.findall(pages[0].page_content)
# 构造 Document 对象，并在 metadata 中添加唯一 id
documents = [
    Document(
        page_content=match.strip(),
        metadata={'source': database_allrun_path, 'id': str(uuid.uuid4())}
    )
    for match in matches
]

persist_directory = f'/mnt/d/Ubantu_run/MetaOpenFOAM/database/openfoam_allrun'

batch_size = 10


for i in range(0, len(documents), batch_size):
    print("i:",i)
    if(i+batch_size<=len(documents)-1):
        batch = documents[i:i + batch_size]
    elif(i<=len(documents)-2):
        batch = documents[i:]
    if len(documents) <=1:
        batch = documents[i:]

    if(i==0):
        vectordb = FAISS.from_documents(
            documents=batch, 
            embedding=HuggingFaceEmbeddings(model_name = "/mnt/d/Ubantu_run/MetaOpenFOAM/Qwen3-Embedding-0.6B"))
    else:
        vectordb.add_documents(documents=batch)





vectordb.save_local(persist_directory)




