from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

persistance_directry='db/chroma_db'

embadding_model=OpenAIEmbeddings(model='text-embedding-3-small')

db=Chroma(
    embedding_function=embadding_model,
    persist_directory=persistance_directry,
    collection_metadata={'hnsw:space':'cosine'} 
)
query = "What was the original name of Microsoft before it became Microsoft?"

retriever=db.as_retriever(search_kwargs={'k':5})
releveant_docs=retriever.invoke(query)

print(f'\n query : {query} \n')
print(f'----------------------- context -------------------------------')

for i, doc in enumerate(releveant_docs,1):
    print(f' {i} : \n - {doc.page_content}  \n')
    
    
    
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in releveant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

model=ChatOpenAI(model="gpt-4o")

message=[
    SystemMessage(content='your are a helpful assistance'),
    HumanMessage(content=combined_input)
]

result=model.invoke(message)

print('------------------------generated response----------------------------')
print(result.content)