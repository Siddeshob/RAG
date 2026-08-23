import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path=None):
    loader=DirectoryLoader( path=docs_path, glob='*.txt', loader_cls=TextLoader )
    
    documents=loader.load()
    
    print(f'\n\n\n {documents} \n\n')
    
    if len(documents)==0:
        raise FileNotFoundError(f'file not found at {docs_path} location')

    # for i, doc in enumerate(documents):
        # print(f'\n\n document : {i+1}')
        # print(f'doc.metadata : {doc.metadata['source']}')
        # print(f'doc.page_content : {len(doc.page_content)}')
        # print(f'doc.page_content[:100]  : {doc.page_content[:100]}...')
        # print(f'doc.metadata  : {doc.metadata}...')
        
    return documents
        
def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter=CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks=text_splitter.split_documents(documents)
    # for i, chunk in enumerate(chunks):
    #     print(f'\n - chunk{i+1}')
    #     print(f'\n - chunk.metadata source {chunk.metadata['source']}')
    #     print(f'\n - chunk.page_content {chunk.page_content}')
    #     print('*'*50)
    
    return chunks

def create_vector_store(chunks, persist_directory='db/chroma_db'):
    embedding_model=OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={'hnsw:space':'cosine'}
    )
    
    print(f' vectordb is created.............{persist_directory}.')
    
    return vector_store


def  main():
    
    #load the files
    path='/home/obs/Desktop/RAG/RAG_initial/docs'
    documents=load_documents(path)
    # chunk the files
    chunks=split_documents(documents)
    # embbade and store in vector db
    create_vector_store(chunks)

if __name__== "__main__":
    main()