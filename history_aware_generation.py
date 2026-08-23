from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

persist_directory='db/chroma_db'

embadding=OpenAIEmbeddings( model='text-embedding-3-small')
db=Chroma(persist_directory=persist_directory, embedding_function=embadding)
model=ChatOpenAI(model='gpt-4o')

chat_history=[]


def ask_question(user_question):
    
    print(f'\n user question is : {user_question}')
    
    if chat_history:
        meaasge=[SystemMessage(content='use the currect user entering question to make conversational question compare with previous question availablein the chat history')] + chat_history +[HumanMessage(content=f' user question : {user_question}')] 
        result=model.invoke(meaasge)
        search_question=result.content
        print(f' searching for it :::: {search_question}')
    else:
        search_question=user_question
        
    print('\n at retreiever')
    retriever=db.as_retriever(search_kwargs={'k':3})
    
    print('\n at embbad invoke')
    documents=retriever.invoke(search_question)
    
    print(f' doc length {len(documents)}')
    
    document= "\n".join([f'-{docs.page_content}' for docs in documents])
    
    combined_output=f""" 
    based on the follow documents answer this question. Question is : {user_question}
    Documents: {document}
    If you can't find the answer in the documents, say:
    "I don't have enough information to answer that question based on the provided documents."
    """
    
    message=[SystemMessage(content='You are a helpful assistant that answers questions based on provided documents and conversation history.')] +chat_history + [HumanMessage(content=combined_output)]
    
    result=model.invoke(message)
    answer=result.content
    
    print(answer)
    
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    return answer







def run():
    print('\n user can enter query here \n')
    
    while True:
        question=input('\n Enter a question : ')
        
        if question == 'q':
            print('thank you...')
            break
        ask_question(question)

if __name__=="__main__":
    run()