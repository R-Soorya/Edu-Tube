from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def generatePrompt():
    system_prompt = (
        
        "You are a YouTube assistant. "
        "I will provide you with the transcription of the YouTube video. "
        "Your role is to assist with questions or doubts related to the video content, providing accurate answers based on the provided context. "
        "If a term or concept from the video isn't clearly explained in the transcript, offer a clear and helpful explanation based on the context, ensuring the user understands it better."
        "Your responses should strictly be relevant to the video's content, but if the user asks a question that is unrelated to the video, politely mention that the question is outside the scope of the video's context and that you can only respond to questions based on the video. "
        "If the question is related to the context but not directly covered, provide an introductory explanation to guide the user."
        "Your goal is to offer clear, concise, and contextually accurate answers based on the video transcript."

        "{context}"
    
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    return prompt


def rag(transcript_text, question):
    print('-----------------------------------------------------------------------------------------')
    print("Generating answer")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
    chunks = text_splitter.split_text(transcript_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    documents = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=200)

    prompt = generatePrompt()

    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input":question})
    return response["answer"]


