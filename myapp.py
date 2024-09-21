import streamlit as st
from rag import rag  
from langchain_community.document_loaders import YoutubeLoader
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def ytLoader(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    data = loader.load()
    transcript_text = data[0].page_content
    
    # Save transcript to file
    with open('transcript_text.txt', 'w', encoding='utf-8') as file:
        file.write(transcript_text)

    return transcript_text

def main():
    st.title("YouTube Video Q&A with RAG")

    url = st.text_input("Enter YouTube Video URL", "")
    if url:
        st.video(url)        
        try:
            transcript_text = ytLoader(url)
            
            question = st.text_input("Enter your question about the video")

            if st.button("Get Answer"):
                if question:
                    st.write("Generating response...")
                    response = rag(transcript_text, question)
                    st.subheader("RAG Response")
                    st.write(response)
                else:
                    st.error("Please enter a question.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
