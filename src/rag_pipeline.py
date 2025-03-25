# src/rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

load_dotenv("config/api_keys.env")
genai.configure(api_key="")

class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = genai.GenerativeModel(model_name='gemini-2.0-flash') #gemini-2.0-flash   #gemini-2.0-flash-lite #Gemini 1.5 Flash
        self.prompt = ChatPromptTemplate.from_template(
            """Answer using this context:
            {context}
            
            Question: {question}
            """
        )
    
    # Gemini 2.0 Flash
    def generate_response(self, question):
        # Retrieve context from the retriever
    
        context = self.retriever.get_relevant_documents(question)
    
        # Format the prompt using your template
        prompt_text = self.prompt.format(context=context, question=question)
        
        # Call the Gemini model with the correct method
        response = self.llm.generate_content(prompt_text)  
        
        # Return the text from the response
        return response.text
        
