import os
import sys
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, document_path: str = "document.txt"):
        """
        Initialize the RAG chatbot with a document.
        
        Args:
            document_path (str): Path to the document file
        """
        self.document_path = document_path
        self.vector_store = None
        self.chat_history = []
        
        # Check if Google API key is set
        if not os.getenv("GOOGLE_API_KEY"):
            print("Error: GOOGLE_API_KEY not found in environment variables!")
            sys.exit(1)
        
        # Initialize Google Gemini components
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        
        print("Initializing RAG Chatbot with Google Gemini")
        self._load_and_process_document()
        print("RAG Chatbot ready! Type 'quit' to exit.\n")
    
    def _load_and_process_document(self):
        """Load the document and process it for RAG."""
        try:
            # Load document
            with open(self.document_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            print(f"Loaded document: {self.document_path}")

            # TODO: Split document text into smaller chunks to be embedded effectively 
            # HINT: use RecursiveCharacterTextSplitter from langchain.text_splitter w/ params chunk_size = 500, chunk_overlap = 50 and separators=["\n\n", "\n", ". ", " ", ""]
            
            print(f"Split document into {len(chunks)} chunks")
            
            # Convert chunks to documents
            documents = [Document(page_content=chunk, metadata={"source": self.document_path}) for chunk in chunks]
            
            # TODO: Create FAISS vector store so the chatbot can search for relevant text
            # HINT: use FAISS.from_documents 
            
            print(f"Created vector store with {len(chunks)} embeddings")
            
        except FileNotFoundError:
            print(f"Error: Document file '{self.document_path}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing document: {e}")
            sys.exit(1)
    
    def _retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve the most relevant document chunks for a given query.
        
        Args:
            query (str): User's question
            k (int): Number of chunks to retrieve
            
        Returns:
            List[str]: Retrieved text chunks
        """
        if not self.vector_store:
            return []
        
        # TODO: Retrieve top-k most relevant chunks from FAISS vector store 
        # HINT: use vector_store.similarity_search

        return [doc.page_content for doc in docs]
    
    def _generate_response(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate a response using the retrieved context.
        
        Args:
            query (str): User's question
            context_chunks (List[str]): Retrieved document chunks
            
        Returns:
            str: Generated response
        """
        # Create a simple prompt string for Gemini compatibility
        context_text = "\n\n".join(context_chunks)
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 
        Use only the information from the context to answer questions. If the context doesn't contain 
        enough information to answer the question, say so. Always be educational and clear in your explanations.
        
        Context:
        {context_text}
        
        Question: {query}
        
        Answer:"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        return response.content
    
    def chat(self):
        """Main chat loop."""
        print("Welcome to the RAG Chatbot! Ask me anything about RAG.")
        print("I'll retrieve relevant information from the document and answer your questions.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for quit command
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! Thanks for learning about RAG!")
                    break
                
                if not user_input:
                    continue
                
                print("Searching for relevant information...")
                
                # TODO: retrieve the relevant chunks and store under relevant_chunks
                
                if not relevant_chunks:
                    print("No relevant information found in the document.")
                    continue
                
                print(f"Found {len(relevant_chunks)} relevant document chunks")
                
                # Generate response
                print("Generating response...")
                response = self._generate_response(user_input, relevant_chunks)
                
                # Display response
                print(f"\nAssistant: {response}\n")
                
                # Store in chat history
                self.chat_history.append({
                    "user": user_input,
                    "assistant": response,
                    "retrieved_chunks": relevant_chunks
                })
                
            except KeyboardInterrupt:
                print("\nGoodbye! Thanks for learning about RAG!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def show_chat_history(self):
        """Display the chat history with retrieved chunks."""
        if not self.chat_history:
            print("No chat history yet.")
            return
        
        print("\nChat History:")
        print("=" * 50)
        
        for i, entry in enumerate(self.chat_history, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {entry['user']}")
            print(f"Assistant: {entry['assistant']}")
            print(f"Retrieved Chunks: {len(entry['retrieved_chunks'])} chunks")
            print("-" * 30)

def main():
    """Main function to run the RAG chatbot."""
    print("RAG Chatbot Workshop")
    print("=" * 30)
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Start chat
    chatbot.chat()

if __name__ == "__main__":
    main()
