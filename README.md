# RAG Chatbot Workshop

A simple RAG (Retrieval-Augmented Generation) chatbot that uses Google Gemini for AI responses and embeddings.

## Setup

1. **Create and activate a virtual environment**:

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   If requirements.txt is giving you errors, just do:

   ```bash
   pip install langchain langchain-google-genai langchain-community
   pip install langchain-text-splitters faiss-cpu python-dotenv
   ```

3. **Get your Free Google API key**:

   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Create and add it to your `.env` file: "GOOGLE_API_KEY=your_api_key_here"

4. **Run the chatbot**:
   ```bash
   python rag_chatbot.py
   ```
