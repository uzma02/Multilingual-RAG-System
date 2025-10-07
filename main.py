
# Bangla RAG System Implementation

!pip install transformers sentence-transformers faiss-cpu nltk torch langchain langdetect 
!pip install sacremoses  # Required for mBART tokenizer
import nltk
nltk.download('punkt')


import re
import nltk
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, pipeline
import asyncio
import os

# Download required NLTK data (punkt might still be useful for other things)
nltk.download('punkt')


class BanglaRAGSystem:
    def __init__(self, document_path):
        # Initialize Bangla BERT for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
        self.model = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")

        # Initialize the new question answering model
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/xlm-roberta-base-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/xlm-roberta-base-squad2")
        self.qa_pipeline = pipeline("question-answering", model=self.qa_model, tokenizer=self.qa_tokenizer)

        self.document_chunks = []
        self.chunk_embeddings = []
        self.short_term_memory = []
        self.max_short_term_memory = 5
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(768)  # Bangla BERT output dimension
        self.load_and_process_document(document_path)

    def clean_text(self, text):
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters while preserving Bengali
        text = re.sub(r'[^\u0980-\u09FF\s\w]', '', text)
        return text

    def chunk_document(self, text):
        # Simple rule-based sentence splitting for Bengali
        sentences = re.split(r'[।?!]', text)
        sentences = [s.strip() for s in sentences if s.strip()] # Remove empty sentences

        # Create chunks of approximately 100 words
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            words = sentence.split()
            if current_word_count + len(words) <= 100:
                current_chunk.append(sentence)
                current_word_count += len(words)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = len(words)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        # Use [CLS] token embedding or mean of token embeddings
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    def load_and_process_document(self, document_path):
        # Read the document
        with open(document_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Clean and chunk the document
        cleaned_text = self.clean_text(text)
        self.document_chunks = self.chunk_document(cleaned_text)

        # Create embeddings and add to FAISS index
        for chunk in self.document_chunks:
            embedding = self.get_embedding(chunk)
            self.chunk_embeddings.append(embedding)
            self.index.add(embedding)

    def retrieve_relevant_chunks(self, query, top_k=3):
        # Clean and encode query
        cleaned_query = self.clean_text(query)
        query_embedding = self.get_embedding(cleaned_query)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.document_chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def generate_answer(self, query, relevant_chunks):
        # Combine relevant chunks with short-term memory
        context = " ".join([chunk for chunk, _ in relevant_chunks])
        context += " ".join(self.short_term_memory[-2:])  # Include last 2 interactions

        # Add to short-term memory
        self.short_term_memory.append(f"Query: {query} Context: {context}")
        if len(self.short_term_memory) > self.max_short_term_memory:
            self.short_term_memory.pop(0)

        # Use the QA pipeline to generate the answer
        qa_input = {
            'question': query,
            'context': context
        }
        result = self.qa_pipeline(qa_input)
        answer = result['answer']

        return answer.strip()

    async def process_query(self, query):
        relevant_chunks = self.retrieve_relevant_chunks(query)
        answer = self.generate_answer(query, relevant_chunks)
        return answer

async def main():
    # Initialize RAG system with the document
    # Ensure test.txt exists and is uploaded
    document_path = "extracted_bengali_text.txt"
    if not os.path.exists(document_path):
        print(f"Error: Document file '{document_path}' not found.")
        return

    rag = BanglaRAGSystem(document_path)

    # Test cases
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]

    for query in test_queries:
        answer = await rag.process_query(query)
        print(f"Query: {query}")
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    # Use nest_asyncio to run the async main function in a Jupyter environment
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
