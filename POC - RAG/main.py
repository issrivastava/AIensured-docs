import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import g4f

class AdvancedRAGSystem:
    def __init__(self, knowledge_base_path: str):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.kb_path = knowledge_base_path
        self.documents = []
        self.index = None
        self._build_vector_index()

    def _build_vector_index(self):
        with open(self.kb_path, "r") as file:
            self.documents = [line.strip() for line in file if line.strip()]
            
        if not self.documents:
            raise ValueError("Knowledge base file is empty.")

        embeddings = self.encoder.encode(self.documents, convert_to_numpy=True).astype('float32')
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 1) -> str:
        query_vector = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        best_match_idx = indices[0][0]
        
        if best_match_idx != -1:
            return self.documents[best_match_idx]
        return "[No secure context found]"

    def augment_and_generate(self, query: str, context: str) -> str:
        prompt = f"""You are a precise database helper. Answer the user's question using ONLY the provided Context block.

Rules:
1. Provide a direct, plain answer based on the context. 
2. Do NOT say "Based on the context...", do NOT explain your reasoning, and do NOT talk about the prompt.
3. If the question asks "what is [X]" and the context says "[Y] is [X]", answer with "[Y]".
4. If the context does not contain the answer, reply exactly with: "I don't know."

Context:
{context}

User Question: {query}
Answer:"""

        response = g4f.ChatCompletion.create(
            model=g4f.models.default,
            messages=[{"role": "user", "content": prompt}],
        )
        return response

    def query_pipeline(self, user_query: str) -> str:
        try:
            context = self.retrieve(user_query, top_k=1)
            output = self.augment_and_generate(user_query, context)
            return output
        except Exception as error:
            return f"Error: {str(error)}"


if __name__ == "__main__":
    rag_engine = AdvancedRAGSystem(knowledge_base_path="knowledge.txt")
    print("\nSystem Ready. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.strip().lower() == 'exit':
            break
            
        if not user_input.strip():
            continue
            
        final_response = rag_engine.query_pipeline(user_input)
        print(f"[System Output]: {final_response}")