import re
from typing import List, Tuple, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY, GROQ_MODEL
from neo4j_connect import GraphRAG
from qdrant_connect import QdrantConnector
from document_utils import load_document, split_into_chunks

class HybridRetriever:
    """Core RAG logic with routing."""
    
    def __init__(self, use_neo4j: bool = False):
        self.use_neo4j = use_neo4j
        
        if not GROQ_API_KEY or "your_groq_api_key" in GROQ_API_KEY:
            raise ValueError("Please set a valid GROQ_API_KEY in your .env file")
            
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY, 
            model_name=GROQ_MODEL,
            temperature=0
        )
        
        # Initialize connectors
        self.qdrant = QdrantConnector()
        self.retriever = self.qdrant.get_retriever()
        
        self.graph_rag = None
        if self.use_neo4j:
            self.graph_rag = GraphRAG(llm=self.llm)

        # Routing patterns
        self.patterns = {
            'qdrant': [
                r'what is', r'define', r'price', r'prix', r'cost', r'tarif', 
                r'spec', r'feature', r'combien', r'c\'est quoi'
            ],
            'neo4j': [
                r'related', r'history', r'evolution', r'connection', r'link',
                r'historique', r'lien', r'impact'
            ]
        }

    def route_query(self, query: str) -> str:
        """Determine which retrieval method to use."""
        query_lower = query.lower()
        
        for method, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return method
                    
        return 'hybrid'

    def retrieve(self, query: str) -> Tuple[List[Any], str]:
        """Retrieve context based on routing."""
        route = self.route_query(query)
        chunks = []
        
        if route in ['qdrant', 'hybrid']:
            # Vector search
            chunks.extend(self.retriever.invoke(query))
            
        if route in ['neo4j', 'hybrid'] and self.use_neo4j and self.graph_rag and self.graph_rag.is_available():
            # Graph search (returning as text/string for now, wrapped in list for consistency)
            graph_context = self.graph_rag.query_graph(query)
            if graph_context:
                # We wrap it in a mock object or string to treat as a chunk
                from langchain_core.documents import Document
                chunks.append(Document(page_content=f"Generic Graph Context: {graph_context}", metadata={"source": "neo4j"}))

        return chunks, route

    def generate_answer(self, query: str, context_chunks: List[Any], route: str) -> str:
        """Generate answer from context."""
        context_text = "\n\n".join([doc.page_content for doc in context_chunks])
        
        if not context_text:
            context_text = "No relevant information found."

        system_prompt = (
            f"You are an intelligent assistant using {route} retrieval strategy. "
            "Use the provided context to answer the user's question.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"input": query, "context": context_text})

    def ingest(self, file_paths: List[str]) -> dict:
        """Ingest documents into enabled stores."""
        docs = []
        for path in file_paths:
            docs.extend(load_document(path))
            
        chunks = split_into_chunks(docs)
        vector_count = self.qdrant.index_documents(chunks)
        
        result = {
            "vector_chunks": vector_count,
            "graph_entities": 0,
            "graph_relations": 0
        }
        
        if self.use_neo4j and self.graph_rag and self.graph_rag.is_available():
            print("Building Knowledge Graph...")
            stats = self.graph_rag.build_graph(docs)
            result.update({
                "graph_entities": stats["entities"],
                "graph_relations": stats["relations"]
            })
            
        return result

