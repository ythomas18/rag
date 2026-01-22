import os
import re
import json
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jConnection:
    """Manages Neo4j database connection and operations."""
    
    def __init__(self):
        self.driver = None
        self.use_http_api = False
        self.http_base_url = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        # First, try the standard Bolt driver
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            print(f"✅ Connected to Neo4j via Bolt at {NEO4J_URI}")
            return
        except Exception as e:
            print(f"⚠️ Bolt connection failed: {e}")
            print("   Trying HTTP API fallback...")
            self.driver = None  # Crucial: Reset driver so we use HTTP fallback
        
        # Fallback to HTTP API (for firewall/proxy restrictions)
        try:
            # Convert neo4j+s:// URI to https://
            http_url = NEO4J_URI.replace("neo4j+s://", "https://").replace("neo4j://", "http://")
            self.http_base_url = http_url
            
            # Test HTTP API connectivity
            query_url = f"{http_url}/db/neo4j/query/v2"
            resp = requests.post(
                query_url,
                headers={"Content-Type": "application/json"},
                json={"statement": "RETURN 1 as test"},
                auth=HTTPBasicAuth(NEO4J_USER, NEO4J_PASSWORD),
                timeout=30
            )
            
            if resp.status_code in [200, 202]:
                self.use_http_api = True
                print(f"✅ Connected to Neo4j via HTTP API at {http_url}")
            else:
                print(f"⚠️ HTTP API returned status {resp.status_code}")
                self.http_base_url = None
        except Exception as e:
            print(f"⚠️ Failed to connect to Neo4j via HTTP API: {e}")
            self.http_base_url = None
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def is_connected(self) -> bool:
        return self.driver is not None or self.use_http_api
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        if self.driver:
            # Use Bolt driver
            try:
                with self.driver.session() as session:
                    result = session.run(query, parameters or {})
                    return [record.data() for record in result]
            except Exception as e:
                print(f"Query error (Bolt): {e}")
                return []
        
        elif self.use_http_api and self.http_base_url:
            # Use HTTP API
            try:
                query_url = f"{self.http_base_url}/db/neo4j/query/v2"
                
                # Use HTTP API with parameters
                payload = {
                    "statement": query,
                    "parameters": parameters or {}
                }
                
                resp = requests.post(
                    query_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    auth=HTTPBasicAuth(NEO4J_USER, NEO4J_PASSWORD),
                    timeout=30
                )
                
                if resp.status_code in [200, 202]:
                    data = resp.json()
                    # Parse the response format
                    if "data" in data and "values" in data["data"]:
                        columns = data["data"].get("fields", [])
                        rows = data["data"].get("values", [])
                        return [dict(zip(columns, row)) for row in rows]
                    return []
                else:
                    print(f"HTTP API error: {resp.status_code} - {resp.text[:200]}")
                    return []
            except Exception as e:
                print(f"Query error (HTTP): {e}")
                return []
        
        return []
    
    def create_node(self, label: str, properties: Dict) -> Optional[int]:
        """Create a node with given label and properties."""
        query = f"CREATE (n:{label} $props) RETURN id(n) as node_id"
        result = self.execute_query(query, {"props": properties})
        return result[0]["node_id"] if result else None
    
    def create_relationship(self, from_id: int, to_id: int, rel_type: str, properties: Dict = None):
        """Create a relationship between two nodes."""
        query = """
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        CREATE (a)-[r:%s $props]->(b)
        RETURN type(r) as rel_type
        """ % rel_type
        return self.execute_query(query, {"from_id": from_id, "to_id": to_id, "props": properties or {}})
    
    def search_nodes(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Search nodes by keyword in their properties."""
        query = """
        MATCH (n)
        WHERE any(key IN keys(n) WHERE toString(n[key]) CONTAINS $keyword)
        RETURN n, labels(n) as labels, id(n) as node_id
        LIMIT $limit
        """
        return self.execute_query(query, {"keyword": keyword, "limit": limit})
    
    def get_node_relationships(self, node_id: int, depth: int = 2) -> List[Dict]:
        """Get all relationships for a node up to specified depth."""
        query = """
        MATCH path = (n)-[*1..%d]-(m)
        WHERE id(n) = $node_id
        RETURN 
            [rel in relationships(path) | {type: type(rel), props: properties(rel)}] as relationships,
            [node in nodes(path) | {id: id(node), labels: labels(node), props: properties(node)}] as nodes
        LIMIT 50
        """ % depth
        return self.execute_query(query, {"node_id": node_id})

class GraphRAG:
    """Graph-based RAG using Neo4j for knowledge graph storage and retrieval."""
    
    def __init__(self, llm=None):
        self.neo4j = Neo4jConnection()
        self.llm = llm
        
        if not self.neo4j.is_connected():
            print("⚠️ Neo4j not connected. GraphRAG will be disabled.")
    
    def is_available(self) -> bool:
        return self.neo4j.is_connected()
    
    
    def build_graph(self, documents: List[Document]) -> Dict[str, int]:
        """Build knowledge graph from documents using LangChain LLMGraphTransformer."""
        if not self.is_available() or not self.llm:
            return {"entities": 0, "relations": 0}
        
        try:
            from langchain_experimental.graph_transformers import LLMGraphTransformer
        except ImportError:
            print("⚠️ langchain-experimental not installed. Skipping graph build.")
            return {"entities": 0, "relations": 0}
            
        print("🧠 Transforming documents to graph structure with LLM...")
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        
        # Convert documents to graph documents
        # Note: We process limited chunks to avoid context limits
        graph_documents = llm_transformer.convert_to_graph_documents(documents[:50])
        
        total_entities = 0
        total_relations = 0
        entity_id_map = {}
        
        print(f"🕸️ Inserting {len(graph_documents)} graph documents into Neo4j...")
        
        for graph_doc in graph_documents:
            # 1. Insert Nodes
            for node in graph_doc.nodes:
                entity_name = node.id
                if entity_name.lower() not in entity_id_map:
                    node_id = self.neo4j.create_node(
                        label=node.type,
                        properties={
                            "name": entity_name,
                            "source": graph_doc.source.metadata.get("source", "unknown")
                        }
                    )
                    if node_id is not None:
                        entity_id_map[entity_name.lower()] = node_id
                        total_entities += 1
            
            # 2. Insert Relationships
            for rel in graph_doc.relationships:
                from_name = rel.source.id.lower()
                to_name = rel.target.id.lower()
                rel_type = rel.type.replace(" ", "_").upper()
                
                if from_name in entity_id_map and to_name in entity_id_map:
                    self.neo4j.create_relationship(
                        from_id=entity_id_map[from_name],
                        to_id=entity_id_map[to_name],
                        rel_type=rel_type,
                        properties={} # LLMGraphTransformer relations usually don't have extra props by default
                    )
                    total_relations += 1
                    
        return {"entities": total_entities, "relations": total_relations}
    
    def query_graph(self, question: str) -> str:
        """Query the knowledge graph based on the question."""
        if not self.is_available():
            return ""
        
        keywords = self._extract_keywords(question)
        graph_context = []
        seen_nodes = set()
        
        for keyword in keywords:
            nodes = self.neo4j.search_nodes(keyword, limit=5)
            for node_data in nodes:
                node_id = node_data.get("node_id")
                if node_id in seen_nodes: continue
                seen_nodes.add(node_id)
                
                node_props = node_data.get("n", {})
                labels = node_data.get("labels", [])
                
                node_info = f"[{'/'.join(labels)}] {node_props.get('name', 'Unknown')}"
                if node_props.get("description"):
                    node_info += f": {node_props.get('description')}"
                graph_context.append(node_info)
                
                relationships = self.neo4j.get_node_relationships(node_id, depth=1)
                for rel_data in relationships[:5]:
                    for rel in rel_data.get("relationships", []):
                        rel_info = f"  -> {rel.get('type', 'RELATED')}"
                        if rel.get("props", {}).get("description"):
                            rel_info += f": {rel['props']['description']}"
                        graph_context.append(rel_info)
        
        if graph_context:
            return "Knowledge Graph Context:\n" + "\n".join(graph_context[:20])
        return ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'who', 'what', 'where', 'when', 'why', 'how', 'which', 'that', 'this', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'with', 'from', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        unique_keywords = []
        for kw in keywords:
            if kw not in unique_keywords: unique_keywords.append(kw)
        return unique_keywords[:10]
