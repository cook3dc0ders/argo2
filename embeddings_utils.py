# embeddings_utils.py
"""
Embedding utilities for ARGO RAG system.
Windows-compatible version with updated ChromaDB configuration.
"""
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
import logging
from config import CHROMA_DIR, EMBEDDING_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model_name=EMBEDDING_MODEL, persist_dir=CHROMA_DIR):
        """Initialize the embedding manager with sentence transformers and ChromaDB"""
        self.model_name = model_name
        self.persist_dir = Path(persist_dir)
        
        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize sentence transformer model
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("✅ Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
        
        try:
            # Initialize ChromaDB client with new configuration approach
            logger.info(f"Initializing ChromaDB with persist directory: {self.persist_dir}")
            
            # Use the new PersistentClient approach
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            logger.info("✅ ChromaDB client initialized")
            
            # Create or get collection
            self._initialize_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _initialize_collection(self):
        """Initialize or get the ChromaDB collection"""
        collection_name = "argo_profiles"
        
        try:
            # Try to get existing collection
            self.col = self.client.get_collection(collection_name)
            logger.info(f"✅ Using existing collection: {collection_name}")
            
        except Exception:
            # Create new collection if it doesn't exist
            try:
                logger.info(f"Creating new collection: {collection_name}")
                
                # Create embedding function
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.model_name
                )
                
                self.col = self.client.create_collection(
                    name=collection_name, 
                    embedding_function=ef
                )
                logger.info("✅ New collection created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                raise

    def embed_text(self, texts):
        """Generate embeddings for a list of texts"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def add_documents(self, ids, metadatas, texts):
        """Add documents to the ChromaDB collection"""
        try:
            # Validate inputs
            if len(ids) != len(metadatas) or len(ids) != len(texts):
                raise ValueError("ids, metadatas, and texts must have the same length")
            
            # Filter out any None values or empty strings
            valid_docs = []
            for i, (doc_id, metadata, text) in enumerate(zip(ids, metadatas, texts)):
                if doc_id and text and text.strip():
                    valid_docs.append((doc_id, metadata, text))
                else:
                    logger.warning(f"Skipping invalid document at index {i}")
            
            if not valid_docs:
                logger.warning("No valid documents to add")
                return
            
            # Unpack valid documents
            valid_ids, valid_metadatas, valid_texts = zip(*valid_docs)
            
            # Add to collection
            self.col.add(
                documents=list(valid_texts),
                metadatas=list(valid_metadatas),
                ids=list(valid_ids)
            )
            
            logger.info(f"✅ Added {len(valid_docs)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def query(self, query_text, n_results=5):
        """Query the ChromaDB collection for similar documents"""
        try:
            if not query_text or not query_text.strip():
                raise ValueError("Query text cannot be empty")
            
            # Perform query
            results = self.col.query(
                query_texts=[query_text],
                n_results=min(n_results, 100)  # Limit to reasonable number
            )
            
            logger.info(f"✅ Query completed, found {len(results.get('ids', [[]])[0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise

    def get_collection_stats(self):
        """Get statistics about the collection"""
        try:
            count = self.col.count()
            return {
                "total_documents": count,
                "collection_name": self.col.name,
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def delete_collection(self):
        """Delete the entire collection (use with caution!)"""
        try:
            self.client.delete_collection("argo_profiles")
            logger.info("✅ Collection deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def reset_collection(self):
        """Reset the collection (delete and recreate)"""
        try:
            self.delete_collection()
            self._initialize_collection()
            logger.info("✅ Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

# Utility functions for testing
def test_embeddings():
    """Test the embedding functionality"""
    try:
        print("🧪 Testing embedding functionality...")
        
        em = EmbeddingManager()
        
        # Test embedding generation
        test_texts = [
            "Temperature profile at 25.5N, 80W with 50 levels",
            "Salinity measurements in Atlantic Ocean",
            "Deep oxygen profile from Pacific float"
        ]
        
        embeddings = em.embed_text(test_texts)
        print(f"✅ Generated embeddings: shape {embeddings.shape}")
        
        # Test document addition
        test_ids = ["test_1", "test_2", "test_3"]
        test_metadatas = [
            {"lat": 25.5, "lon": -80.0, "type": "temperature"},
            {"lat": 35.0, "lon": -40.0, "type": "salinity"},
            {"lat": 10.0, "lon": -150.0, "type": "oxygen"}
        ]
        
        em.add_documents(test_ids, test_metadatas, test_texts)
        print("✅ Documents added to collection")
        
        # Test querying
        results = em.query("temperature in Atlantic", n_results=2)
        print(f"✅ Query completed, found {len(results['ids'][0])} results")
        
        # Show stats
        stats = em.get_collection_stats()
        print(f"✅ Collection stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_embeddings()