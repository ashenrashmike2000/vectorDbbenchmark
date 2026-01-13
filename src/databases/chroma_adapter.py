"""
Chroma vector database adapter.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.base import VectorDBInterface
from src.core.types import (
    DatabaseInfo,
    DatabaseType,
    DistanceMetric,
    FilterCondition,
    IndexConfig,
    IndexType,
)
from src.databases.factory import register_database

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None


@register_database("chroma")
class ChromaAdapter(VectorDBInterface):
    """Chroma vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma not installed. Install with: pip install chromadb")
        super().__init__(config)
        self._client = None
        self._collection = None
        self._collection_name: str = ""
        self._search_params: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "chroma"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="chroma",
            display_name="Chroma",
            version="0.4.x",
            type=DatabaseType.DATABASE,
            language="Python",
            license="Apache-2.0",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[IndexType.HNSW],
            supports_filtering=True,
            supports_hybrid_search=False,
            supports_gpu=False,
            is_distributed=False,
        )

    def connect(self) -> None:
        """Establish connection to Chroma."""
        conn_config = self.config.get("connection", {})

        if conn_config.get("mode") == "http":
            host = conn_config.get("http", {}).get("host", "localhost")
            port = conn_config.get("http", {}).get("port", 8000)
            self._client = chromadb.HttpClient(host=host, port=port)
        else:
            path = conn_config.get("persistent", {}).get("path", "./chroma_db")
            self._client = chromadb.PersistentClient(path=path)

        self._is_connected = True

    def disconnect(self) -> None:
        self._client = None
        self._is_connected = False

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Create a collection and index vectors."""
        self.validate_vectors(vectors)

        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        prefix = self.config.get("collection", {}).get("name_prefix", "benchmark")
        self._collection_name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        start_time = time.perf_counter()

        # 1. Configure Metric
        metric_map = {
            DistanceMetric.L2: "l2",
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.IP: "ip"
        }
        hnsw_space = metric_map.get(distance_metric, "l2")

        # 2. Configure HNSW Params (Realism Fix)
        # We now pass 'M' and 'ef_construction' from your config to Chroma
        collection_metadata = {"hnsw:space": hnsw_space}

        if "ef_construct" in index_config.params:
            collection_metadata["hnsw:construction_ef"] = index_config.params["ef_construct"]
        if "m" in index_config.params:
            collection_metadata["hnsw:M"] = index_config.params["m"]

        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata=collection_metadata
        )

        # =========================================================
        # FIX: Safe Batch Insert (Works for SIFT & MSMARCO)
        # =========================================================
        print(f"🚀 Chroma: Inserting {n_vectors} vectors in batches...")

        vector_ids = [str(i) for i in (ids if ids else range(n_vectors))]
        
        # Batch size 2000 is safe for 768d vectors (approx 6MB payload)
        batch_size = 2000

        for i in range(0, n_vectors, batch_size):
            end = min(i + batch_size, n_vectors)

            # FIX: Convert ONLY the current batch to list to save memory
            # Converting the entire 1M vectors to list at once causes OOM
            batch_embeddings = vectors[i:end].tolist()

            self._collection.add(
                ids=vector_ids[i:end],
                embeddings=batch_embeddings,
                metadatas=metadata[i:end] if metadata else None
            )

            if i % 10000 == 0:
                print(f"   Processed {end}/{n_vectors} vectors...", end="\r")

        print(f"\n✅ Chroma: Insertion complete.")
        # =========================================================

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        if self._collection_name:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
        self._collection = None
        self._collection_name = ""

    def save_index(self, path: str) -> None:
        pass

    def load_index(self, path: str) -> float:
        return 0.0

    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:

        if not self._collection:
            raise RuntimeError("Collection not created")

        self.validate_vectors(queries)

        latencies = []
        all_indices = []
        all_distances = []

        query_list = queries.tolist()

        for q in query_list:
            start_q = time.perf_counter()

            results = self._collection.query(
                query_embeddings=[q],
                n_results=k,
                include=["distances"]
            )

            latencies.append((time.perf_counter() - start_q) * 1000)

            # Chroma IDs are strings, convert back to int
            ids = [int(i) for i in results['ids'][0]]
            dists = results['distances'][0]

            all_indices.append(ids)
            all_distances.append(dists)

        return (
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            latencies
        )

    # ==========================================
    #  Added methods for Runner Compatibility
    # ==========================================

    def insert_one(self, id: str, vector: np.ndarray):
        """Inserts a single object (Required for benchmark runner)."""
        self._collection.add(
            ids=[str(id)],
            embeddings=[vector.tolist()]
        )

    def update_one(self, id: str, vector: np.ndarray):
        """Updates a single object (Required for benchmark runner)."""
        self._collection.update(
            ids=[str(id)],
            embeddings=[vector.tolist()]
        )

    def delete_one(self, id: str):
        """Deletes a single object (Required for benchmark runner)."""
        self._collection.delete(ids=[str(id)])

    # ==========================================

    # Standard Bulk CRUD (kept for interface compliance)
    def insert(self, vectors: NDArray[np.float32], metadata=None, ids=None) -> float:
        if ids is None: ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))
        str_ids = [str(i) for i in ids]
        start = time.perf_counter()
        self._collection.add(embeddings=vectors.tolist(), ids=str_ids, metadatas=metadata)
        self._num_vectors += len(vectors)
        return time.perf_counter() - start

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata=None) -> float:
        str_ids = [str(i) for i in ids]
        start = time.perf_counter()
        self._collection.update(ids=str_ids, embeddings=vectors.tolist(), metadatas=metadata)
        return time.perf_counter() - start

    def delete(self, ids: List[int]) -> float:
        str_ids = [str(i) for i in ids]
        start = time.perf_counter()
        self._collection.delete(ids=str_ids)
        self._num_vectors -= len(ids)
        return time.perf_counter() - start

    def get_index_stats(self) -> Dict[str, Any]:
        if not self._collection: return {}
        return {"num_vectors": self._collection.count(), "dimensions": self._dimensions}

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params = params

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params