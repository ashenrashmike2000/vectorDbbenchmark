"""
Milvus vector database adapter.
"""

import time
import uuid
import warnings
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
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusClient,
        connections,
        utility,
        MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")

@register_database("milvus")
class MilvusAdapter(VectorDBInterface):
    """Milvus vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus not installed. Install with: pip install pymilvus")
        super().__init__(config)
        self._collection: Optional[Collection] = None
        self._collection_name: str = ""
        self._search_params: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "milvus"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="milvus",
            display_name="Milvus",
            version="2.3.0",
            type=DatabaseType.DATABASE,
            language="Go/C++",
            license="Apache-2.0",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[IndexType.HNSW],
            supports_filtering=True,
            supports_hybrid_search=False,
            supports_gpu=True,
            is_distributed=True,
        )

    def connect(self) -> None:
        """Establish connection to Milvus."""
        conn_config = self.config.get("connection", {})
        host = conn_config.get("host", "localhost")
        port = conn_config.get("port", 19530)

        try:
            connections.connect(
                alias="default",
                host=host,
                port=port,
                timeout=300
            )
            self._is_connected = True
        except Exception as e:
            if connections.has_connection("default"):
                self._is_connected = True
            else:
                raise e

    def disconnect(self) -> None:
        try:
            connections.disconnect("default")
        except Exception:
            pass
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

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimensions)
        ]

        schema = CollectionSchema(fields, "Benchmark collection")
        self._collection = Collection(self._collection_name, schema)

        print(f"🚀 Milvus: Inserting {n_vectors} vectors in batches...")

        insert_ids = ids if ids is not None else list(range(n_vectors))
        batch_size = 10000

        for i in range(0, n_vectors, batch_size):
            end = i + batch_size
            batch_ids = insert_ids[i:end]
            batch_vectors = vectors[i:end]

            self._collection.insert([batch_ids, batch_vectors])
            print(f"   Inserted batch {i} to {min(end, n_vectors)}", end="\r")

        self._collection.flush()
        print("\n✅ Insertion complete.")

        metric_map = {
            DistanceMetric.L2: "L2",
            DistanceMetric.COSINE: "COSINE",
            DistanceMetric.IP: "IP"
        }

        idx_type_str = "HNSW"
        if hasattr(index_config.type, "name"):
             idx_type_str = index_config.type.name
        elif isinstance(index_config.type, str):
             idx_type_str = index_config.type

        if "IVF" in idx_type_str.upper() and "FLAT" in idx_type_str.upper():
            idx_type_str = "IVF_FLAT"
        elif "HNSW" in idx_type_str.upper():
            idx_type_str = "HNSW"

        idx_params = {
            "metric_type": metric_map.get(distance_metric, "L2"),
            "index_type": idx_type_str,
            "params": index_config.params
        }

        print(f"🔨 Milvus: Building index ({idx_type_str})...")
        self._collection.create_index("vector", idx_params)
        self._collection.load()

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        if self._collection_name and utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
        self._collection = None
        self._collection_name = ""

    def save_index(self, path: str) -> None:
        if self._collection:
            self._collection.flush()

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
            raise RuntimeError("Collection not loaded")

        self.validate_vectors(queries)

        metric_map = {
            DistanceMetric.L2: "L2",
            DistanceMetric.COSINE: "COSINE",
            DistanceMetric.IP: "IP"
        }
        current_metric = metric_map.get(self._distance_metric, "L2")

        if not search_params:
            index_type_guess = "HNSW"
            if hasattr(self, "_index_config") and self._index_config:
                if hasattr(self._index_config.type, "name"):
                    index_type_guess = self._index_config.type.name
                elif isinstance(self._index_config.type, str):
                    index_type_guess = self._index_config.type

            search_params = {"metric_type": current_metric, "params": {}}
            if "HNSW" in str(index_type_guess).upper():
                search_params["params"] = {"ef": max(k * 2, 64)}
            else:
                search_params["params"] = {"nprobe": 10}
        else:
            if "metric_type" not in search_params:
                search_params["metric_type"] = current_metric

        latencies = []
        all_indices = []
        all_distances = []

        for query in queries:
            start_q = time.perf_counter()
            res = self._collection.search(
                data=[query],
                anns_field="vector",
                param=search_params,
                limit=k,
                expr=None
            )
            latencies.append((time.perf_counter() - start_q) * 1000)
            hits = res[0]
            all_indices.append(hits.ids)
            all_distances.append(hits.distances)

        return (
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            latencies
        )

    def insert(self, vectors: NDArray[np.float32], metadata=None, ids=None) -> float:
        if ids is None: ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))
        start = time.perf_counter()
        self._collection.insert([ids, vectors])
        return time.perf_counter() - start

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata=None) -> float:
        self.delete(ids)
        return self.insert(vectors, metadata, ids)

    def delete(self, ids: List[int]) -> float:
        start = time.perf_counter()
        expr = f"id in {ids}"
        self._collection.delete(expr)
        self._num_vectors -= len(ids)
        return time.perf_counter() - start

    def get_index_stats(self) -> Dict[str, Any]:
        if not self._collection: return {}
        
        self._collection.flush()
        num_vectors = self._collection.num_entities
        total_size = 0

        # Try modern .stats() method first
        try:
            stats = self._collection.stats()
            if "partitions_stats" in stats:
                for partition in stats["partitions_stats"]:
                    if "segments_stats" in partition:
                        for segment in partition["segments_stats"]:
                            total_size += int(segment.get("data_size", 0))
        except (MilvusException, AttributeError):
            total_size = 0 # Reset size if first method fails

        # If .stats() fails or returns 0, try older segment info method
        if total_size == 0:
            try:
                segments_info = utility.get_query_segment_info(self._collection_name)
                total_size = sum(segment.mem_size for segment in segments_info)
            except (MilvusException, AttributeError):
                total_size = 0 # Reset size if second method also fails

        # If all API calls fail, fall back to estimation
        if total_size == 0 and num_vectors > 0:
            print(f"⚠️ Milvus: Could not get detailed stats via API. Falling back to estimation.")
            total_size = num_vectors * self._dimensions * 4  # float32

        return {
            "num_vectors": num_vectors,
            "dimensions": self._dimensions,
            "index_size_bytes": total_size
        }

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params = params

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params

    def insert_one(self, id: str, vector: np.ndarray):
        # Milvus requires int64 for primary key. If ID is not int, hash it.
        milvus_id = int(id) if str(id).isdigit() else hash(id) % (2**63 - 1)
        self._collection.insert([[milvus_id], [vector]])
        self._collection.flush()

    def update_one(self, id: str, vector: np.ndarray):
        milvus_id = int(id) if str(id).isdigit() else hash(id) % (2**63 - 1)
        expr = f"id in [{milvus_id}]"
        self._collection.delete(expr)