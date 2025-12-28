"""
Qdrant vector database adapter.
"""

import time
import uuid
import gc
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
from qdrant_client.models import Batch

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None


@register_database("qdrant")
class QdrantAdapter(VectorDBInterface):
    """Qdrant vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client is not installed.")

        super().__init__(config)
        self._client: Optional[QdrantClient] = None
        self._collection_name: str = ""
        self._search_params: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "qdrant"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="qdrant",
            display_name="Qdrant",
            version="1.7.0",
            type=DatabaseType.DATABASE,
            language="Rust",
            license="Apache-2.0",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[IndexType.HNSW],
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_gpu=False,
            is_distributed=True,
        )

    def connect(self) -> None:
        """Establish connection to Qdrant server."""
        conn_config = self.config.get("connection", {})

        if conn_config.get("type") == "memory":
            self._client = QdrantClient(":memory:")
        else:
            host = conn_config.get("host", "localhost")
            grpc_port = conn_config.get("grpc_port", 6334)
            http_port = conn_config.get("http_port", 6333)
            prefer_grpc = conn_config.get("prefer_grpc", True)

            # Infinite timeout for safety
            timeout = conn_config.get("timeout", None)

            self._client = QdrantClient(
                host=host,
                port=http_port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )

        self._is_connected = True

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
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
        """Create a Qdrant collection and index vectors."""
        self.validate_vectors(vectors)

        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        prefix = self.config.get("collection", {}).get("name_prefix", "benchmark")
        self._collection_name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        start_time = time.perf_counter()

        distance_map = {
            DistanceMetric.L2: models.Distance.EUCLID,
            DistanceMetric.COSINE: models.Distance.COSINE,
            DistanceMetric.IP: models.Distance.DOT,
        }
        qdrant_distance = distance_map.get(distance_metric, models.Distance.EUCLID)

        params = index_config.params
        hnsw_config = models.HnswConfigDiff(
            m=params.get("m", 16),
            ef_construct=params.get("ef_construct", 100),
            full_scan_threshold=params.get("full_scan_threshold", 10000),
            max_indexing_threads=params.get("max_indexing_threads", 0),
            on_disk=True, # Safety: Keep HNSW graph on disk too
        )

        # === HYBRID STRATEGY: QUANTIZATION + DISK ===
        quantization_config = None

        if "quantization" in index_config.params or index_config.quantization:
            quant_params = index_config.quantization or index_config.params.get("quantization", {})
            quantization_config = self._create_quantization_config(quant_params)
        elif dimensions > 512:
            print("⚡ Auto-enabling Scalar Quantization (Int8) for speed...")
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True # Keep compressed vectors in RAM (Small enough)
                )
            )

        # Configure vectors
        vectors_config = models.VectorParams(
            size=dimensions,
            distance=qdrant_distance,
            hnsw_config=hnsw_config,
            quantization_config=quantization_config,
            on_disk=True,  # <--- SAFETY: Force heavy storage to Disk
        )

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=vectors_config,
        )

        # Optimized Parallel Upload
        print(f"🚀 Using Optimized Parallel Upload for {n_vectors} vectors...")
        vector_ids = ids if ids is not None else list(range(n_vectors))

        self._client.upload_collection(
            collection_name=self._collection_name,
            vectors=vectors,
            payload=metadata,
            ids=vector_ids,
            parallel=2,
            wait=True
        )

        del vectors
        gc.collect()

        # === UPDATED WAIT LOOP (PRINTS STATUS) ===
        print("⏳ Waiting for Qdrant indexing...")
        while True:
            try:
                info = self._client.get_collection(self._collection_name)
                status = info.status

                if status == models.CollectionStatus.GREEN:
                    print("\n✅ Optimization complete. Collection is GREEN.")
                    break
                elif status == models.CollectionStatus.RED:
                    print(f"\n❌ CRITICAL: Qdrant Status is RED. Check 'docker logs qdrant_benchmark'.")
                    raise RuntimeError("Qdrant Collection Status became RED (Failed).")
                else:
                    # Status is likely YELLOW (Optimizing)
                    print(f"   Status: {status} (Optimizing)...", end="\r")

                time.sleep(5)
            except Exception as e:
                print(f"Warning: Could not check status: {e}")
                time.sleep(5)
        # =========================================

        self._num_vectors = n_vectors
        build_time = time.perf_counter() - start_time

        return build_time

    def _create_quantization_config(self, quant_params: Dict[str, Any]) -> Optional[models.QuantizationConfig]:
        if "scalar" in quant_params:
            scalar_params = quant_params["scalar"]
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=scalar_params.get("quantile", 0.99),
                    always_ram=scalar_params.get("always_ram", True),
                )
            )
        elif "product" in quant_params:
            product_params = quant_params["product"]
            return models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=getattr(models.CompressionRatio, product_params.get("compression", "x16").upper()),
                    always_ram=product_params.get("always_ram", True),
                )
            )
        elif "binary" in quant_params:
            binary_params = quant_params["binary"]
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=binary_params.get("always_ram", True),
                )
            )
        return None

    def delete_index(self) -> None:
        if self._client and self._collection_name:
            try:
                self._client.delete_collection(self._collection_name)
            except UnexpectedResponse:
                pass
        self._num_vectors = 0
        self._collection_name = ""

    def save_index(self, path: str) -> None:
        if not self._collection_name:
            raise RuntimeError("No collection to save")
        self._client.create_snapshot(collection_name=self._collection_name)

    def load_index(self, path: str) -> float:
        raise NotImplementedError("Use Qdrant's snapshot restore functionality.")

    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:
        if not self._collection_name:
            raise RuntimeError("Collection not created")

        self.validate_vectors(queries)

        params = search_params or {}
        ef = params.get("ef", params.get("efSearch", 128))
        search_params_obj = models.SearchParams(
            hnsw_ef=ef,
            exact=params.get("exact", False),
        )

        qdrant_filter = self._build_filter(filters) if filters else None

        reverse_sort = True
        if self._distance_metric == DistanceMetric.L2:
             reverse_sort = False

        n_queries = len(queries)
        all_indices = []
        all_distances = []
        latencies = []

        for query in queries:
            start_time = time.perf_counter()

            results_obj = self._client.query_points(
                collection_name=self._collection_name,
                query=query.tolist(),
                limit=k,
                search_params=search_params_obj,
                query_filter=qdrant_filter,
            )

            results = results_obj.points
            results.sort(key=lambda x: x.score, reverse=reverse_sort)

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

            indices = [r.id for r in results]
            distances = [r.score for r in results]

            while len(indices) < k:
                indices.append(-1)
                distances.append(float("inf"))

            all_indices.append(indices[:k])
            all_distances.append(distances[:k])

        return (
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            latencies,
        )

    def _build_filter(self, filters: List[FilterCondition]) -> Optional[models.Filter]:
        if not filters: return None
        must_conditions = []
        for f in filters:
            condition = self._convert_filter_condition(f)
            if condition: must_conditions.append(condition)
        if must_conditions: return models.Filter(must=must_conditions)
        return None

    def _convert_filter_condition(self, f: FilterCondition) -> Optional[models.Condition]:
        field = f.field
        op = f.operator.lower()
        value = f.value
        if op == "eq": return models.FieldCondition(key=field, match=models.MatchValue(value=value))
        elif op == "ne": return models.FieldCondition(key=field, match=models.MatchExcept(**{"except": [value]}))
        elif op == "gt": return models.FieldCondition(key=field, range=models.Range(gt=value))
        elif op == "gte": return models.FieldCondition(key=field, range=models.Range(gte=value))
        elif op == "lt": return models.FieldCondition(key=field, range=models.Range(lt=value))
        elif op == "lte": return models.FieldCondition(key=field, range=models.Range(lte=value))
        elif op == "in": return models.FieldCondition(key=field, match=models.MatchAny(any=value))
        return None

    def insert(self, vectors: NDArray[np.float32], metadata=None, ids=None) -> float:
        self.validate_vectors(vectors)
        if ids is None: ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))

        points = []
        for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
            payload = metadata[i] if metadata else {}
            points.append(models.PointStruct(id=vec_id, vector=vector.tolist(), payload=payload))

        start_time = time.perf_counter()
        self._client.upsert(collection_name=self._collection_name, points=points, wait=True)
        self._num_vectors += len(vectors)
        return time.perf_counter() - start_time

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata=None) -> float:
        self.validate_vectors(vectors)
        points = []
        for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
            payload = metadata[i] if metadata else None
            points.append(models.PointStruct(id=vec_id, vector=vector.tolist(), payload=payload))
        start_time = time.perf_counter()
        self._client.upsert(collection_name=self._collection_name, points=points, wait=True)
        return time.perf_counter() - start_time

    def delete(self, ids: List[int]) -> float:
        start_time = time.perf_counter()
        self._client.delete(collection_name=self._collection_name, points_selector=models.PointIdsList(points=ids), wait=True)
        self._num_vectors -= len(ids)
        return time.perf_counter() - start_time

    def get_index_stats(self) -> Dict[str, Any]:
        if not self._collection_name: return {}
        info = self._client.get_collection(self._collection_name)
        return {
            "num_vectors": info.points_count,
            "dimensions": info.config.params.vectors.size,
            "index_type": "HNSW",
            "distance_metric": self._distance_metric.value if self._distance_metric else None,
            "status": info.status.value,
            "optimizer_status": str(info.optimizer_status),
        }

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params.update(params)

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params.copy()

    # === NEW: Single-Item Wrappers for Benchmarking ===

    def insert_one(self, id: str, vector: np.ndarray):
        """Inserts/Upserts a single point."""
        try:
            # Qdrant supports both Int and UUID strings for IDs
            point_id = int(id) if str(id).isdigit() else id

            point = models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={"benchmark": "ops_test"}
            )

            self._client.upsert(
                collection_name=self._collection_name,
                points=[point],
                wait=True
            )
        except Exception as e:
            print(f"Qdrant insert_one failed: {e}")

    def delete_one(self, id: str):
        """Deletes a single point."""
        try:
            point_id = int(id) if str(id).isdigit() else id

            self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.PointIdsList(points=[point_id]),
                wait=True
            )
        except Exception:
            pass

    def update_one(self, id: str, vector: np.ndarray):
        """Updates a single point (Qdrant upsert handles this)."""
        self.insert_one(id, vector)