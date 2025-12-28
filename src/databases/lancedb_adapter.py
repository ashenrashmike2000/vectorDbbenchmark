"""
LanceDB vector database adapter.
"""

import time
import uuid
import shutil
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd

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
    import lancedb
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False


@register_database("lancedb")
class LanceDBAdapter(VectorDBInterface):
    """LanceDB vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB not installed. Install with: pip install lancedb")
        super().__init__(config)
        self._db = None
        self._table = None
        self._table_name: str = ""
        self._db_path = config.get("connection", {}).get("uri", "./lancedb_data")

    @property
    def name(self) -> str:
        return "lancedb"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="lancedb",
            display_name="LanceDB",
            version="0.4.x",
            type=DatabaseType.DATABASE,
            language="Rust/Python",
            license="Apache-2.0",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE],
            # FIX: Removed invalid IndexType to prevent AttributeError
            supported_index_types=[],
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_gpu=True,
            is_distributed=False,
        )

    def connect(self) -> None:
        """Connect to LanceDB (Embedded)."""
        self._db = lancedb.connect(self._db_path)
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Create a table and index vectors."""
        self.validate_vectors(vectors)

        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        prefix = self.config.get("collection", {}).get("name_prefix", "benchmark")
        self._table_name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        start_time = time.perf_counter()

        print(f"🚀 LanceDB: Ingesting {n_vectors} vectors (PyArrow Table Mode)...")

        vector_ids = ids if ids is not None else list(range(n_vectors))

        # === FIX: Convert to PyArrow Table ===
        # This handles the memory efficiently and creates the correct schema
        # 1. Create ID Array
        pa_ids = pa.array(vector_ids)

        # 2. Create Vector Array (FixedSizeList)
        # Flatten numpy array (Zero-copy view if possible)
        flat_vectors = vectors.reshape(-1)
        pa_vectors = pa.FixedSizeListArray.from_arrays(pa.array(flat_vectors), dimensions)

        # 3. Create Table
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dimensions))
        ])

        data = pa.Table.from_arrays([pa_ids, pa_vectors], schema=schema)
        # ======================================

        # Create Table
        if self._table_name in self._db.table_names():
            self._db.drop_table(self._table_name)

        self._table = self._db.create_table(self._table_name, data=data)

        # Create Index
        metric_map = {
            DistanceMetric.L2: "L2",
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.IP: "dot"
        }
        metric = metric_map.get(distance_metric, "L2")

        params = index_config.params
        target_m = params.get("m", 96)

        def get_valid_m(dim, target):
            # Try target first
            if dim % target == 0: return target
            # Try to find the largest divisor close to target
            for i in range(target, 1, -1):
                if dim % i == 0: return i
            # Fallback to small divisors
            for i in [16, 12, 8, 4, 2]:
                if dim % i == 0: return i
            return 1

        num_sub_vectors = get_valid_m(self._dimensions, target_m)
        num_partitions = params.get("nlist", 256)

        print(f"🔨 LanceDB: Building IVF-PQ (partitions={num_partitions}, sub_vectors={num_sub_vectors})...")

        self._table.create_index(
            metric=metric,
            vector_column_name="vector",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors
        )

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        if self._table_name and self._db:
            if self._table_name in self._db.table_names():
                self._db.drop_table(self._table_name)
        self._table = None
        self._table_name = ""

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

        if not self._table:
            raise RuntimeError("Table not created")

        self.validate_vectors(queries)

        params = search_params or {}
        nprobes = params.get("nprobes", 20)

        latencies = []
        all_indices = []
        all_distances = []

        for query in queries:
            start_q = time.perf_counter()

            results = self._table.search(query) \
                .metric(self._distance_metric.value if isinstance(self._distance_metric, DistanceMetric) else "L2") \
                .nprobes(nprobes) \
                .limit(k) \
                .to_pandas()

            latencies.append((time.perf_counter() - start_q) * 1000)

            indices = results["id"].values.tolist()
            dists = results["_distance"].values.tolist()

            all_indices.append(indices)
            all_distances.append(dists)

        return (
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            latencies
        )

    def insert(self, vectors: NDArray[np.float32], metadata=None, ids=None) -> float:
        if ids is None: ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))
        data = [{"id": id, "vector": vec} for id, vec in zip(ids, vectors)]
        start = time.perf_counter()
        self._table.add(data)
        self._num_vectors += len(vectors)
        return time.perf_counter() - start

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata=None) -> float:
        self.delete(ids)
        return self.insert(vectors, metadata, ids)

    def delete(self, ids: List[int]) -> float:
        start = time.perf_counter()
        ids_str = ", ".join(map(str, ids))
        self._table.delete(f"id IN ({ids_str})")
        self._num_vectors -= len(ids)
        return time.perf_counter() - start

    def get_index_stats(self) -> Dict[str, Any]:
        if not self._table: return {}
        return {"num_vectors": len(self._table), "dimensions": self._dimensions}

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params = params

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params

    # === NEW: Single-Item Wrappers for Benchmarking ===

    def insert_one(self, id: str, vector: np.ndarray):
        """Inserts a single vector."""
        # Wrap the single vector in a list and pass to LanceDB
        data = [{"id": id, "vector": vector, "text": "benchmark_update_test"}]
        self._table.add(data)

    def delete_one(self, id: str):
        """Deletes a single vector by ID."""
        # Note: We use quotes '{id}' because the ID is a string
        self._table.delete(f"id = '{id}'")

    def update_one(self, id: str, vector: np.ndarray):
        """Updates a single vector (Delete + Insert)."""
        # LanceDB (and many vector DBs) handles updates as delete-then-insert
        self.delete_one(id)
        self.insert_one(id, vector)