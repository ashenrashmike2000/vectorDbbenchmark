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

        # 1. Create ID Array (Converted to STRING to support UUIDs later)
        # FIX: Converted int IDs to string so the schema matches the CRUD UUIDs
        pa_ids = pa.array([str(i) for i in vector_ids])

        # 2. Create Vector Array (FixedSizeList)
        flat_vectors = vectors.reshape(-1)
        pa_vectors = pa.FixedSizeListArray.from_arrays(pa.array(flat_vectors), dimensions)

        # 3. Create Table
        # FIX: Changed "id" field to pa.string()
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimensions))
        ])

        data = pa.Table.from_arrays([pa_ids, pa_vectors], schema=schema)

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
        # FIX: Reduced default 'm' to a more standard value (16) to improve build speed
        # Optimized for GIST1M (High Dimensionality)
        target_m = params.get("m", 16)

        def get_valid_m(dim, target):
            if dim % target == 0: return target
            for i in range(target, 1, -1):
                if dim % i == 0: return i
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
            num_sub_vectors=num_sub_vectors,
            replace=True # Overwrite existing index
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
        # FIX: Increased default nprobes to 50 to improve recall
        nprobes = params.get("nprobes", 50)

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

            # FIX: Convert string IDs back to Integers for metrics calculation
            # If conversion fails (e.g. UUIDs), default to 0
            try:
                indices = [int(i) for i in results["id"].values]
            except ValueError:
                indices = [0] * len(results)

            dists = results["_distance"].values.tolist()

            all_indices.append(indices)
            all_distances.append(dists)

        return (
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            latencies
        )

    # Standard Bulk Interface
    def insert(self, vectors: NDArray[np.float32], metadata=None, ids=None) -> float:
        if ids is None: ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))
        # Ensure IDs are strings
        data = [{"id": str(id), "vector": vec} for id, vec in zip(ids, vectors)]
        start = time.perf_counter()
        self._table.add(data)
        self._num_vectors += len(vectors)
        return time.perf_counter() - start

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata=None) -> float:
        self.delete(ids)
        return self.insert(vectors, metadata, ids)

    def delete(self, ids: List[int]) -> float:
        start = time.perf_counter()
        # IDs are strings now
        ids_str = ", ".join([f"'{i}'" for i in ids])
        self._table.delete(f"id IN ({ids_str})")
        self._num_vectors -= len(ids)
        return time.perf_counter() - start

    def get_index_stats(self) -> Dict[str, Any]:
        if not self._table: return {}

        # Calculate size of the specific table directory
        import os
        total_size = 0
        table_path = os.path.join(self._db_path, f"{self._table_name}.lance")

        if os.path.exists(table_path):
            for dirpath, _, filenames in os.walk(table_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)

        return {
            "num_vectors": len(self._table),
            "dimensions": self._dimensions,
            "index_size_bytes": total_size  # <--- This fixes the "0 bytes" error for LanceDB
        }

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params = params

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params

    # === FIXED: Single-Item Wrappers for Benchmarking ===

    def insert_one(self, id: str, vector: np.ndarray):
        """Inserts a single vector."""
        # FIX: Removed 'text' field (it was not in schema)
        # FIX: Cast id to str
        data = [{"id": str(id), "vector": vector}]
        self._table.add(data)

    def delete_one(self, id: str):
        """Deletes a single vector by ID."""
        # FIX: Cast id to str. Query works now because schema 'id' is string.
        self._table.delete(f"id = '{str(id)}'")

    def update_one(self, id: str, vector: np.ndarray):
        """Updates a single vector (Delete + Insert)."""
        self.delete_one(id)
        self.insert_one(id, vector)