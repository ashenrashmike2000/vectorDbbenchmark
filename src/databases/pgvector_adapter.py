"""
pgvector PostgreSQL extension adapter.

pgvector enables vector similarity search in PostgreSQL, allowing
integration of vector search with existing SQL workflows.

Documentation: https://github.com/pgvector/pgvector
"""
import sys
import time
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
    import psycopg2
    from psycopg2.extras import execute_values
    from pgvector.psycopg2 import register_vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


@register_database("pgvector")
class PgvectorAdapter(VectorDBInterface):
    """pgvector PostgreSQL adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not PGVECTOR_AVAILABLE:
            raise ImportError("pgvector not installed. Install with: pip install pgvector psycopg2-binary")
        super().__init__(config)
        self._conn = None
        self._table_name: str = ""
        self._search_params: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "pgvector"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="pgvector",
            display_name="pgvector",
            version="0.6.0",
            type=DatabaseType.EXTENSION,
            language="C",
            license="PostgreSQL",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[IndexType.IVF, IndexType.HNSW],
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_gpu=False,
            is_distributed=False,
        )

    def connect(self) -> None:
        """Connect to PostgreSQL."""
        conn_config = self.config.get("connection", {})
        self._conn = psycopg2.connect(
            host=conn_config.get("host", "localhost"),
            port=conn_config.get("port", 5432),
            database=conn_config.get("database", "benchmark"),
            user=conn_config.get("user", "postgres"),
            password=conn_config.get("password", "postgres"),
        )

        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        register_vector(self._conn)

        # Enable pgvector extension
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.commit()
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._conn:
            self._conn.close()
        self._conn = None
        self._is_connected = False

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Create PostgreSQL table with pgvector index."""
        self.validate_vectors(vectors)
        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        prefix = self.config.get("collection", {}).get("name_prefix", "benchmark")
        self._table_name = f"{prefix}_{int(time.time())}"

        start_time = time.perf_counter()

        with self._conn.cursor() as cur:
            # 1. Enable extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # 2. Create Table (UNLOGGED)
            cur.execute(
                f"CREATE UNLOGGED TABLE IF NOT EXISTS {self._table_name} (id bigserial PRIMARY KEY, vector vector({dimensions}));")

            # === BATCH INSERT FIX ===
            print(f"🚀 Pgvector: Inserting {len(vectors)} vectors in batches...")
            batch_size = 10000
            insert_ids = ids if ids is not None else list(range(len(vectors)))

            for i in range(0, len(vectors), batch_size):
                end = min(i + batch_size, len(vectors))
                # Convert only this chunk to list to save RAM
                chunk_data = list(zip(insert_ids[i:end], vectors[i:end].tolist()))
                execute_values(cur, f"INSERT INTO {self._table_name} (id, vector) VALUES %s", chunk_data)
                print(f"   Inserted batch {end}/{len(vectors)}", end="\r")
            # ========================

            print(f"\n✅ Pgvector: Insertion complete.")
            # =========================================================

            # 3. Create Index
            metric_op = {
                DistanceMetric.L2: "vector_l2_ops",
                DistanceMetric.COSINE: "vector_cosine_ops",
                DistanceMetric.IP: "vector_ip_ops"
            }.get(distance_metric, "vector_l2_ops")

            idx_method = "hnsw" if index_config.type == IndexType.HNSW else "ivfflat"

            params = index_config.params
            idx_options = ""

            if idx_method == "hnsw":
                m = params.get("m", 16)
                ef = params.get("ef_construct", 64)

                # Auto-tune for GIST1M
                if dimensions > 700 and m < 32:
                    print(f"⚡ High dimensionality ({dimensions}d) detected. Boosting HNSW parameters...")
                    m = 32
                    ef = 128

                idx_options = f"(m = {m}, ef_construction = {ef})"
            elif idx_method == "ivfflat":
                lists = params.get("nlist", 100)
                idx_options = f"(lists = {lists})"

            print(f"🔨 Pgvector: Building {idx_method} index...")
            cur.execute(
                f"CREATE INDEX ON {self._table_name} USING {idx_method} (vector {metric_op}) WITH {idx_options}"
            )

            self._conn.commit()

        self._num_vectors = n_vectors
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        """Drop the table."""
        if self._table_name and self._conn:
            with self._conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self._table_name}")
            self._conn.commit()
        self._num_vectors = 0

    def save_index(self, path: str) -> None:
        """PostgreSQL manages persistence automatically."""
        pass

    def load_index(self, path: str) -> float:
        """Load existing table."""
        start_time = time.perf_counter()
        self._table_name = path
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")
            self._num_vectors = cur.fetchone()[0]
        return time.perf_counter() - start_time

    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:
        """Search for nearest neighbors."""
        if not self._table_name:
            raise RuntimeError("Table not created")

        self.validate_vectors(queries)
        params = search_params or {}

        # Set search parameters
        with self._conn.cursor() as cur:
            if "probes" in params:
                cur.execute(f"SET ivfflat.probes = {params['probes']}")
            if "ef_search" in params:
                cur.execute(f"SET hnsw.ef_search = {params['ef_search']}")

        # Determine distance operator
        op_map = {
            DistanceMetric.L2: "<->",
            DistanceMetric.COSINE: "<=>",
            DistanceMetric.IP: "<#>",
        }
        operator = op_map.get(self._distance_metric, "<->")

        # Build filter clause
        where_clause = self._build_filter(filters) if filters else ""

        all_indices = []
        all_distances = []
        latencies = []

        with self._conn.cursor() as cur:
            for query in queries:
                start_time = time.perf_counter()

                vector_str = "[" + ",".join(str(x) for x in query.tolist()) + "]"

                sql = f"""
                    SELECT id, vector {operator} %s::vector as distance
                    FROM {self._table_name}
                    {where_clause}
                    ORDER BY vector {operator} %s::vector
                    LIMIT {k}
                """

                cur.execute(sql, (vector_str, vector_str))
                results = cur.fetchall()

                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)

                indices = [r[0] for r in results]
                distances = [r[1] for r in results]

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

    def _build_filter(self, filters: List[FilterCondition]) -> str:
        """Build SQL WHERE clause."""
        if not filters:
            return ""

        conditions = []
        for f in filters:
            if f.operator == "eq":
                if isinstance(f.value, str):
                    conditions.append(f"{f.field} = '{f.value}'")
                else:
                    conditions.append(f"{f.field} = {f.value}")
            elif f.operator == "gt":
                conditions.append(f"{f.field} > {f.value}")
            elif f.operator == "gte":
                conditions.append(f"{f.field} >= {f.value}")
            elif f.operator == "lt":
                conditions.append(f"{f.field} < {f.value}")
            elif f.operator == "lte":
                conditions.append(f"{f.field} <= {f.value}")

        return "WHERE " + " AND ".join(conditions) if conditions else ""

    def insert(self, vectors: NDArray[np.float32], metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[int]] = None) -> float:
        if ids is None:
            ids = list(range(self._num_vectors, self._num_vectors + len(vectors)))

        start_time = time.perf_counter()
        with self._conn.cursor() as cur:
            data = []
            for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
                meta = metadata[i] if metadata else {}
                data.append((vec_id, vector.tolist(), meta.get("category"), meta.get("price"), meta.get("timestamp"), meta.get("active")))

            execute_values(cur, f"INSERT INTO {self._table_name} (id, vector, category, price, timestamp, active) VALUES %s",
                         data, template="(%s, %s::vector, %s, %s, %s, %s)")
        self._conn.commit()
        self._num_vectors += len(vectors)
        return time.perf_counter() - start_time

    def update(self, ids: List[int], vectors: NDArray[np.float32], metadata: Optional[List[Dict[str, Any]]] = None) -> float:
        start_time = time.perf_counter()
        with self._conn.cursor() as cur:
            for i, (vec_id, vector) in enumerate(zip(ids, vectors)):
                cur.execute(f"UPDATE {self._table_name} SET vector = %s::vector WHERE id = %s", (vector.tolist(), vec_id))
        self._conn.commit()
        return time.perf_counter() - start_time

    def delete(self, ids: List[int]) -> float:
        start_time = time.perf_counter()
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table_name} WHERE id = ANY(%s)", (ids,))
        self._conn.commit()
        self._num_vectors -= len(ids)
        return time.perf_counter() - start_time

    def get_index_stats(self) -> Dict[str, Any]:
        if not self._table_name:
            return {}
        
        stats = {
            "dimensions": self._dimensions,
            "index_type": self._index_config.type if self._index_config else None
        }
        
        with self._conn.cursor() as cur:
            # Get count
            cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")
            stats["num_vectors"] = cur.fetchone()[0]
            
            # Get size in bytes
            cur.execute(f"SELECT pg_total_relation_size('{self._table_name}')")
            stats["index_size_bytes"] = cur.fetchone()[0]
            
        return stats

    def set_search_params(self, params: Dict[str, Any]) -> None:
        self._search_params.update(params)

    def get_search_params(self) -> Dict[str, Any]:
        return self._search_params.copy()

    # === NEW: Single-Item Wrappers for Benchmarking ===

    def insert_one(self, id: str, vector: np.ndarray):
        """Inserts a single row via SQL."""
        try:
            int_id = int(id) if str(id).isdigit() else 999999

            with self._conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._table_name} (id, vector) VALUES (%s, %s)",
                    (int_id, vector.tolist())
                )
            self._conn.commit()
        except Exception as e:
            # Catch primary key collision or other SQL errors
            print(f"Pgvector insert_one failed: {e}")
            self._conn.rollback()

    def delete_one(self, id: str):
        """Deletes a single row via SQL."""
        try:
            int_id = int(id) if str(id).isdigit() else 999999

            with self._conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table_name} WHERE id = %s",
                    (int_id,)
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()

    def update_one(self, id: str, vector: np.ndarray):
        """Updates a single row via SQL."""
        try:
            int_id = int(id) if str(id).isdigit() else 999999

            with self._conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self._table_name} SET vector = %s WHERE id = %s",
                    (vector.tolist(), int_id)
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
