"""
Weaviate vector database adapter.
Compatible with Weaviate Python Client v4.
"""

import time
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.core.base import VectorDBInterface
from src.core.types import DatabaseInfo, DatabaseType, DistanceMetric, IndexConfig, IndexType
from src.databases.factory import register_database

# === WEAVIATE v4 IMPORTS ===
try:
    import weaviate
    from weaviate.classes import config as wvc
    from weaviate.classes import query as wvq
    from weaviate.classes import aggregate as wva
    from weaviate.classes.init import AdditionalConfig, Timeout
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

# Suppress Weaviate DeprecationWarnings for cleaner logs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="weaviate")

@register_database("weaviate")
class WeaviateAdapter(VectorDBInterface):
    """Weaviate vector database adapter."""

    def __init__(self, config: Dict[str, Any]):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate not installed. Please run: pip install weaviate-client")
        super().__init__(config)
        self._client = None
        self._collection = None
        self._collection_name: str = ""
        # Reverted named vector to ensure stability with current client version

    @property
    def name(self) -> str: return "weaviate"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="weaviate",
            display_name="Weaviate",
            version="1.24.1",
            type=DatabaseType.DATABASE,
            language="Go",
            license="BSD-3-Clause",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[IndexType.HNSW],
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_gpu=False,
            is_distributed=True,
        )

    def connect(self) -> None:
        c = self.config.get("connection", {})
        host = c.get("host", "localhost")
        port = c.get("http_port", 8080)
        grpc_port = c.get("grpc_port", 50051)

        # === FIX: Use AdditionalConfig for Timeouts in v4 ===
        # We set long timeouts (1 hour) to prevent benchmark crashes during heavy loads
        self._client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=120, query=3600, insert=3600)
            )
        )
        self._is_connected = True

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
        self._is_connected = False

    def create_index(self, vectors, index_config, distance_metric, metadata=None, ids=None) -> float:
        self.validate_vectors(vectors)
        self._dimensions = vectors.shape[1]
        self._distance_metric = distance_metric

        prefix = self.config.get("collection", {}).get("name_prefix", "Benchmark")
        self._collection_name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        start_time = time.perf_counter()

        # Map metrics to Weaviate v4 enums
        dist_map = {
            DistanceMetric.L2: wvc.VectorDistances.L2_SQUARED,
            DistanceMetric.COSINE: wvc.VectorDistances.COSINE,
            DistanceMetric.IP: wvc.VectorDistances.DOT,
        }

        # Create Collection (Reverted to deprecated but working arguments)
        self._client.collections.create(
            name=self._collection_name,
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            vector_index_config=wvc.Configure.VectorIndex.hnsw(
                distance_metric=dist_map.get(distance_metric, wvc.VectorDistances.L2_SQUARED),
                ef_construction=index_config.params.get("ef_construct", 128),
                max_connections=index_config.params.get("m", 16),
                cleanup_interval_seconds=300
            ),
            # Explicitly define 'vec_id' property
            properties=[wvc.Property(name="vec_id", data_type=wvc.DataType.INT)]
        )

        self._collection = self._client.collections.get(self._collection_name)

        print(f"🚀 Weaviate: Inserting {len(vectors)} vectors (Fixed Batch Mode)...")

        # === FIXED BATCHING STRATEGY ===
        # We use a conservative fixed batch size to avoid overwhelming the local Docker network
        # and causing DNS/gRPC timeouts.
        batch_size = 1000

        with self._collection.batch.fixed_size(batch_size=batch_size, concurrent_requests=2) as batch:
            for i, vector in enumerate(vectors):
                vec_id_int = ids[i] if ids else i
                # Create a deterministic UUID from the integer ID
                uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vec_id_int)))

                batch.add_object(
                    properties={"vec_id": vec_id_int},
                    vector=vector.tolist(), # Reverted to simple vector
                    uuid=uid
                )

                if i > 0 and i % 10000 == 0:
                    print(f"   Processed {i} vectors...", end="\r")

        # Check for failures
        if len(self._collection.batch.failed_objects) > 0:
            print(f"⚠️ Warning: {len(self._collection.batch.failed_objects)} objects failed to upload.")
            # Optional: Print first error to debug
            print(f"   First error: {self._collection.batch.failed_objects[0]}")

        print("\n⏳ Weaviate: Waiting for indexing (Shards READY)...")
        # Wait for shards to report "READY" status
        for _ in range(30):
            try:
                shards = self._collection.config.get_shards()
                if all(s.status == "READY" for s in shards):
                    print("✅ All shards READY.")
                    break
            except Exception:
                pass
            time.sleep(5)
        
        # Stabilization wait (heuristic)
        time.sleep(60)

        self._num_vectors = len(vectors)
        return time.perf_counter() - start_time

    def delete_index(self) -> None:
        if self._collection_name:
            try:
                self._client.collections.delete(self._collection_name)
            except Exception as e:
                print(f"⚠️ Delete index failed: {e}")

    def save_index(self, path: str) -> None: pass
    def load_index(self, path: str) -> float: return 0.0

    def search(self, queries, k, search_params=None, filters=None):
        if not self._collection: raise RuntimeError("No collection connected")

        latencies = []
        all_indices = []
        all_distances = []

        for query in queries:
            start = time.perf_counter()

            try:
                # Retry logic for transient gRPC errors
                res = None
                for attempt in range(3):
                    try:
                        res = self._collection.query.near_vector(
                            near_vector=query,
                            limit=k,
                            # target_vector removed
                            return_metadata=wvq.MetadataQuery(distance=True),
                            return_properties=["vec_id"]
                        )
                        break
                    except Exception as e:
                        if attempt == 2:
                            # If final attempt fails, log it
                            print(f"   Search attempt failed: {e}")
                        time.sleep(0.1)

                if res:
                    latencies.append((time.perf_counter() - start) * 1000)

                    indices = []
                    dists = []
                    for obj in res.objects:
                        indices.append(obj.properties["vec_id"])
                        dists.append(obj.metadata.distance)

                    all_indices.append(indices)
                    all_distances.append(dists)
                else:
                    latencies.append(0.0)
                    all_indices.append([])
                    all_distances.append([])

            except Exception as e:
                print(f"⚠️ Search error: {e}")
                latencies.append(0.0)
                all_indices.append([])
                all_distances.append([])

        return (np.array(all_indices), np.array(all_distances), latencies)

    def insert(self, vectors, metadata=None, ids=None) -> float: return 0.0
    def update(self, ids, vectors, metadata=None) -> float: return 0.0
    def delete(self, ids) -> float: return 0.0

    def get_index_stats(self) -> Dict[str, Any]:
        if not self._collection: return {}
        
        stats = {"dimensions": self._dimensions}
        try:
            # Use aggregation to get total count
            agg = self._collection.aggregate.over_all(total_count=True)
            stats["num_vectors"] = agg.total_count
            # Estimate size
            stats["index_size_bytes"] = agg.total_count * self._dimensions * 4 # float32
        except Exception:
            stats["num_vectors"] = 0
            stats["index_size_bytes"] = 0
            
        return stats

    def set_search_params(self, params): pass
    def get_search_params(self): return {}

    # === NEW: Single-Item Wrappers for Benchmarking ===

    def insert_one(self, id: str, vector: np.ndarray):
        """Inserts a single object."""
        # Replicate the UUID generation logic from create_index
        try:
            vec_id_int = int(id) if str(id).isdigit() else 0
            # FIX: Use deterministic UUID to match delete_one
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vec_id_int)))

            self._collection.data.insert(
                properties={"vec_id": vec_id_int},
                vector=vector.tolist(), # Reverted to simple vector
                uuid=uid
            )
        except Exception as e:
            print(f"Weaviate insert_one failed: {e}")

    def delete_one(self, id: str):
        """Deletes a single object by UUID."""
        try:
            vec_id_int = int(id) if str(id).isdigit() else 0
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vec_id_int)))

            self._collection.data.delete_by_id(uid)
        except Exception:
            pass

    def update_one(self, id: str, vector: np.ndarray):
        """Updates a single object."""
        try:
            vec_id_int = int(id) if str(id).isdigit() else 0
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vec_id_int)))

            # FIX: Use replace instead of insert for updates
            self._collection.data.replace(
                properties={"vec_id": vec_id_int},
                vector=vector.tolist(),
                uuid=uid
            )
        except Exception as e:
            print(f"Weaviate update_one failed: {e}")