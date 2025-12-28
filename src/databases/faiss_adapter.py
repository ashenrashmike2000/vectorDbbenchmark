"""
FAISS (Facebook AI Similarity Search) adapter.

FAISS is a library for efficient similarity search and clustering of dense vectors.
It is often used as the baseline for ANN benchmark comparisons.

Documentation: https://faiss.ai/
"""

import os
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

# Import FAISS with error handling
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@register_database("faiss")
class FAISSAdapter(VectorDBInterface):
    """
    FAISS vector database adapter.

    Supports multiple index types including:
        - Flat (exact search)
        - IVF (Inverted File)
        - IVFPQ (IVF with Product Quantization)
        - HNSW (Hierarchical Navigable Small World)
        - IVF_SQ (IVF with Scalar Quantization)

    Attributes:
        index: The FAISS index object
        index_type: Type of index being used
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FAISS adapter.

        Args:
            config: FAISS-specific configuration
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            )

        super().__init__(config)
        self._index: Optional[faiss.Index] = None
        self._index_type: Optional[str] = None
        self._trained: bool = False
        self._search_params: Dict[str, Any] = {}
        self._use_gpu: bool = config.get("gpu", {}).get("enabled", False)
        self._gpu_device: int = config.get("gpu", {}).get("device_id", 0)

    @property
    def name(self) -> str:
        return "faiss"

    @property
    def info(self) -> DatabaseInfo:
        return DatabaseInfo(
            name="faiss",
            display_name="FAISS",
            version=faiss.__version__ if hasattr(faiss, "__version__") else "unknown",
            type=DatabaseType.LIBRARY,
            language="C++/Python",
            license="MIT",
            supported_metrics=[DistanceMetric.L2, DistanceMetric.COSINE, DistanceMetric.IP],
            supported_index_types=[
                IndexType.FLAT,
                IndexType.IVF,
                IndexType.IVFPQ,
                IndexType.HNSW,
                IndexType.IVFSQ,
            ],
            supports_filtering=False,  # Native FAISS doesn't support filtering
            supports_hybrid_search=False,
            supports_gpu=True,
            is_distributed=False,
        )

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        """FAISS is an in-process library, no connection needed."""
        self._is_connected = True

    def disconnect(self) -> None:
        """Clean up FAISS resources."""
        if self._index is not None:
            del self._index
            self._index = None
        self._is_connected = False
        self._trained = False

    # =========================================================================
    # Index Management
    # =========================================================================

    def create_index(
        self,
        vectors: NDArray[np.float32],
        index_config: IndexConfig,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """
        Create a FAISS index from vectors.

        Args:
            vectors: Vector data of shape (n, d)
            index_config: Index configuration
            distance_metric: Distance metric to use
            metadata: Not supported by FAISS (ignored)
            ids: Not directly supported (FAISS uses sequential IDs)

        Returns:
            Index build time in seconds
        """
        self.validate_vectors(vectors)

        n_vectors, dimensions = vectors.shape
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._index_config = index_config

        start_time = time.perf_counter()

        # Determine metric type
        metric_type = self._get_faiss_metric(distance_metric)

        # Normalize vectors for cosine similarity
        if distance_metric == DistanceMetric.COSINE:
            vectors = self.normalize_vectors(vectors.copy())

        # Build the index based on configuration
        index_type = index_config.type.upper()
        params = index_config.params

        if index_type == "FLAT":
            self._index = self._create_flat_index(dimensions, metric_type)
        elif index_type in ("IVF", "IVF_FLAT"):
            self._index = self._create_ivf_index(dimensions, metric_type, params, vectors)
        elif index_type in ("IVFPQ", "IVF_PQ"):
            self._index = self._create_ivfpq_index(dimensions, metric_type, params, vectors)
        elif index_type in ("IVFSQ", "IVF_SQ"):
            self._index = self._create_ivfsq_index(dimensions, metric_type, params, vectors)
        elif index_type == "HNSW":
            self._index = self._create_hnsw_index(dimensions, metric_type, params)
        elif index_type == "OPQ":
            self._index = self._create_opq_index(dimensions, metric_type, params, vectors)
        else:
            # Try to build from string description
            self._index = faiss.index_factory(dimensions, index_config.name, metric_type)
            if not self._index.is_trained:
                train_vectors = self._get_training_vectors(vectors)
                self._index.train(train_vectors)
                self._trained = True

        # Add vectors to index
        self._index.add(vectors)
        self._num_vectors = n_vectors
        self._index_type = index_type

        # Move to GPU if configured
        if self._use_gpu:
            self._index = self._move_to_gpu(self._index)

        build_time = time.perf_counter() - start_time
        return build_time

    def _create_flat_index(
        self, dimensions: int, metric_type: int
    ) -> faiss.Index:
        """Create a flat (exact search) index."""
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            return faiss.IndexFlatIP(dimensions)
        return faiss.IndexFlatL2(dimensions)

    def _create_ivf_index(
        self,
        dimensions: int,
        metric_type: int,
        params: Dict[str, Any],
        vectors: NDArray[np.float32],
    ) -> faiss.Index:
        """Create an IVF index."""
        nlist = params.get("nlist", 1024)

        # Create quantizer
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(dimensions)
        else:
            quantizer = faiss.IndexFlatL2(dimensions)

        # Create IVF index
        index = faiss.IndexIVFFlat(quantizer, dimensions, nlist, metric_type)

        # Train the index
        train_vectors = self._get_training_vectors(vectors)
        index.train(train_vectors)
        self._trained = True

        return index

    def _create_ivfpq_index(
        self,
        dimensions: int,
        metric_type: int,
        params: Dict[str, Any],
        vectors: NDArray[np.float32],
    ) -> faiss.Index:
        """Create an IVF+PQ index."""
        nlist = params.get("nlist", 1024)
        m = params.get("m", 16)  # Number of sub-quantizers
        nbits = params.get("nbits", 8)

        # Create quantizer
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(dimensions)
        else:
            quantizer = faiss.IndexFlatL2(dimensions)

        # Create IVFPQ index
        index = faiss.IndexIVFPQ(quantizer, dimensions, nlist, m, nbits, metric_type)

        # Train the index
        train_vectors = self._get_training_vectors(vectors)
        index.train(train_vectors)
        self._trained = True

        return index

    def _create_ivfsq_index(
        self,
        dimensions: int,
        metric_type: int,
        params: Dict[str, Any],
        vectors: NDArray[np.float32],
    ) -> faiss.Index:
        """Create an IVF+SQ (Scalar Quantizer) index."""
        nlist = params.get("nlist", 1024)
        qtype = params.get("qtype", "QT_8bit")

        # Map quantizer type string to FAISS constant
        qtype_map = {
            "QT_8bit": faiss.ScalarQuantizer.QT_8bit,
            "QT_4bit": faiss.ScalarQuantizer.QT_4bit,
            "QT_8bit_uniform": faiss.ScalarQuantizer.QT_8bit_uniform,
            "QT_4bit_uniform": faiss.ScalarQuantizer.QT_4bit_uniform,
            "QT_fp16": faiss.ScalarQuantizer.QT_fp16,
        }
        sq_type = qtype_map.get(qtype, faiss.ScalarQuantizer.QT_8bit)

        # Create quantizer
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(dimensions)
        else:
            quantizer = faiss.IndexFlatL2(dimensions)

        # Create IVFSQ index
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dimensions, nlist, sq_type, metric_type
        )

        # Train the index
        train_vectors = self._get_training_vectors(vectors)
        index.train(train_vectors)
        self._trained = True

        return index

    def _create_hnsw_index(
        self, dimensions: int, metric_type: int, params: Dict[str, Any]
    ) -> faiss.Index:
        """Create an HNSW index."""
        M = params.get("M", 32)
        ef_construction = params.get("efConstruction", 200)

        # Create HNSW index
        index = faiss.IndexHNSWFlat(dimensions, M, metric_type)
        index.hnsw.efConstruction = ef_construction

        # HNSW doesn't need training
        self._trained = True

        return index

    def _create_opq_index(
        self,
        dimensions: int,
        metric_type: int,
        params: Dict[str, Any],
        vectors: NDArray[np.float32],
    ) -> faiss.Index:
        """Create an OPQ (Optimized Product Quantization) index."""
        M_OPQ = params.get("M_OPQ", 32)
        nlist = params.get("nlist", 4096)
        m = params.get("m", 32)
        nbits = params.get("nbits", 8)

        # Build index string
        index_str = f"OPQ{M_OPQ},IVF{nlist},PQ{m}x{nbits}"

        index = faiss.index_factory(dimensions, index_str, metric_type)

        # Train the index
        train_vectors = self._get_training_vectors(vectors)
        index.train(train_vectors)
        self._trained = True

        return index

    def _get_training_vectors(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get vectors for training (subset if dataset is large)."""
        training_config = self.config.get("training", {})
        train_ratio = training_config.get("train_size_ratio", 0.1)
        min_train = training_config.get("min_train_size", 10000)
        max_train = training_config.get("max_train_size", 1000000)

        n_vectors = len(vectors)
        train_size = int(n_vectors * train_ratio)
        train_size = max(min_train, min(train_size, max_train, n_vectors))

        if train_size < n_vectors:
            indices = np.random.choice(n_vectors, train_size, replace=False)
            return vectors[indices]
        return vectors

    def _get_faiss_metric(self, metric: DistanceMetric) -> int:
        """Convert DistanceMetric to FAISS metric type."""
        if metric in (DistanceMetric.IP, DistanceMetric.COSINE):
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2

    def _move_to_gpu(self, index: faiss.Index) -> faiss.Index:
        """Move index to GPU if available."""
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, self._gpu_device, index)
        except Exception:
            # GPU not available, return CPU index
            return index

    def delete_index(self) -> None:
        """Delete the current index."""
        if self._index is not None:
            del self._index
            self._index = None
        self._num_vectors = 0
        self._trained = False

    def save_index(self, path: str) -> None:
        """Save index to disk."""
        if self._index is None:
            raise RuntimeError("No index to save")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Move to CPU before saving if on GPU
        index_to_save = self._index
        if self._use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self._index)

        faiss.write_index(index_to_save, path)

    def load_index(self, path: str) -> float:
        """Load index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        start_time = time.perf_counter()

        self._index = faiss.read_index(path)
        self._num_vectors = self._index.ntotal
        self._dimensions = self._index.d
        self._trained = True

        if self._use_gpu:
            self._index = self._move_to_gpu(self._index)

        return time.perf_counter() - start_time

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], List[float]]:
        """
        Search for k nearest neighbors.

        Args:
            queries: Query vectors of shape (n_queries, d)
            k: Number of neighbors to return
            search_params: Search parameters (nprobe, efSearch, etc.)
            filters: Not supported by FAISS

        Returns:
            Tuple of (indices, distances, latencies)
        """
        if self._index is None:
            raise RuntimeError("Index not created. Call create_index first.")

        if filters:
            raise NotImplementedError("FAISS does not support metadata filtering")

        self.validate_vectors(queries)

        # Normalize queries for cosine similarity
        if self._distance_metric == DistanceMetric.COSINE:
            queries = self.normalize_vectors(queries.copy())

        # Apply search parameters
        if search_params:
            self._apply_search_params(search_params)

        # Perform search with per-query timing
        n_queries = len(queries)
        latencies = []
        all_distances = []
        all_indices = []

        for i in range(n_queries):
            query = queries[i : i + 1]
            start_time = time.perf_counter()
            distances, indices = self._index.search(query, k)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            all_distances.append(distances[0])
            all_indices.append(indices[0])

        return (
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32),
            latencies,
        )

    def _apply_search_params(self, params: Dict[str, Any]) -> None:
        """Apply search parameters to the index."""
        if self._index is None:
            return

        # nprobe for IVF indices
        if "nprobe" in params and hasattr(self._index, "nprobe"):
            self._index.nprobe = params["nprobe"]

        # efSearch for HNSW indices
        if "efSearch" in params or "ef" in params:
            ef = params.get("efSearch", params.get("ef"))
            if hasattr(self._index, "hnsw"):
                self._index.hnsw.efSearch = ef

        self._search_params.update(params)

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def insert(
        self,
        vectors: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
    ) -> float:
        """Insert additional vectors into the index."""
        if self._index is None:
            raise RuntimeError("Index not created")

        self.validate_vectors(vectors)

        if self._distance_metric == DistanceMetric.COSINE:
            vectors = self.normalize_vectors(vectors.copy())

        start_time = time.perf_counter()
        self._index.add(vectors)
        self._num_vectors += len(vectors)

        return time.perf_counter() - start_time

    def update(
        self,
        ids: List[int],
        vectors: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Update vectors by ID.

        Note: FAISS doesn't support true updates, so this removes and re-adds.
        For IndexFlatL2/IP, this is not directly supported.
        """
        raise NotImplementedError(
            "FAISS does not efficiently support updates. "
            "Rebuild the index for modified vectors."
        )

    def delete(self, ids: List[int]) -> float:
        """
        Delete vectors by ID.

        Note: Only supported for IndexIDMap wrapped indices.
        """
        if not hasattr(self._index, "remove_ids"):
            raise NotImplementedError(
                "Delete is only supported for ID-mapped FAISS indices"
            )

        start_time = time.perf_counter()
        id_selector = faiss.IDSelectorArray(len(ids), faiss.swig_ptr(np.array(ids, dtype=np.int64)))
        self._index.remove_ids(id_selector)
        self._num_vectors -= len(ids)

        return time.perf_counter() - start_time

    # =========================================================================
    # Statistics and Configuration
    # =========================================================================

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self._index is None:
            return {}

        stats = {
            "num_vectors": self._index.ntotal,
            "dimensions": self._index.d,
            "index_type": self._index_type,
            "distance_metric": self._distance_metric.value if self._distance_metric else None,
            "is_trained": self._index.is_trained,
        }

        # Try to estimate memory usage
        try:
            # This is an approximation
            base_memory = self._index.ntotal * self._index.d * 4  # float32
            stats["index_size_bytes"] = base_memory
            stats["bytes_per_vector"] = base_memory / max(self._index.ntotal, 1)
        except Exception:
            stats["index_size_bytes"] = 0

        return stats

    def set_search_params(self, params: Dict[str, Any]) -> None:
        """Set search parameters."""
        self._apply_search_params(params)

    def get_search_params(self) -> Dict[str, Any]:
        """Get current search parameters."""
        params = self._search_params.copy()

        if self._index is not None:
            if hasattr(self._index, "nprobe"):
                params["nprobe"] = self._index.nprobe
            if hasattr(self._index, "hnsw"):
                params["efSearch"] = self._index.hnsw.efSearch

        return params

    # === NEW: Single-Item Wrappers for Benchmarking ===

    def insert_one(self, id: str, vector: np.ndarray):
        """Inserts a single vector (converted to batch)."""
        # Faiss expects shape (1, d)
        vec_reshaped = np.array([vector], dtype=np.float32)

        # If ID mapping is supported, we would pass IDs here,
        # but standard Faiss add() creates sequential IDs.
        # For benchmarking valid insert time, we just call add().
        self.insert(vec_reshaped)

    def delete_one(self, id: str):
        """Deletes a single vector (if supported)."""
        try:
            # Try to convert string ID to int if your Faiss index uses int IDs
            int_id = int(id) if id.isdigit() else 0
            self.delete([int_id])
        except Exception:
            # Faiss often throws errors for delete if not using IndexIDMap
            pass

    def update_one(self, id: str, vector: np.ndarray):
        """Updates a single vector."""
        self.delete_one(id)
        self.insert_one(id, vector)
