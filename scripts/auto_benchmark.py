import subprocess
import time
import argparse
import sys
import os

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def get_root_path(filename):
    return os.path.join(PROJECT_ROOT, filename)


DOCKER_MAP = {
    "milvus": get_root_path("milvus-docker-compose.yml"),
    "qdrant": get_root_path("qdrant-docker-compose.yml"),
    "weaviate": get_root_path("weaviate-docker-compose.yml"),
    "pgvector": get_root_path("pgvector-docker-compose.yml"),
    "chroma": get_root_path("chroma-docker-compose.yml"),
    "lancedb": None,
    "faiss": None
}

WARMUP_TIME = {
    "milvus": 60, "qdrant": 30, "weaviate": 30, "pgvector": 30,
    "chroma": 30, "lancedb": 0, "faiss": 0
}


def run_command_simple(cmd):
    """Executes a shell command silently (for docker operations)."""
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=PROJECT_ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass


def run_benchmark_command(cmd):
    """Executes the benchmark script and streams output to console."""
    print(f"Executing: {cmd}")
    # check=True will raise CalledProcessError if the script crashes (exit code != 0)
    subprocess.run(cmd, shell=True, check=True, cwd=PROJECT_ROOT)


def manage_container(db_name, action):
    compose_file = DOCKER_MAP.get(db_name)
    if not compose_file: return

    if action == "up":
        print(f"🚀 Initializing {db_name}...")
        # Standard Docker Reset (Internal Volumes only)
        run_command_simple(f"docker compose -f \"{compose_file}\" down -v")
        run_command_simple(f"docker compose -f \"{compose_file}\" up -d")
        time.sleep(WARMUP_TIME.get(db_name, 10))

    elif action == "down":
        print(f"🛑 Stopping {db_name}...")
        run_command_simple(f"docker compose -f \"{compose_file}\" down -v")

    elif action == "restart":
        print(f"♻️  RESET: Restarting {db_name}...")
        run_command_simple(f"docker compose -f \"{compose_file}\" down -v")
        time.sleep(5)
        run_command_simple(f"docker compose -f \"{compose_file}\" up -d")
        print(f"⏳ Waiting {WARMUP_TIME.get(db_name, 10)}s for recovery...")
        time.sleep(WARMUP_TIME.get(db_name, 10))


def force_cleanup_all():
    print("\n🛑 INTERRUPTED! Shutting down all containers...")
    for db, file in DOCKER_MAP.items():
        if file and os.path.exists(file):
            run_command_simple(f"docker compose -f \"{file}\" down")
    print("✅ Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(description="Automated VectorDB Benchmark Runner")
    parser.add_argument("--database", nargs='+', required=True, help="List of databases (or 'all')")
    parser.add_argument("--dataset", nargs='+', required=True, help="List of datasets")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--export", nargs='+', default=["json"], help="Export formats")
    args = parser.parse_args()

    # Expand 'all' keyword
    target_databases = []
    if 'all' in args.database:
        target_databases = list(DOCKER_MAP.keys())
    else:
        target_databases = args.database

    # Initial Cleanup
    print("🧹 Cleaning up workspace...")
    for db, file in DOCKER_MAP.items():
        if file and os.path.exists(file): run_command_simple(f"docker compose -f \"{file}\" down")

    try:
        # === OUTER LOOP: Databases ===
        for db in target_databases:
            print(f"\n{'=' * 60}\n PREPARING BENCHMARK FOR: {db.upper()}\n{'=' * 60}")
            manage_container(db, "up")

            runner_script = get_root_path("scripts/run_benchmark.py")
            export_str = " ".join(args.export)

            # === INNER LOOP: Datasets ===
            for i, dataset in enumerate(args.dataset):

                print(f"\n▶️  Running: {db} on {dataset}...")
                cmd = (
                    f"\"{sys.executable}\" \"{runner_script}\" --database {db} --dataset {dataset} --runs {args.runs} --export {export_str}")

                try:
                    run_benchmark_command(cmd)
                    print("✅ Benchmark Run Successful.")
                except subprocess.CalledProcessError:
                    print(f"❌ Run Failed (Exit Code Error). Moving to next dataset.")

                # --- SWITCHING LOGIC ---
                # We restart the container between datasets to ensure a clean state
                is_last_dataset = (i == len(args.dataset) - 1)

                if not is_last_dataset:
                    if DOCKER_MAP.get(db):
                        manage_container(db, "restart")
                    else:
                        print("❄️  Cooling down...")
                        time.sleep(10)

            manage_container(db, "down")

    except KeyboardInterrupt:
        force_cleanup_all()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        force_cleanup_all()
        sys.exit(1)


if __name__ == "__main__":
    main()