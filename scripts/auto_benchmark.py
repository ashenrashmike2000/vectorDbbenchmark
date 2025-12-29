import subprocess
import time
import argparse
import sys
import os
import shutil

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
    "milvus": 200, "qdrant": 30, "weaviate": 30, "pgvector": 30,
    "chroma": 30, "lancedb": 0, "faiss": 0
}

def run_command_simple(cmd):
    """Executes a shell command AND PRINTS ERRORS if it fails."""
    try:
        # Removed stderr=DEVNULL so errors show up in your terminal
        subprocess.run(cmd, shell=True, check=True, cwd=PROJECT_ROOT,
                       stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Command Failed: {cmd}")
        print(f"   Error: {e}")


def cleanup_host_volumes():
    """Forcefully deletes the local 'volumes' folder to ensure fresh state."""
    vol_path = os.path.join(PROJECT_ROOT, "volumes")
    if os.path.exists(vol_path):
        print(f"🧹 Deleting local volumes folder: {vol_path}...")
        try:
            # Python's way of doing 'sudo rm -rf'
            shutil.rmtree(vol_path)
        except PermissionError:
            # If root created the files, Python might fail. Try shell command.
            run_command_simple(f"sudo rm -rf \"{vol_path}\"")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete volumes: {e}")


def wait_for_port(port, timeout=60):
    """Wait until a port is open on localhost."""
    import socket
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(1)
    return False


def run_benchmark_command(cmd):
    """Executes the benchmark script and streams output to console."""
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=PROJECT_ROOT)


def manage_container(db_name, action):
    compose_file = DOCKER_MAP.get(db_name)
    if not compose_file: return

    # =========================================================
    # PART 1: EXECUTE DOCKER COMMANDS
    # =========================================================
    if action == "up":
        # 1. CLEANUP OLD DATA
        cleanup_host_volumes()

        # 2. START CONTAINER
        print(f"🚀 Starting {db_name}...")
        run_command_simple(f"docker-compose -f \"{compose_file}\" up -d")

    elif action == "restart":
        print(f"♻️  RESET: Restarting {db_name}...")
        run_command_simple(f"docker-compose -f \"{compose_file}\" down -v")

        # 1. CLEANUP DATA BEFORE RESTARTING
        cleanup_host_volumes()

        time.sleep(5)
        run_command_simple(f"docker-compose -f \"{compose_file}\" up -d")

    elif action == "down":
        print(f"🛑 Stopping {db_name}...")
        run_command_simple(f"docker-compose -f \"{compose_file}\" down -v")

        # 1. FINAL CLEANUP
        cleanup_host_volumes()
        return

    # =========================================================
    # PART 2: WAIT FOR READINESS
    # =========================================================
    if db_name == "milvus":
        print("⏳ Polling Milvus port 19530...")
        if wait_for_port(19530, timeout=500):
            print("✅ Milvus is ready!")
            time.sleep(10)
        else:
            print("❌ Milvus failed to start (Port 19530 closed)!")
    else:
        wait_time = WARMUP_TIME.get(db_name, 30)
        print(f"⏳ Waiting {wait_time}s for {db_name}...")
        time.sleep(wait_time)


def force_cleanup_all():
    print("\n🛑 INTERRUPTED! Shutting down all containers...")
    for db, file in DOCKER_MAP.items():
        if file and os.path.exists(file):
            run_command_simple(f"docker-compose -f \"{file}\" down -v")
    cleanup_host_volumes()
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
        if file and os.path.exists(file): run_command_simple(f"docker-compose -f \"{file}\" down -v")
    cleanup_host_volumes()

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
