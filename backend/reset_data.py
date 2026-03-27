import os
import shutil

# This script resets the DocuMind AI data directory to a clean state.
# Run this if you experience 'Index Out of Range' or synchronization errors.

backend_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(backend_dir, "data")
rag_state = os.path.join(data_dir, "rag_state.json")
vectors_file = os.path.join(data_dir, "vectors.npy")
uploads_dir = os.path.join(data_dir, "uploads")
exports_dir = os.path.join(data_dir, "exports")

print("--- DocuMind AI: Data Reset Utility ---")

def remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {os.path.basename(path)}")
        else:
            os.remove(path)
            print(f"Removed file: {os.path.basename(path)}")

remove_if_exists(rag_state)
remove_if_exists(vectors_file)
remove_if_exists(uploads_dir)
remove_if_exists(exports_dir)

# Re-create empty directories
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(exports_dir, exist_ok=True)

print("\nSUCCESS: All data reset. Your RAG engine is now clean and ready for a fresh start! 🚀")
