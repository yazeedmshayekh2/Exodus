import numpy as np
import os

def check_embeddings_shape():
    """Check the dimensions of the embeddings in the cache file."""
    try:
        cache_file = os.path.join(os.getcwd(), 'embeddings_cache.npz')
        if not os.path.exists(cache_file):
            print(f"Cache file not found: {cache_file}")
            return
            
        print(f"Loading cache file: {cache_file}")
        cache_data = np.load(cache_file, allow_pickle=True)
        
        # Check keys
        print(f"Cache data keys: {list(cache_data.keys())}")
        
        # Check vectors shape
        if 'vectors' in cache_data:
            vectors = cache_data['vectors']
            print(f"Vectors shape: {vectors.shape}")
            print(f"Vector dimension: {vectors.shape[1]}")
            
            # Check a sample vector
            if len(vectors) > 0:
                print(f"Sample vector (first 5 elements): {vectors[0][:5]}")
        else:
            print("No 'vectors' key found in cache file")
            
        # Check number of entries
        if 'ids' in cache_data:
            ids = cache_data['ids']
            print(f"Number of entries: {len(ids)}")
        
    except Exception as e:
        print(f"Error checking embeddings: {e}")

if __name__ == "__main__":
    check_embeddings_shape() 