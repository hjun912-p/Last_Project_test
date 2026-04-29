import videoseal
import torch
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    print("Attempting to load 'videoseal' model...")
    # Try different possible model names if 'videoseal' fails
    try:
        model = videoseal.load("videoseal")
        print("Successfully loaded 'videoseal'")
    except Exception as e1:
        print(f"Failed 'videoseal': {e1}")
        print("Attempting to load 'pixelseal' model...")
        model = videoseal.load("pixelseal")
        print("Successfully loaded 'pixelseal'")
        
except Exception as e:
    print(f"Critical Error: {e}")
