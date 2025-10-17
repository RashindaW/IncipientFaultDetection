"""
Quick dry-run test to verify training script works for 1 epoch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dyedgegat"))

# Simulate running train_dyedgegat.py with minimal settings
if __name__ == "__main__":
    # Override sys.argv to simulate command-line args
    sys.argv = [
        "train_dyedgegat.py",
        "--epochs", "1",
        "--batch-size", "8",
        "--train-stride", "200",  # Large stride for quick test
        "--val-stride", "200",
        "--learning-rate", "0.001",
    ]
    
    # Import and run the training script
    import train_dyedgegat
    
    print("\n✅ Training dry-run completed successfully!")

