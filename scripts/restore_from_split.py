"""
restore_from_split.py
Restore the full dataset from split parts (part1~part4).

Input: data/interim/split_archive/step4_full_data.part*
Output: data/interim/step4_full_data.parquet
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


def restore():
    archive_dir = os.path.join(Config.DATA_INTERIM_PATH, "split_archive")
    output_path = os.path.join(Config.DATA_INTERIM_PATH, "step4_full_data.parquet")
    
    # Check archive exists
    if not os.path.exists(archive_dir):
        print(f"❌ Archive directory not found: {archive_dir}")
        print("   Please ensure the split parts are in this directory.")
        return
    
    # Check for parts
    parts = [f"step4_full_data.part{i}" for i in range(1, 5)]
    missing = []
    for part in parts:
        if not os.path.exists(os.path.join(archive_dir, part)):
            missing.append(part)
    
    if missing:
        print(f"❌ Missing parts: {missing}")
        print("   Please download all 4 parts before restoring.")
        return
    
    print("Merging split parts...")
    with open(output_path, "wb") as outfile:
        for part in parts:
            part_path = os.path.join(archive_dir, part)
            with open(part_path, "rb") as infile:
                outfile.write(infile.read())
            print(f"  ✅ Merged {part}")
    
    # Verify size
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ Restored: {output_path} ({size_mb:.1f} MB)")
    
    if size_mb < 100:
        print("⚠️ Warning: Restored file is very small (<100 MB).")
        print("   Please check that all parts were downloaded completely.")


if __name__ == "__main__":
    restore()
