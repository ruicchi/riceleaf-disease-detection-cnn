"""
Check dataset for bad/corrupted/non-image files.
"""

import os
from PIL import Image

DATA_DIR = "dataset"


def check_dataset():
    print("🔍 Checking dataset for problems...")
    print("=" * 50)

    bad_files = []
    good_files = 0

    for split in ["train", "val"]:
        split_path = os.path.join(DATA_DIR, split)

        if not os.path.exists(split_path):
            print(f"❌ Folder not found: {split_path}")
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)

            if not os.path.isdir(class_path):
                print(f"⚠️  Not a folder (delete this): {class_path}")
                continue

            for filename in os.listdir(class_path):
                filepath = os.path.join(class_path, filename)

                # Skip if it's a folder
                if os.path.isdir(filepath):
                    print(f"⚠️  Folder inside class (delete this): {filepath}")
                    bad_files.append(filepath)
                    continue

                # Skip non-image files
                if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                    print(f"⚠️  Not an image file (delete this): {filepath}")
                    bad_files.append(filepath)
                    continue

                # Try to open the image
                try:
                    img = Image.open(filepath)
                    img.convert("RGB")
                    img.load()  # Force full read
                    img.close()
                    good_files += 1
                except PermissionError:
                    print(f"🔒 PERMISSION DENIED: {filepath}")
                    bad_files.append(filepath)
                except Exception as e:
                    print(f"❌ CORRUPTED: {filepath} → {e}")
                    bad_files.append(filepath)

    print("=" * 50)
    print(f"✅ Good images: {good_files}")
    print(f"❌ Bad files:   {len(bad_files)}")

    if bad_files:
        print("\n⚠️  DELETE these files and try training again:")
        for f in bad_files:
            print(f"    {f}")
    else:
        print("\n✅ All images look fine!")

    return bad_files


if __name__ == "__main__":
    check_dataset()