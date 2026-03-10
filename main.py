"""
Rice Leaf Disease Classification using CNN
===========================================
A simple deep learning program to classify rice leaf diseases
into categories: Bacterial Leaf Blight, Brown Spot, Leaf Smut, and Healthy.

Usage:
    Train:   python main.py train
    Predict: python main.py predict <image_path>
"""

import sys


def print_banner():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║   🌾 Rice Leaf Disease Classification (CNN)      ║
    ║                                                  ║
    ║   Classes:                                       ║
    ║     1. Brown Spot                                ║
    ║     2. Healthy                                   ║
    ║     3. Hispa                                     ║
    ║     4. Leaf Blast                                ║
    ╚══════════════════════════════════════════════════╝
    """)


def main():
    print_banner()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py train              - Train the model")
        print("  python main.py predict <image>     - Predict disease from image")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        from train import main as train_main
        train_main()

    elif command == "predict":
        if len(sys.argv) < 3:
            print("Error: Please provide an image path.")
            print("Usage: python main.py predict <image_path>")
            sys.exit(1)
        from predict import predict
        predict(sys.argv[2])

    else:
        print(f"Unknown command: {command}")
        print("Use 'train' or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    main()