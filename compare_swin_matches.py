import argparse
import os
from pathlib import Path
from PIL import Image
from swin_faiss import load_swin_faiss_classifier


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def build_metadata_index(image_paths):
    index = {}
    for entry in image_paths:
        path = normalize_path(entry.get("path", ""))
        if path:
            index[path] = entry
            index[os.path.basename(path)] = entry
    return index


def print_neighbors(neighbors):
    print("Top neighbors:")
    for rank, n in enumerate(neighbors, start=1):
        print(
            f"  {rank}. path={n.get('path')} | label={n.get('label')} | subcat={n.get('subcategory')} | score={n.get('score'):.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect SWIN FAISS neighbor matches for sample crop images.")
    parser.add_argument("image", help="Path to the crop or shelf image to inspect.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top neighbors to print.")
    parser.add_argument("--show-csv-metadata", action="store_true", help="Show matched CSV metadata for the top neighbors.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    classifier = load_swin_faiss_classifier()
    if not classifier.is_ready():
        raise RuntimeError("Swin FAISS classifier is not ready. Check FAISS installation and resource files.")

    image = Image.open(image_path).convert("RGB")
    neighbors = classifier.query(image, top_k=args.top_k)

    print(f"Query image: {image_path}")
    print(f"Loaded {len(classifier.image_paths)} indexed entries")
    print_neighbors(neighbors)

    if args.show_csv_metadata:
        metadata_index = build_metadata_index(classifier.image_paths)
        matched = metadata_index.get(normalize_path(image_path.name)) or metadata_index.get(normalize_path(str(image_path)))
        if matched:
            print("\nMatched CSV metadata for query image:")
            print(matched)
        else:
            print("\nNo exact CSV metadata match found for the query image path.")

    if neighbors:
        print("\nTop neighbor CSV matches:")
        metadata_index = build_metadata_index(classifier.image_paths)
        for rank, neighbor in enumerate(neighbors[: min(5, len(neighbors))], start=1):
            key = normalize_path(neighbor.get("path", ""))
            metadata = metadata_index.get(key) or metadata_index.get(os.path.basename(key))
            if metadata:
                print(f"  {rank}. csv label={metadata.get('label')} subcat={metadata.get('subcategory')} path={metadata.get('path')}")
            else:
                print(f"  {rank}. no CSV metadata for path={neighbor.get('path')}")


if __name__ == "__main__":
    main()
