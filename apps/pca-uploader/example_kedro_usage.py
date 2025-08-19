#!/usr/bin/env python3
"""
Example usage of the redesigned PCA uploader with Kedro catalog system.

Staff Engineer Implementation - Clean, simple API with checkpoint support.
"""

from upload_pca_to_neo4j import upload_pca_coordinates_from_catalog


def workbench_example(catalog):
    """Example for use in _workbench.py or similar Kedro environment."""

    # Simple upload with defaults
    upload_pca_coordinates_from_catalog(catalog)

    # Upload with custom settings
    upload_pca_coordinates_from_catalog(
        catalog,
        dataset_name="embeddings.reporting.topological_pca",
        batch_size=5000,
        max_workers=16,
    )


def main():
    """Show usage examples."""
    print("PCA Uploader - Clean Staff Engineer Implementation")
    print("=" * 60)

    print("\n🚀 CLI Usage (Recommended):")
    print("   python upload_pca_to_neo4j.py upload")
    print("   python upload_pca_to_neo4j.py upload --batch-size 5000 --max-workers 16")
    print("   python upload_pca_to_neo4j.py status  # Check progress")
    print("   python upload_pca_to_neo4j.py clean   # Clean checkpoints")

    print("\n📚 Programmatic Usage (from _workbench.py):")
    print("   from upload_pca_to_neo4j import upload_pca_coordinates_from_catalog")
    print("   upload_pca_coordinates_from_catalog(catalog)")

    print("\n✨ Key Features:")
    print("   • Automatic checkpoint/resume on interruption")
    print("   • Kedro-first design")
    print("   • Clean separation of concerns")
    print("   • Robust error handling")
    print("   • Modern Click CLI")


if __name__ == "__main__":
    main()
