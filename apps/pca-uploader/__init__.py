"""
PCA Uploader Application

A Python application to efficiently upload PCA coordinates from embeddings to Neo4j nodes.
"""

__version__ = "1.0.0"
__author__ = "Matrix Team"

from .upload_pca_to_neo4j import (
    Neo4jPCAUploader,
    upload_pca_coordinates,
    upload_pca_coordinates_from_catalog,
    upload_pca_coordinates_from_file,
)

__all__ = [
    "upload_pca_coordinates",
    "upload_pca_coordinates_from_catalog",
    "upload_pca_coordinates_from_file",
    "Neo4jPCAUploader",
]
