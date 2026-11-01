"""Local, CPU-first lineage evidence; importing this package never loads weights."""

from .records import FingerprintRecord, compatible_views
from .retrieval import retrieve

__all__ = ['FingerprintRecord', 'compatible_views', 'retrieve']
