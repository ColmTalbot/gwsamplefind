from fsspec.implementations.local import LocalFileSystem

from .fsspec import FSSpecClient

__all__ = ["LocalClient"]


class LocalClient(FSSpecClient):

    def __init__(self, path: str = None):
        fs = LocalFileSystem()
        super().__init__(fs=fs, path=path)
