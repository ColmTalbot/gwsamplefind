from pelicanfs.core import PelicanFileSystem

from .fsspec import FSSpecClient

__all__ = ["PelicanClient"]


class PelicanClient(FSSpecClient):

    def __init__(self, *, host: str | None = None, path: str, **kwargs):
        self.host = host
        fs = PelicanFileSystem(self.host, **kwargs)
        super().__init__(fs=fs, path=path)
