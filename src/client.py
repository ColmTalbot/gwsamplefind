import json
import os
from functools import cache

import pandas as pd
import requests

from .exceptions import GWSampleFindError

__all__ = ["Client"]


def _sample_request(request: str) -> pd.DataFrame:
    result = requests.get(request, verify=True)
    if result.status_code == 502:
        # retry on timeout error
        result = requests.get(request, verify=True)

    if result.status_code != 200:
        raise GWSampleFindError(f'Failed to process {request} with message {json.loads(result.content)["detail"]}')

    data = result.json()
    meta = data["metadata"]
    meta["model"] = data["model"]
    return pd.DataFrame(data["samples"], index=data["idxs"]), meta


class Client:

    _cache = dict()

    def __init__(self, host=None):
        if host is None:
            host = os.getenv("GWSAMPLEFIND_SERVER", None)
            if host is None:
                raise GWSampleFindError("No host provided and GWSAMPLEFIND_SERVER not set")
        self.host = host

    @classmethod
    def clear_cache(cls):
        cls._cache = dict()

    @cache
    def events(self):
        return requests.get(f"{self.host}/events").json()

    def samples(self, event, parameters, n_samples=-1, model="C01:IMRPhenomXPHM"):
        hash_ = f"{event}-{'&'.join(parameters)}-{n_samples}-{model}"
        if self._cache.get(hash_, None) is None:
            samples = self.get_samples(
                event=event,
                parameters=parameters,
                n_samples=n_samples,
                model=model,
            )
            self._cache[hash_] = samples
        return self._cache[hash_]

    def get_samples(self, event, parameters, n_samples=-1, model="C01:IMRPhenomXPHM", seed=None):
        var = "&".join([f"variable={par}" for par in parameters])
        request = f"{self.host}/events/{event}/?n_samples={n_samples}&{var}&seed={seed}&model={model}"
        return _sample_request(request)

    @cache
    def injection_sets(self):
        return requests.get(f"{self.host}/injections").json()

    def get_injections(self, injection_set, parameters, n_samples=-1, ifar_threshold=1, seed=None):
        var = "&".join([f"variable={par}" for par in parameters])
        request = f"{self.host}/injections/{injection_set}/?n_samples={n_samples}&{var}&seed={seed}&ifar_threshold={ifar_threshold}"
        return _sample_request(request)
