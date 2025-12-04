import os
from abc import ABC
from functools import cache

import pandas as pd

from .exceptions import GWSampleFindError

__all__ = ["Client"]


class Client(ABC):

    _cache = dict()

    def __init__(self, host=None):
        """
        >>> server = os.environ.pop("GWSAMPLEFIND_SERVER", None)
        >>> client = Client()
        Traceback (most recent call last):
            ...
        src.exceptions.GWSampleFindError: No host provided and GW_SAMPLEFIND_SERVER not set
        """
        self.host = host

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, host):
        if host is None:
            host = os.getenv("GWSAMPLEFIND_SERVER", None)
            if host is None:
                raise GWSampleFindError("No host provided and GWSAMPLEFIND_SERVER not set")
        self._host = host

    @classmethod
    def clear_cache(cls):
        cls._cache = dict()

    @cache
    def events(self) -> list[str]:
        """
        >>> client = Client()
        >>> client.events()[:3]
        ['GW150914_095045', 'GW151012_095443', 'GW151226_033853']
        """
        raise NotImplementedError

    def samples(
        self,
        event: str,
        parameters: list[str],
        *,
        n_samples: int = -1,
        model: str = "C01:IMRPhenomXPHM",
    ) -> tuple[pd.DataFrame, dict[str, str]]:
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

    def get_samples(
        self,
        event: str,
        parameters: list[str],
        *,
        n_samples: int = -1,
        model: str = "C01:IMRPhenomXPHM",
        seed: int | None = None,
        pattern: str| None = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        >>> client = Client()
        >>> client.get_samples("GW190403_051519", ["mass_1_source", "mass_2_source"], n_samples=10, seed=123)
        (       mass_1_source  mass_2_source
        171        84.189941      12.951107
        10120      65.196794      39.803265
        2453       93.339017      13.003137
        3715       90.226224      18.606987
        7594       58.999799      30.350026
        6600       77.465397      28.502002
        2840       77.523519      21.019240
        1959      106.907594      11.683538
        2053       95.548452      26.507532
        599        71.564036      25.451262, {'filename': 'IGWN-GWTC2p1-v2-GW190403_051519_PEDataRelease_mixed_cosmo.h5', 'model': 'C01:IMRPhenomXPHM'})
        """
        raise NotImplementedError

    @cache
    def injection_sets(self) -> list[str]:
        """
        >>> client = Client()
        >>> client.injection_sets()[:3]
        ['endo3_bbhpop', 'endo3_bnspop', 'endo3_imbhpop']
        """
        raise NotImplementedError

    def get_injections(
        self,
        injection_set: str,
        parameters: list[str],
        *,
        n_samples: int = -1,
        ifar_threshold: float = 1,
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        >>> client = Client()
        >>> client.get_injections("endo3_bbhpop", ["mass1_source"], n_samples=10, seed=123)
        (        mass1_source
        4924       11.553748
        18794       5.232413
        55483      89.296440
        58055       2.752108
        68562      60.005047
        80165      10.042098
        105525     62.770905
        178852     59.683643
        203918      2.689875
        261395     45.102314, {'filename': 'endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'analysis_time': 0.9110190889040992, 'total_generated': 73957576.0, 'n_found': 81117, 'model': 'injections'})
        """
        raise NotImplementedError
