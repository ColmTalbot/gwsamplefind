import json
from functools import cache

import pandas as pd
import requests

from .exceptions import GWSampleFindError

__all__ = ["Client"]


def _sample_request(request: str) -> tuple[pd.DataFrame, dict]:
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


class SampleFindClient:

    @cache
    def events(self) -> list[str]:
        """
        >>> client = SampleFindClient("https://gwsamples.duckdns.org")
        >>> client.events()[:3]
        ['GW150914_095045', 'GW151012_095443', 'GW151226_033853']
        >>> client.get_samples("GW190403_051519", ["mass_1_source", "mass_2_source"], 10, seed=123)
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
        return requests.get(f"{self.host}/events").json()

    def get_samples(
                self,
        event: str,
        parameters: list[str],
        *,
        n_samples: int = -1,
        model: str = "C01:IMRPhenomXPHM",
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        >>> client = SampleFindClient("https://gwsamples.duckdns.org")
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
        var = "&".join([f"variable={par}" for par in parameters])
        request = f"{self.host}/events/{event}/?n_samples={n_samples}&{var}&seed={seed}&model={model}"
        return _sample_request(request)

    @cache
    def injection_sets(self) -> list[str]:
        """
        >>> client = SampleFindClient("https://gwsamples.duckdns.org")
        >>> client.injection_sets()[:3]
        ['endo3_bbhpop', 'endo3_bnspop', 'endo3_imbhpop']
        """
        return requests.get(f"{self.host}/injections").json()

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
        >>> client = SampleFindClient("https://gwsamples.duckdns.org")
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
        var = "&".join([f"variable={par}" for par in parameters])
        request = f"{self.host}/injections/{injection_set}/?n_samples={n_samples}&{var}&seed={seed}&ifar_threshold={ifar_threshold}"
        return _sample_request(request)
