import re
from collections.abc import Iterable
from functools import cache
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

from .base import Client

__all__ = ["FSSpecClient"]


class FSSpecClient(Client):

    def __init__(self, fs: "AbstractFileSystem", path: str | Iterable):
        self.fs = fs
        if isinstance(path, str):
            self.files = self.fs.ls(path, detail=False)
        elif isinstance(path, Iterable):
            self.files = list()
            for path_ in path:
                self.files.extend(self.fs.ls(path_, detail=False))
        else:
            raise ValueError(f"Unable to process provided path {path}")

    @cache
    def events(self):
        pattern = r"GW\d{6}_\d{6}"
        posterior_files = filter(lambda fname: re.search(pattern, fname), self.files)
        events = [re.search(pattern, match).group() for match in posterior_files]
        return sorted(list(set(events)))

    def samples(self, event, parameters, n_samples=-1, model="C01:IMRPhenomXPHM"):
        raise NotImplementedError

    def get_samples(self, event, parameters, n_samples=-1, model="C01:IMRPhenomXPHM", seed=None, pattern=None):
        event_filenames = list(filter(lambda fname: event in fname, self.files))

        if isinstance(pattern, str):
            event_filenames = list(filter(lambda fname: pattern in fname, event_filenames))

        if len(event_filenames) == 1:
            event_filename = event_filenames[0]
        elif len(event_filenames) == 0:
            raise FileNotFoundError(
                f"Failed to identify {event} matching {pattern} file in {self.files}"
            )
        else:
            raise ValueError(f"Found multiple files matching {event}: {event_filename}")

        with h5py.File(self.fs.open(event_filename, "rb")) as ff:
            if model not in ff:
                raise KeyError(
                    f"{model} not found in {event_filename}. "
                    f"Available models are {', '.join(ff.keys())}"
                )
            dataset = ff[model]["posterior_samples"]
            total_n_samples = len(dataset)

            if n_samples == -1:
                n_samples = total_n_samples
            elif n_samples > total_n_samples:
                raise ValueError(
                    f"{n_samples} requested but only {total_n_samples} "
                    f"available for {event}-{model}"
                )
            
            generator = np.random.Generator(np.random.PCG64(seed=seed))
            idxs = np.sort(generator.choice(total_n_samples, n_samples, replace=False).astype(int))

            samples = dict()
            for parameter in parameters:
                try:
                    samples[parameter] = dataset[parameter][idxs]
                except ValueError as e:
                    raise KeyError(f"Parameter {parameter} not available for {event}-{model}") from e

        posterior = pd.DataFrame(samples, index=idxs)

        metadata = dict(model=model, filename=f"{self.host}{event_filename}")
        return posterior, metadata

    def injection_sets(self):
        raise NotImplementedError

    def get_injections(self, injection_set, parameters, n_samples=-1, ifar_threshold=1, seed=None):
        raise NotImplementedError
