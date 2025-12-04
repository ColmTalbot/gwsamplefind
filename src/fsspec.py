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
        self._injection_files = list()

        self.fs = fs
        if isinstance(path, str):
            self.files = self.fs.ls(path, detail=False)
        elif isinstance(path, Iterable):
            self.files = list()
            for path_ in path:
                new_files = self.fs.ls(path_, detail=False)
                if any([fname.endswith("sensitivity-estimates.md") for fname in new_files]):
                    self._injection_files.extend(filter(
                        lambda fname: fname.endswith(".hdf5") or fname.endswith(".hdf"),
                        new_files,
                    ))
                self.files.extend(self.fs.ls(path_, detail=False))
        else:
            raise ValueError(f"Unable to process provided path {path}")

    @cache
    def events(self) -> list[str]:
        pattern = r"GW\d{6}_\d{6}"
        posterior_files = filter(lambda fname: re.search(pattern, fname), self.files)
        events = [re.search(pattern, match).group() for match in posterior_files]
        return sorted(list(set(events)))

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
            raise ValueError(f"Found multiple files matching {event}: {event_filenames}")

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
            
            generator = np.random.default_rng(seed=seed)
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

    def injection_sets(self) -> list[str]:
        return [fname.split("/")[-1].rsplit(".", maxsplit=1)[0] for fname in self._injection_files]

    def get_injections(
        self,
        injection_set: str,
        parameters: list[str],
        *,
        n_samples: int = -1,
        ifar_threshold: float = 1,
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        injection_filenames = list(filter(lambda fname: injection_set in fname, self.files))

        if len(injection_filenames) == 1:
            injection_filename = injection_filenames[0]
        elif len(injection_filenames) == 0:
            raise FileNotFoundError(
                f"Failed to identify injection set matching {injection_set} file in {self.files}"
            )
        else:
            raise ValueError(f"Found multiple files matching {injection_set}: {injection_filenames}")

        with h5py.File(self.fs.open(injection_filename, "rb")) as ff:
            if "injections" in ff:
                dataset = ff["injections"]
                total_n_samples = len(dataset)

                metadata = dict()
                metadata["analysis_time"] = float(dataset.attrs["analysis_time_s"] / (365.25 * 24 * 60 * 60))
                metadata["total_generated"] = float(dataset.attrs["total_generated"])

                keep = np.zeros(len(dataset["sampling_pdf"]), dtype=bool)
                for key in dataset:
                    if "ifar" in key:
                        keep |= dataset[key][()] > ifar_threshold

                total_n_samples = sum(keep)
                metadata["n_found"] = int(total_n_samples)
            elif "events" in ff:
                dataset = ff["events"]
                total_n_samples = len(dataset)

                metadata = dict()
                metadata["analysis_time"] = float(ff.attrs["total_analysis_time"] / (365.25 * 24 * 60 * 60))
                metadata["total_generated"] = float(ff.attrs["total_generated"])

                keep = np.zeros(len(dataset), dtype=bool)
                for key in dataset.dtype.names:
                    if "far" in key:
                        keep |= 1 / dataset[key][()] > ifar_threshold

                total_n_samples = sum(keep)
                metadata["n_found"] = int(total_n_samples)
            else:
                raise ValueError(
                    f"Unable to parse {injection_filename}."
                )

            if n_samples == -1:
                n_samples = total_n_samples
            elif n_samples > total_n_samples:
                raise ValueError(
                    f"{n_samples} requested but only {total_n_samples} available for {injection_set}"
                )
            
            generator = np.random.default_rng(seed=seed)
            idxs = np.sort(generator.choice(total_n_samples, n_samples, replace=False).astype(int))

            samples = dict()
            for parameter in parameters:
                try:
                    samples[parameter] = dataset[parameter][idxs]
                except ValueError as e:
                    raise KeyError(f"Parameter {parameter} not available for {injection_set}") from e

            posterior = pd.DataFrame(samples, index=idxs)
        return posterior, metadata
