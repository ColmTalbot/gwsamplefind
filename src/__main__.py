import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from gwosc.api import fetch_filtered_events_json

from .client import Client
from .exceptions import GWSampleFindError

SIMPLE_EVENT_NAMES = dict(
    GW150914="GW150914_095045",
    GW151012="GW151012_095443",
    GW151226="GW151226_033853",
    GW170104="GW170104_101158",
    GW170608="GW170608_020116",
    GW170729="GW170729_185629",
    GW170809="GW170809_082821",
    GW170814="GW170814_103043",
    GW170818="GW170818_022509",
    GW170823="GW170823_131358",
    GW190412="GW190412_053044",
    GW190425="GW190425_081805",
    GW190521="GW190521_030229",
    GW190814="GW190814_211039",
)


def create_parser():

    parser = ArgumentParser(description="GW Sample Finder")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to use for the search",
    )
    parser.add_argument(
        "--events",
        nargs="*",
        default=None,
        help="List of events to search for",
    )
    parser.add_argument(
        "--parameters",
        nargs="*",
        default=None,
        help="List of parameters to search for",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=-1,
        help="Number of samples to search for",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="C01:IMRPhenomXPHM",
        help="Model to use for the search",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use for the search",
    )
    parser.add_argument(
        "--injection-set",
        type=str,
        default=None,
        help="Injection set to use for the search",
    )
    parser.add_argument(
        "--ifar-threshold",
        type=float,
        default=1,
        help="IFAR threshold to use for the search",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Output directory to save the results",
    )

    return parser


def _fetch_events(events, args, client, strict=True):
    event_data = dict()
    metadata = dict()
    for event in events:
        try:
            event_data[event], metadata[event] = client.get_samples(
                event=event,
                parameters=args.parameters,
                n_samples=args.n_samples,
                model=args.model,
                seed=args.seed,
            )
        except GWSampleFindError as e:
            if "Sample file" in str(e):
                print(f"Sample file not found for {event}")
            elif "Available models" in str(e):
                print(f"{args.model} not found for {event}")
            else:
                print(f"Failed to fetch samples for {event} with {e}")
            if strict:
                raise
            else:
                continue
    return event_data, metadata


def main(args=None):
    args = create_parser().parse_args(args=args)

    client = Client(host=args.host)

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.injection_set is not None:
        injections, metadata = client.get_injections(
            injection_set=args.injection_set,
            parameters=args.parameters,
            n_samples=args.n_samples,
            ifar_threshold=args.ifar_threshold,
            seed=args.seed,
        )
        injections.to_pickle(args.outdir / f"injections.pkl")
        with open(args.outdir / "injection_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    elif args.events is None:
        data = fetch_filtered_events_json([f"far <= {1 / args.ifar_threshold}"])
        events = [data["events"][event]["commonName"] for event in data["events"]]
        events = list(dict.fromkeys(events))
        events = [SIMPLE_EVENT_NAMES.get(event, event) for event in events]
        event_data, metadata = _fetch_events(events, args, client, strict=False)
        event_data = [event_data[event] for event in event_data]
        pd.to_pickle(event_data, args.outdir / "samples.pkl")
        with open(args.outdir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    
    else:
        event_data, metadata = _fetch_events(args.events, args, client, strict=True)
        event_data = [event_data[event] for event in event_data]
        pd.to_pickle(event_data, args.outdir / "samples.pkl")
        with open(args.outdir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()