:code:`gwsamplefind` allows access to inidividual events posterior samples and found injection sets.
It is primarily intended as a command line tool, but can also be used as a library.

.. warning::

    This code (and associated server) is currently very much in development and so may not be reliable.
    Especially, the specific host referred to below may not be available at all times.

Basic Usage
-----------

To download a set of samples for all events more significant than a given inverse false alarm rate (IFAR), you can use the following command:

.. code-block:: bash

    $ python -m gwsamplefind --outdir ./tmp --n-samples 10 --parameters mass_1_source --seed 10 --host https://gwsamples.duckdns.org --ifar-threshold 5

To select only a subset of events you can use the `--events` flag:

.. code-block:: bash

    $ python -m gwsamplefind --outdir ./tmp --n-samples 10 --parameters mass_1_source --seed 10 --host https://gwsamples.duckdns.org --ifar-threshold 5 --events GW150914_095045 GW190517_055101

.. note::

    The `--events` flag is a space-separated list of event names using the `GWYYMMDD_SUBDAY` format.

Alternatively, to download a set of injections passing a matching threshold on IFAR, you can use the following command:

.. code-block:: bash

    $ python -m gwsamplefind --outdir ./tmp --n-samples 10 --parameters mass1_source --seed 10 --host https://gwsamples.duckdns.org --ifar-threshold 5 --injection-set o1+o2+o3_bbhpop_real+semianalytic

If repeated calls are going to be made, the `--host` argument can be avoided by setting the :code:`GWSAMPLEFIND_SERVER` environment variable.

Alternatively, :code:`gwsamplefind` can be used as a library:

.. code-block:: python

    In [1]: from gwsamplefind.client import Client

    In [2]: client = Client("https://gwsamples.duckdns.org")

    In [3]: client.events()[:3]
    Out[3]: ['GW150914_095045', 'GW151012_095443', 'GW151226_033853']

    In [4]: client.samples("GW190403_051519", ["mass_1_source", "mass_2_source"], 10, seed=123)
    Out[4]: (       mass_1_source  mass_2_source
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
