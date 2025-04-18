`gwsamplefind` allows access to inidividual events posterior samples and found injection sets.
It is primarily intended as a command line tool, but can also be used as a library.

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

If repeated calls are going to be made, the `--host` argument can be avoided by setting the `GWSAMPLEFIND_SERVER` environment variable.