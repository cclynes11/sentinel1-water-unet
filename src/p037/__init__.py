"""P037: landscape proxies for out-of-distribution performance of SAR flood models.

One workflow, run the same way for every experiment; only the dataset and the model
initialisation change (see ``analysis/p037/README.md``):

    data -> train -> evaluate -> proxies -> analyze

Each stage reads files written by the previous one, so any stage can be re-run alone.
"""

__version__ = "0.1.0"
