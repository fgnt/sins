import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


class Normalizer:
    """perform global mean and scale normalization.
    """
    def __init__(
            self, key, center_axis=None, scale_axis=None, storage_dir=None,
            name=None
    ):
        """

        Args:
            key: key to be normalized within an example dict
            center_axis: int or tuple stating the axis over which to compute
                the mean
            scale_axis: int or tuple stating the axis over which to compute
                the scale (variance if center_axis != None)
            storage_dir: directory where to store the precomputed means and scales.
            name: file prefix when storing moments
        """
        self.key = key
        self.center_axis = None if center_axis is None else tuple(center_axis)
        self.scale_axis = None if scale_axis is None else tuple(scale_axis)
        self.storage_dir = None if storage_dir is None else Path(storage_dir)
        self.name = name
        self.moments = None

    def normalize(self, x):
        assert self.moments is not None
        mean, scale = self.moments
        x -= mean
        x /= (scale + 1e-18)
        return x

    def __call__(self, example):
        example[self.key] = self.normalize(example[self.key])
        return example

    def initialize_moments(self, dataset=None, verbose=False):
        """loads or computes the global mean (center) and scale over a dataset.

        Args:
            dataset: lazy dataset providing example dicts
            verbose:

        """
        filepath = None if self.storage_dir is None \
            else self.storage_dir / f"{self.key}_moments_{self.name}.json" \
            if self.name else self.storage_dir / f"{self.key}_moments.json"
        if filepath is not None and Path(filepath).exists():
            with filepath.open() as fid:
                mean, scale = json.load(fid)
            if verbose:
                print(f'Restored moments from {filepath}')
        else:
            assert dataset is not None
            mean = 0.
            mean_count = 0
            energy = 0.
            energy_count = 0
            for example in tqdm(dataset, disable=not verbose):
                x = example[self.key]
                if self.center_axis is not None:
                    if not mean_count:
                        mean = np.sum(x, axis=self.center_axis, keepdims=True)
                    else:
                        mean += np.sum(x, axis=self.center_axis, keepdims=True)
                    mean_count += np.prod(
                        np.array(x.shape)[np.array(self.center_axis)]
                    )
                if self.scale_axis is not None:
                    if not energy_count:
                        energy = np.sum(x**2, axis=self.scale_axis, keepdims=True)
                    else:
                        energy += np.sum(x**2, axis=self.scale_axis, keepdims=True)
                    energy_count += np.prod(
                        np.array(x.shape)[np.array(self.scale_axis)]
                    )
            if self.center_axis is not None:
                mean /= mean_count
            if self.scale_axis is not None:
                energy /= energy_count
                scale = np.sqrt(np.mean(
                    energy - mean ** 2, axis=self.scale_axis, keepdims=True
                ))
            else:
                scale = np.array(1.)

            if filepath is not None:
                with filepath.open('w') as fid:
                    json.dump(
                        (mean.tolist(), scale.tolist()), fid,
                        sort_keys=True, indent=4
                    )
                if verbose:
                    print(f'Saved moments to {filepath}')
        self.moments = np.array(mean), np.array(scale)
