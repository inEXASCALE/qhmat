"""Stage 1: generate independent low-rank inputs for the allocator study."""
import argparse

import numpy as np

from qmat_exp.allocator_study import (experiment_spec, sha256, spec_hash,
                                     synthetic_case, validate_config)
from qmat_exp.io import ensure_dir, load_config, write_json


def prepare(cfg):
    validate_config(cfg)
    directory = ensure_dir(cfg["data_dir"])
    items = []
    for spectrum in cfg["spectra"]:
        for seed in cfg["seeds"]:
            case_id = f"{spectrum}_seed{seed}"
            u, sigma, v = synthetic_case(cfg, spectrum, seed)
            path = directory / f"{case_id}.npz"
            np.savez_compressed(path, u=u, sigma=sigma, v=v)
            items.append(dict(case_id=case_id, spectrum=spectrum, seed=seed,
                              filename=path.name, sha256=sha256(path)))
    manifest = dict(spec_hash=spec_hash(cfg), spec=experiment_spec(cfg), cases=items)
    write_json(directory / "manifest.json", manifest)
    print(f"Prepared {len(items)} independent factor triples in {directory}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/allocator_study.json")
    prepare(load_config(parser.parse_args().config))
