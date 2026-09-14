"""Versioned NumPy I/O for the flattened DQME MPO."""

import json
import os
from pathlib import Path

import numpy as np

import params as pa


FORMAT_NAME = "split-ah-mps-dqme-mpo"
FORMAT_VERSION = 1


def _directory() -> Path:
    return Path("mpodata")


def _manifest_path() -> Path:
    return _directory() / f"{pa.pall_file}.manifest.json"


def _node_path(site: int) -> Path:
    return _directory() / f"{pa.pall_file}.node_{site:03d}.npy"


def _validate_flat_mpo(pall):
    if len(pall.nodes) != pa.nsite_total:
        raise ValueError(
            f"MPO has {len(pall.nodes)} sites, expected {pa.nsite_total}"
        )
    if not np.array_equal(np.asarray(pall.nb), pa.nbmat):
        raise ValueError("MPO physical dimensions do not match params.nbmat")
    for site, node in enumerate(pall.nodes):
        expected = int(pa.nbmat[site])
        if node.ndim != 3 or node.shape[1] != expected:
            raise ValueError(
                f"MPO site {site}: got {node.shape}, expected (rL,{expected},rR)"
            )
        if site + 1 < pa.nsite_total and node.shape[2] != pall.nodes[site + 1].shape[0]:
            raise ValueError(f"MPO bond mismatch between sites {site} and {site + 1}")


def write_mpo_file(pall):
    _validate_flat_mpo(pall)
    directory = _directory()
    directory.mkdir(parents=True, exist_ok=True)

    node_files = []
    for site, node in enumerate(pall.nodes):
        final_path = _node_path(site)
        temporary_path = final_path.with_suffix(final_path.suffix + ".tmp")
        with open(temporary_path, "wb") as stream:
            np.save(stream, np.asarray(node, dtype=np.complex128), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, final_path)
        node_files.append(final_path.name)

    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "nsite_total": int(pa.nsite_total),
        "nbmat": pa.nbmat.tolist(),
        "node_files": node_files,
        "node_shapes": [list(map(int, node.shape)) for node in pall.nodes],
        "dtype": "complex128",
    }
    manifest_path = _manifest_path()
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(temporary_manifest, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_manifest, manifest_path)
    print("MPO checkpoint written:", manifest_path)


def read_mpo_file():
    manifest_path = _manifest_path()
    if not manifest_path.exists():
        legacy = _directory() / f"{pa.pall_file}.dim"
        if legacy.exists():
            raise ValueError(
                f"legacy MPO checkpoint {legacy} belongs to the old layout and "
                "is intentionally not loaded; reconstruct the 60-site MPO"
            )
        raise FileNotFoundError(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"unsupported MPO format {manifest.get('format')!r}")
    if manifest.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported MPO format version {manifest.get('version')}")
    if manifest.get("nsite_total") != pa.nsite_total:
        raise ValueError(
            f"MPO checkpoint has {manifest.get('nsite_total')} sites; "
            f"current layout requires {pa.nsite_total}"
        )
    if manifest.get("nbmat") != pa.nbmat.tolist():
        raise ValueError("MPO checkpoint physical dimensions do not match params.nbmat")

    node_files = manifest.get("node_files", [])
    node_shapes = manifest.get("node_shapes", [])
    if len(node_files) != pa.nsite_total or len(node_shapes) != pa.nsite_total:
        raise ValueError("MPO manifest has an incomplete node list")

    pall = pa.Density()
    pall.nb = pa.nbmat.copy()
    for site, (filename, expected_shape) in enumerate(zip(node_files, node_shapes)):
        path = _directory() / filename
        node = np.load(path, allow_pickle=False)
        if node.dtype != np.complex128:
            node = node.astype(np.complex128)
        if list(node.shape) != expected_shape:
            raise ValueError(
                f"MPO node {site} shape {node.shape} disagrees with manifest "
                f"{expected_shape}"
            )
        pall.nodes.append(node)

    _validate_flat_mpo(pall)
    print("MPO checkpoint read:", manifest_path)
    return pall
