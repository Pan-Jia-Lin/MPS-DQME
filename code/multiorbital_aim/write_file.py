"""Versioned binary RDT checkpoint I/O for the 60-site split MPS."""

import json
import os
from pathlib import Path

import numpy as np

import params as pa


FORMAT_NAME = "split-ah-mps-dqme-rdt"
FORMAT_VERSION = 1


def _checkpoint_directory(filename) -> Path:
    # Keep the historical user-facing filename while storing variable-shaped
    # tensors safely in a companion directory.
    path = Path(filename)
    return path.with_name(path.name + ".nodes")


def _manifest_path(filename) -> Path:
    return Path(filename)


def _validate_density(rhoall):
    if len(rhoall.nodes) != pa.nsite_total:
        raise ValueError(
            f"RDT has {len(rhoall.nodes)} sites, expected {pa.nsite_total}"
        )
    if not np.array_equal(np.asarray(rhoall.nb), pa.nb):
        raise ValueError("RDT physical dimensions do not match params.nb")
    for site, node in enumerate(rhoall.nodes):
        expected = int(pa.nb[site])
        if node.ndim != 3 or node.shape[1] != expected:
            raise ValueError(
                f"RDT site {site}: got {node.shape}, expected (rL,{expected},rR)"
            )
        if site == 0 and node.shape[0] != 1:
            raise ValueError("RDT left boundary dimension is not one")
        if site == pa.nsite_total - 1 and node.shape[2] != 1:
            raise ValueError("RDT right boundary dimension is not one")
        if site + 1 < pa.nsite_total and node.shape[2] != rhoall.nodes[site + 1].shape[0]:
            raise ValueError(f"RDT bond mismatch between sites {site} and {site + 1}")


def write_density(rhoall, filename, newfile=False):
    del newfile  # retained for compatibility; versioned checkpoints overwrite atomically
    _validate_density(rhoall)
    manifest_path = _manifest_path(filename)
    node_directory = _checkpoint_directory(filename)
    node_directory.mkdir(parents=True, exist_ok=True)

    node_files = []
    for site, node in enumerate(rhoall.nodes):
        final_path = node_directory / f"node_{site:03d}.npy"
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
        "nb": pa.nb.tolist(),
        "nrtt_parameter": int(pa.nrtt),
        "node_directory": node_directory.name,
        "node_files": node_files,
        "node_shapes": [list(map(int, node.shape)) for node in rhoall.nodes],
        "dtype": "complex128",
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(temporary_manifest, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_manifest, manifest_path)
    return 0


def read_density(filename):
    manifest_path = _manifest_path(filename)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    try:
        with open(manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{filename} is a legacy text checkpoint without a 60-site layout "
            "manifest; start a new run or convert it explicitly"
        ) from exc

    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"unsupported RDT checkpoint format {manifest.get('format')!r}")
    if manifest.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported RDT checkpoint version {manifest.get('version')}"
        )
    if manifest.get("nsite_total") != pa.nsite_total:
        raise ValueError(
            f"RDT checkpoint has {manifest.get('nsite_total')} sites; "
            f"current layout requires {pa.nsite_total}"
        )
    if manifest.get("nb") != pa.nb.tolist():
        raise ValueError("RDT checkpoint physical dimensions do not match params.nb")

    node_directory = manifest_path.parent / manifest["node_directory"]
    node_files = manifest.get("node_files", [])
    node_shapes = manifest.get("node_shapes", [])
    if len(node_files) != pa.nsite_total or len(node_shapes) != pa.nsite_total:
        raise ValueError("RDT manifest has an incomplete node list")

    rhoall = pa.Density()
    rhoall.nb = pa.nb.copy()
    for site, (filename_node, expected_shape) in enumerate(
        zip(node_files, node_shapes)
    ):
        node = np.load(node_directory / filename_node, allow_pickle=False)
        if node.dtype != np.complex128:
            node = node.astype(np.complex128)
        if list(node.shape) != expected_shape:
            raise ValueError(
                f"RDT node {site} shape {node.shape} disagrees with manifest "
                f"{expected_shape}"
            )
        rhoall.nodes.append(node)

    _validate_density(rhoall)
    return rhoall
