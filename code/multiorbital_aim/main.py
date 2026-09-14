"""Executable entry point for the pure-fermion multi-orbital AIM MPS-DQME."""

from time import perf_counter

import numpy as np

import construct as cst
import init_rho as ir
import params as pa
import prop as pp
import read_write_mpo as rwm
import write_file as wf


def _run_correlation_enabled():
    """Safe default for params.py versions predating this entry-point option."""
    return bool(getattr(pa, "run_correlation", False))


def _rho_checkpoint_file():
    """Safe default for params.py versions predating versioned checkpoints."""
    return str(getattr(pa, "rho_checkpoint_file", "rhoall.dat"))


def validate_mps(rho, label="MPS"):
    if len(rho.nodes) != pa.nsite_total:
        raise ValueError(
            f"{label}: {len(rho.nodes)} sites, expected {pa.nsite_total}"
        )
    if not np.array_equal(np.asarray(rho.nb), pa.nb):
        raise ValueError(f"{label}: physical-dimension metadata does not match params.nb")
    for site, node in enumerate(rho.nodes):
        expected_physical = int(pa.nb[site])
        if node.ndim != 3:
            raise ValueError(f"{label} site {site}: expected rank 3, got {node.shape}")
        if node.shape[1] != expected_physical:
            raise ValueError(
                f"{label} site {site}: physical dimension {node.shape[1]}, "
                f"expected {expected_physical}"
            )
        if site == 0 and node.shape[0] != 1:
            raise ValueError(f"{label}: left boundary dimension is not one")
        if site == pa.nsite_total - 1 and node.shape[2] != 1:
            raise ValueError(f"{label}: right boundary dimension is not one")
        if site + 1 < pa.nsite_total:
            next_left = rho.nodes[site + 1].shape[0]
            if node.shape[2] != next_left:
                raise ValueError(
                    f"{label}: bond mismatch {site}-{site + 1}: "
                    f"{node.shape[2]} != {next_left}"
                )


def validate_flat_mpo(mpo, label="flattened MPO"):
    if len(mpo.nodes) != pa.nsite_total:
        raise ValueError(
            f"{label}: {len(mpo.nodes)} sites, expected {pa.nsite_total}"
        )
    if not np.array_equal(np.asarray(mpo.nb), pa.nbmat):
        raise ValueError(f"{label}: metadata does not match params.nbmat")
    for site, node in enumerate(mpo.nodes):
        expected = int(pa.nbmat[site])
        if node.ndim != 3:
            raise ValueError(f"{label} site {site}: expected rank 3, got {node.shape}")
        if node.shape[1] != expected:
            raise ValueError(
                f"{label} site {site}: flattened physical dimension "
                f"{node.shape[1]}, expected {expected}"
            )
        if site == 0 and node.shape[0] != 1:
            raise ValueError(f"{label}: left boundary dimension is not one")
        if site == pa.nsite_total - 1 and node.shape[2] != 1:
            raise ValueError(f"{label}: right boundary dimension is not one")
        if site + 1 < pa.nsite_total and node.shape[2] != mpo.nodes[site + 1].shape[0]:
            raise ValueError(f"{label}: bond mismatch between sites {site} and {site + 1}")


def validate_operator_mpo(mpo, label="operator MPO"):
    if len(mpo.nodes) != pa.nsite_total:
        raise ValueError(
            f"{label}: {len(mpo.nodes)} sites, expected {pa.nsite_total}"
        )
    for site, node in enumerate(mpo.nodes):
        d = int(pa.nb[site])
        if node.ndim != 4 or node.shape[1:3] != (d, d):
            raise ValueError(
                f"{label} site {site}: got {node.shape}, expected "
                f"(r_left, {d}, {d}, r_right)"
            )
        if site + 1 < pa.nsite_total and node.shape[3] != mpo.nodes[site + 1].shape[0]:
            raise ValueError(f"{label}: bond mismatch between sites {site} and {site + 1}")


def copy_mps(source):
    result = pa.Density()
    result.nb = np.asarray(source.nb, dtype=np.int64).copy()
    result.nodes = [np.asarray(node, dtype=np.complex128).copy() for node in source.nodes]
    validate_mps(result, "copied MPS")
    return result


def unflatten_mpo(source):
    """Convert (rL, d*d, rR) nodes to (rL, d_out, d_in, rR)."""
    validate_flat_mpo(source)
    result = pa.Density()
    # The TDVP operator has four-index nodes; this metadata is informational.
    result.nb = pa.nb.copy()
    for site, node in enumerate(source.nodes):
        left_rank, flattened_dim, right_rank = node.shape
        d = int(pa.nb[site])
        if d * d != flattened_dim:
            raise ValueError(
                f"MPO site {site}: {flattened_dim} is not {d} squared"
            )
        result.nodes.append(
            np.reshape(
                node.copy(),
                (left_rank, d, d, right_rank),
                order="F",
            )
        )
    validate_operator_mpo(result)
    return result


# Compatibility names used by earlier scripts.
copy2tensor = copy_mps
copy2tensorMat0 = unflatten_mpo


def print_layout_summary():
    print("=" * 72)
    print("Pure-fermion multi-orbital AIM MPS-DQME")
    print("total MPS sites:", pa.nsite_total)
    print("system fermion ket sites:", pa.system_fermion_ket_site)
    print("fermion dissipaton ket sites:", (pa.site_fermion_diss_ket[0], pa.site_fermion_diss_ket[-1]))
    print("fermion dissipaton bra sites:", (pa.site_fermion_diss_bra[0], pa.site_fermion_diss_bra[-1]))
    print("system fermion bra sites:", pa.system_fermion_bra_site)
    print("RDO dimension:", pa.ndvr_dense, "x", pa.ndvr_dense)
    print("read MPO:", pa.read_pall, "write MPO:", pa.write_pall)
    print("read RDT:", pa.Isread_rho, "propagate:", pa.Isprop_rho, "write RDT:", pa.Iswrite_rho)
    print("correlation enabled:", _run_correlation_enabled())
    print("=" * 72)


def _load_or_initialize_rho():
    if pa.Isread_rho:
        checkpoint = _rho_checkpoint_file()
        rho = wf.read_density(checkpoint)
        validate_mps(rho, "loaded RDT checkpoint")
        print("loaded RDT checkpoint:", checkpoint)
        return rho
    rho = ir.init_rho()
    validate_mps(rho, "initial RDT")
    return rho


def _load_or_construct_mpo():
    if pa.read_pall:
        mpo = rwm.read_mpo_file()
        validate_flat_mpo(mpo, "loaded flattened MPO")
        print("loaded MPO checkpoint")
        return mpo
    mpo = cst.construct(Iscorr=False)
    validate_flat_mpo(mpo, "constructed flattened MPO")
    if pa.write_pall:
        rwm.write_mpo_file(mpo)
    return mpo


def _run_correlations_if_requested(rho):
    if not _run_correlation_enabled():
        return
    # Import lazily: ordinary propagation does not depend on the legacy module.
    import corrfun as cf

    for is_commutator in (True, False):
        corr_flat = cst.construct(Iscorr=True, Iscommu=is_commutator)
        corr_mpo = unflatten_mpo(corr_flat)
        cf.corrfun(rho.copy(), corr_mpo, is_commutator)


def main():
    start = perf_counter()
    print_layout_summary()

    rho = _load_or_initialize_rho()
    flat_mpo = _load_or_construct_mpo()
    mpo = unflatten_mpo(flat_mpo)
    print("initialization and MPO construction completed")

    if pa.Isprop_rho:
        result = pp.prop(copy_mps(rho), mpo)
    elif pa.Isread_rho:
        result = rho
        print("propagation disabled; using the loaded RDT")
    else:
        raise RuntimeError(
            "nothing to do: both Isprop_rho and Isread_rho are False"
        )

    validate_mps(result, "final RDT")
    if pa.Iswrite_rho:
        checkpoint = _rho_checkpoint_file()
        wf.write_density(result, checkpoint, newfile=True)
        print("wrote RDT checkpoint:", checkpoint)

    _run_correlations_if_requested(result)
    print(f"all requested tasks completed in {perf_counter() - start:.3f} s")
    return result


if __name__ == "__main__":
    main()
