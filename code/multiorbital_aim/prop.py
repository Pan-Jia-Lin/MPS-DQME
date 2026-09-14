"""TDVP propagation and output for the split-system MPS-DQME.

The TDVP implementation itself remains in ``ksltt.py`` and is not modified.
This module only coordinates propagation, RDO reconstruction, observables,
lead/spin-resolved currents, and continuity-equation diagnostics.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, TextIO, Tuple

import numpy as np

import calc_rho as cr
import ksltt as ksl
import params as pa


@dataclass
class Snapshot:
    time: float
    rho: np.ndarray
    observables: Dict[str, complex]
    currents: cr.CurrentResult


def _calculate_snapshot(rin, time: float) -> Snapshot:
    rho = cr.calc_rho(rin)
    observables = cr.calc_observables_from_rho(rho)
    currents = cr.calc_spin_resolved_currents(rin)
    return Snapshot(float(time), rho, observables, currents)


def _write_real_diagonal(stream: TextIO, snapshot: Snapshot) -> None:
    values = [f"{snapshot.time:.16e}"]
    values.extend(f"{value.real:.16e}" for value in np.diag(snapshot.rho))
    stream.write("  ".join(values) + "\n")
    stream.flush()


def _write_complex_value(stream: TextIO, time: float, value: complex) -> None:
    stream.write(f"{time:.16e}  {value.real:.16e}  {value.imag:.16e}\n")
    stream.flush()


def _write_rdo(stream: TextIO, step: int, snapshot: Snapshot) -> None:
    stream.write(f"# istep={step}  time={snapshot.time:.16e}\n")
    for row in snapshot.rho:
        stream.write(
            "["
            + ",  ".join(
                f"{value.real:.16e}{value.imag:+.16e}j" for value in row
            )
            + "]\n"
        )
    stream.flush()


def _open_outputs(stack: ExitStack):
    """Open all output files and return named streams."""
    filenames = {
        "population": "output.dat",
        "rdo": "rdoout.dat",
        "trace": "test_trace.dat",
        "left_total": "L_total_curr.dat",
        "right_total": "R_total_curr.dat",
        "current_total": "total_curr.dat",
        "continuity": "continuity.dat",
    }
    for m in range(pa.nvarf):
        for s in range(pa.nspinf):
            filenames[f"n_m{m}_s{s}"]=f"n_m{m}_s{s}.dat"
            for a in range(pa.nalphaf):
                filenames[f"I_a{a}_m{m}_s{s}"]=f"I_a{a}_m{m}_s{s}.dat"
    streams = {
        name: stack.enter_context(open(filename, "w", encoding="utf-8"))
        for name, filename in filenames.items()
    }

    streams["population"].write(
        "# time  "
        + "  ".join(f"rho_diag_{i}" for i in range(pa.ndvr_dense))
        + "\n"
    )
    streams["rdo"].write(
        "# dense basis: |(0,up),(0,down),(1,up),(1,down),...>; matrix dimension "
        f"{pa.ndvr_dense} x {pa.ndvr_dense}\n"
    )
    for name in filenames:
        if name not in ("population","rdo","continuity"):
            streams[name].write("# time  real  imag\n")
    streams["continuity"].write(
        "# time  dN_system_dt  sum_lead_currents  residual_real  residual_imag\n"
    )
    return streams


def _write_snapshot(
    streams,
    snapshot: Snapshot,
    step: int,
    previous_snapshot: Snapshot = None,
) -> None:
    _write_real_diagonal(streams["population"], snapshot)
    _write_rdo(streams["rdo"], step, snapshot)

    _write_complex_value(streams["trace"],snapshot.time,snapshot.observables["trace"])
    for m in range(pa.nvarf):
        for s in range(pa.nspinf):
            key=f"n_m{m}_s{s}"
            _write_complex_value(streams[key],snapshot.time,snapshot.observables[key])

    lead_spin = snapshot.currents.by_lead_spin()
    lead_total = snapshot.currents.by_lead()
    current_total = snapshot.currents.total()

    for m in range(pa.nvarf):
        for s in range(pa.nspinf):
            for a in range(pa.nalphaf):
                _write_complex_value(streams[f"I_a{a}_m{m}_s{s}"],snapshot.time,
                                     snapshot.currents.resolved[(a,s,m)])
    _write_complex_value(
        streams["left_total"], snapshot.time, lead_total[pa.LEAD_LEFT]
    )
    _write_complex_value(
        streams["right_total"], snapshot.time, lead_total[pa.LEAD_RIGHT]
    )
    _write_complex_value(streams["current_total"], snapshot.time, current_total)

    if previous_snapshot is not None:
        delta_time = snapshot.time - previous_snapshot.time
        if delta_time <= 0.0:
            raise ValueError(f"non-positive snapshot interval {delta_time}")
        number_now = snapshot.observables["number_total"]
        number_previous = previous_snapshot.observables["number_total"]
        dnumber_dt = (number_now - number_previous) / delta_time
        residual = dnumber_dt - current_total
        streams["continuity"].write(
            f"{snapshot.time:.16e}  "
            f"{dnumber_dt.real:.16e}{dnumber_dt.imag:+.16e}j  "
            f"{current_total.real:.16e}{current_total.imag:+.16e}j  "
            f"{residual.real:.16e}  {residual.imag:.16e}\n"
        )
        streams["continuity"].flush()


def prop(rin, pall):
    """Propagate an MPS with the existing one-site projector-splitting TDVP."""
    with ExitStack() as stack:
        streams = _open_outputs(stack)

        previous = _calculate_snapshot(rin, time=0.0)
        _write_snapshot(streams, previous, step=0)
        print("dimension of initial MPS =", rin.ndim())
        print("initial trace =", previous.observables["trace"])
        print("initial lead-spin currents =", previous.currents.by_lead_spin())

        for step in range(1, pa.nsteps + 1):
            print(f"istep = {step}/{pa.nsteps}")
            rin = ksl.ksltt(rin, pall)
            current = _calculate_snapshot(rin, time=step * pa.dt)
            _write_snapshot(streams, current, step=step, previous_snapshot=previous)

            if step == 1 or step % 10 == 0 or step == pa.nsteps:
                particle_number = current.observables["number_total"]
                print(
                    "time =", current.time,
                    "trace =", current.observables["trace"],
                    "N =", particle_number,
                    "I =", current.currents.total(),
                )
            previous = current

    return rin


if __name__ == "__main__":
    raise SystemExit(
        "prop.py is a propagation module. Run main.py, or call prop(rho, mpo)."
    )
