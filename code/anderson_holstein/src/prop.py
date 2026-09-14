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
        "n_up": "n1_ave.dat",
        "n_down": "n2_ave.dat",
        "x": "x_ave.dat",
        "p": "p_ave.dat",
        "x2": "x2_ave.dat",
        "p2": "p2_ave.dat",
        "phonon_number": "phonon_number_ave.dat",
        "symmetrized_xp": "xp_px_ave.dat",
        "trace": "test_trace.dat",
        "left_up": "L_up_curr.dat",
        "left_down": "L_down_curr.dat",
        "right_up": "R_up_curr.dat",
        "right_down": "R_down_curr.dat",
        "left_total": "L_total_curr.dat",
        "right_total": "R_total_curr.dat",
        "current_total": "total_curr.dat",
        "continuity": "continuity.dat",
    }
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
        "# dense basis: |n_up,n_down,n_phonon>; matrix dimension "
        f"{pa.ndvr_dense} x {pa.ndvr_dense}\n"
    )
    for name in (
        "n_up",
        "n_down",
        "x",
        "p",
        "x2",
        "p2",
        "phonon_number",
        "symmetrized_xp",
        "trace",
        "left_up",
        "left_down",
        "right_up",
        "right_down",
        "left_total",
        "right_total",
        "current_total",
    ):
        streams[name].write("# time  real  imag\n")
    streams["continuity"].write(
        "# time  dN_system_dt  sum_lead_currents  residual_real  residual_imag\n"
    )
    return streams


def _write_snapshot(
    streams,
    snapshot: Snapshot,
    step: int,
) -> None:
    _write_real_diagonal(streams["population"], snapshot)
    _write_rdo(streams["rdo"], step, snapshot)

    for name in (
        "n_up",
        "n_down",
        "x",
        "p",
        "x2",
        "p2",
        "phonon_number",
        "symmetrized_xp",
        "trace",
    ):
        _write_complex_value(
            streams[name], snapshot.time, snapshot.observables[name]
        )

    lead_spin = snapshot.currents.by_lead_spin()
    lead_total = snapshot.currents.by_lead()
    current_total = snapshot.currents.total()

    _write_complex_value(
        streams["left_up"],
        snapshot.time,
        lead_spin[pa.LEAD_LEFT, pa.SPIN_UP],
    )
    _write_complex_value(
        streams["left_down"],
        snapshot.time,
        lead_spin[pa.LEAD_LEFT, pa.SPIN_DOWN],
    )
    _write_complex_value(
        streams["right_up"],
        snapshot.time,
        lead_spin[pa.LEAD_RIGHT, pa.SPIN_UP],
    )
    _write_complex_value(
        streams["right_down"],
        snapshot.time,
        lead_spin[pa.LEAD_RIGHT, pa.SPIN_DOWN],
    )
    _write_complex_value(
        streams["left_total"], snapshot.time, lead_total[pa.LEAD_LEFT]
    )
    _write_complex_value(
        streams["right_total"], snapshot.time, lead_total[pa.LEAD_RIGHT]
    )
    _write_complex_value(streams["current_total"], snapshot.time, current_total)



def _write_centered_continuity(
    stream: TextIO,
    left: Snapshot,
    center: Snapshot,
    right: Snapshot,
) -> None:
    """Write the centered-difference particle-continuity residual at center."""
    delta_time = right.time - left.time
    if delta_time <= 0.0 or not (left.time < center.time < right.time):
        raise ValueError(
            "continuity snapshots must have strictly increasing times"
        )
    number_left = left.observables["n_up"] + left.observables["n_down"]
    number_right = right.observables["n_up"] + right.observables["n_down"]
    dnumber_dt = (number_right - number_left) / delta_time
    current_total = center.currents.total()
    residual = dnumber_dt - current_total
    stream.write(
        f"{center.time:.16e}  "
        f"{dnumber_dt.real:.16e}{dnumber_dt.imag:+.16e}j  "
        f"{current_total.real:.16e}{current_total.imag:+.16e}j  "
        f"{residual.real:.16e}  {residual.imag:.16e}\n"
    )
    stream.flush()


def prop(rin, pall):
    """Propagate an MPS with the existing one-site projector-splitting TDVP."""
    with ExitStack() as stack:
        streams = _open_outputs(stack)

        previous_previous = None
        previous = _calculate_snapshot(rin, time=0.0)
        _write_snapshot(streams, previous, step=0)
        print("dimension of initial MPS =", rin.ndim())
        print("initial trace =", previous.observables["trace"])
        print("initial lead-spin currents =", previous.currents.by_lead_spin())

        for step in range(1, pa.nsteps + 1):
            print(f"istep = {step}/{pa.nsteps}")
            rin = ksl.ksltt(rin, pall)
            current = _calculate_snapshot(rin, time=step * pa.dt)
            _write_snapshot(streams, current, step=step)

            if previous_previous is not None:
                _write_centered_continuity(
                    streams["continuity"],
                    previous_previous,
                    previous,
                    current,
                )

            if step == 1 or step % 10 == 0 or step == pa.nsteps:
                particle_number = (
                    current.observables["n_up"]
                    + current.observables["n_down"]
                )
                print(
                    "time =", current.time,
                    "trace =", current.observables["trace"],
                    "N =", particle_number,
                    "I =", current.currents.total(),
                )
            previous_previous, previous = previous, current

    return rin


if __name__ == "__main__":
    raise SystemExit(
        "prop.py is a propagation module. Run main.py, or call prop(rho, mpo)."
    )
