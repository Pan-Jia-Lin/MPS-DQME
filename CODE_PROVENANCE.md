# Code provenance and attribution

This repository combines scripts developed or substantially modified for the
MPS-DQME implementation with MPS/TDVP utility scripts retained from earlier
HEOM+MPS work.

## Retained or adapted HEOM+MPS utilities

The files listed below contain an original header identifying the HEOM+MPS
code authors as **Qiang Shi, Meng Xu, and Xiaohan Dan (ICCAS)** and citing:

> Q. Shi, Y. Xu, Y.-M. Yan, and M. Xu, "Efficient propagation of the
> hierarchical equations of motion using the matrix product state method,"
> *The Journal of Chemical Physics* **148**, 174102 (2018).

The headers have been preserved as the authoritative per-file attribution.
These utilities originate from the HEOM+MPS/TDVP implementation associated
with Prof. Qiang Shi's group. Where integration into MPS-DQME required local
changes, the resulting file should be understood as an adapted version rather
than an unmodified upstream copy.

### Anderson-Holstein implementation

- `code/anderson_holstein/src/add_tensor.py`
- `code/anderson_holstein/src/corrfun.py`
- `code/anderson_holstein/src/ksltt.py`
- `code/anderson_holstein/src/print_tensor.py`
- `code/anderson_holstein/src/split.py`
- `code/anderson_holstein/src/trun.py`
- `code/anderson_holstein/src/ttfunc.py`

### Multi-orbital AIM implementation

- `code/multiorbital_aim/add_tensor.py`
- `code/multiorbital_aim/ksltt.py`
- `code/multiorbital_aim/print_tensor.py`
- `code/multiorbital_aim/split.py`
- `code/multiorbital_aim/trun.py`
- `code/multiorbital_aim/ttfunc.py`

## MPS-DQME-specific development

The remaining source files implement or support the DQME construction,
fermionic and bosonic dissipaton layouts, observables, model parameters,
validation, checkpoint handling, and release workflows used in this project.
Several of these files were developed by modifying earlier project scripts.
This repository-level statement does not override any attribution or license
notice present in an individual source file.

## Manuscript acknowledgment

The accompanying manuscript thanks Qiang Shi, Xiaohan Dan, Yu Su, Zi-Fan Zhu,
Long Cao, and Liwei Ge for constructive discussions and software support.
