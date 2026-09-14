# MPS-DQME

**Matrix-Product-State Dissipaton-Embedded Quantum Master Equation**

This repository contains the MPS-DQME implementations and numerical data
associated with the manuscript:

> Jia-Lin Pan, Hao Zhang, Xiao Zheng, YiJing Yan, and Yao Wang,
> "Matrix product state formulation of the dissipaton-embedded quantum master
> equation for open quantum system dynamics."

MPS-DQME embeds the dissipaton
degrees of freedom into a single matrix-product-state representation for the
non-Markovian dynamics of open quantum systems.

Repository: <https://github.com/Pan-Jia-Lin/MPS-DQME>

## Repository contents

| Path | Description |
| --- | --- |
| `code/anderson_holstein/` | Spinful Anderson-Holstein implementation with fermionic reservoirs and a bosonic environment |
| `code/multiorbital_aim/` | Purely fermionic implementation for the multi-orbital Anderson impurity model |
| `paper_support_data/` | Numerical data supporting the figures in the manuscript |
| `CODE_PROVENANCE.md` | Provenance and attribution for retained and modified scripts |

Each code directory is self-contained and has its own `README.md`, dependency
list, parameters, validation instructions, and model-specific conventions.

## Quick validation

### Anderson-Holstein implementation

```bash
cd code/anderson_holstein
python -m pip install -r requirements.txt
python scripts/validate_setup.py
```

### Multi-orbital AIM implementation

```bash
cd code/multiorbital_aim
python -m pip install -r requirements.txt
python validate_hsys.py
```

These are lightweight integrity checks. They do not establish convergence of
the production calculations. See the model-specific READMEs before running
long-time or high-rank simulations.

## Reproducing the manuscript results

The raw numerical results are organized by figure in
`paper_support_data/`. See `paper_support_data/README.md` for the directory
map, file conventions, and interpretation of the most common output names.

The supplied data are the archived results used for plotting. A full
recalculation can be computationally expensive and may require large MPS ranks,
long propagation times, and convergence checks with respect to timestep,
dissipaton cutoff, and bath-decomposition size.

## Software versions

The two implementations retain their original internal version records:

- Anderson-Holstein: `code/anderson_holstein/VERSION`
- Multi-orbital AIM: `code/multiorbital_aim/VERSION`

The public repository uses Git tags and GitHub Releases (for example,
`v1.0.0`) to identify the exact snapshot associated with the manuscript.

## Citation

If you use this software or its supporting data, please cite the accompanying
publication and the archived software release. The manuscript title, author
order, and affiliations are recorded in `CITATION.cff`. Add the final GitHub
URL and article DOI when they become available.

## License

No license has been selected in this prepared package because software
ownership and reuse terms should be confirmed with the authors' institution
and research group. Add an approved `LICENSE` file before making the repository
public. BSD-3-Clause and MIT are common permissive choices for research
software, but this repository does not presume that choice.

## Authors and affiliations

- Jia-Lin Pan - State Key Laboratory of Precision and Intelligent Chemistry
  and Department of Chemical Physics, University of Science and Technology of
  China, Hefei, Anhui 230026, China
- Hao Zhang - School of Information Science and Technology and National
  Engineering Laboratory for Brain-inspired Intelligence Technology and
  Application, University of Science and Technology of China, Hefei, Anhui
  230026, China
- Xiao Zheng - Department of Chemistry, Fudan University, Shanghai 200433,
  China; and Hefei National Laboratory, Hefei, Anhui 230088, China
- YiJing Yan - Hefei National Research Center for Physical Sciences at the
  Microscale, University of Science and Technology of China, Hefei, Anhui
  230026, China
- Yao Wang - Hefei National Research Center for Physical Sciences at the
  Microscale, University of Science and Technology of China, Hefei, Anhui
  230026, China

Corresponding author: Yao Wang (`wy2010@ustc.edu.cn`).

## Code provenance and acknowledgments

This repository combines MPS-DQME-specific development with retained and
adapted MPS/TDVP utilities from the HEOM+MPS code associated with Qiang Shi and
co-workers. The original source comments are preserved in the relevant files.
See `CODE_PROVENANCE.md` for the exact file lists and reference.

The manuscript acknowledges Qiang Shi, Xiaohan Dan, Yu Su, Zi-Fan Zhu, Long
Cao, and Liwei Ge for constructive discussions and software support.
