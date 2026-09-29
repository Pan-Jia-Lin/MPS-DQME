# MPS-DQME

**Matrix-Product-State Dissipaton-Embedded Quantum Master Equation**

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

## Software versions

The two implementations retain their original internal version records:

- Anderson-Holstein: `code/anderson_holstein/VERSION`
- Multi-orbital AIM: `code/multiorbital_aim/VERSION`

The public repository uses Git tags and GitHub Releases (for example,
`v1.0.0`) to identify the exact snapshot associated with the manuscript.
