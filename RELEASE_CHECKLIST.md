# Public-release checklist

Complete these items before making the GitHub repository public:

- [x] Add the final repository URL to `README.md` and `CITATION.cff`:
      `https://github.com/Pan-Jia-Lin/MPS-DQME`.
- [ ] Replace `https://github.com/xxx` in the manuscript's Data Availability
      section with the final repository URL.
- [x] Confirm the manuscript author list, author order, affiliations, and
      corresponding-author email in `CITATION.cff` and `README.md`.
- [ ] Obtain institutional/group approval for a software license and add `LICENSE`.
- [x] Align the figure-directory descriptions in
      `paper_support_data/README.md` with manuscript Figures 2-11.
- [x] Document the retained/adapted Qiang Shi group HEOM+MPS utilities and
      preserve their original per-file author/reference headers.
- [ ] Confirm that `R_up_curr_part3.dat` is the intended corrected spelling of
      the original `R_up_curr__part3.dat`.
- [ ] Run both validation commands documented in the root `README.md`.
- [ ] Review the repository for unpublished notes, credentials, absolute paths,
      or unintended large generated files.
- [ ] Create and push a frozen Git tag (recommended: `v1.0.0`).
- [ ] Create a GitHub Release from that tag and archive it with Zenodo if a
      software DOI is desired.
- [ ] After acceptance, add the article DOI and full bibliographic metadata to
      `CITATION.cff` as `preferred-citation`.
