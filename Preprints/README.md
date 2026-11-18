# Preprints for Urban-Grid (`crvscience-urban-grid`)

**Author:** Bryan P. White (ORCID: [0000-0003-4191-3511](https://orcid.org/0000-0003-4191-3511))  
**Hub:** https://crvscience.com/apps

This directory contains the self-contained LaTeX source (`.tex`) and single-column PDF (`.pdf`) editions for both:

1. **Companion Application Preprint (`APP-PREPRINT-11`)**
   - **Title:** Urban-Grid: Modeling City Traffic, Emergency Response Coverage, and Earthquake Evacuation
   - **Files:** `urban-grid-microscopic-traffic-resilience.tex`, `urban-grid-microscopic-traffic-resilience.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222394
   - **Figshare Mirror:** https://figshare.com/articles/preprint/Urban-Grid_Modeling_City_Traffic_Emergency_Response_Coverage_and_Earthquake_Evacuation/35013772
   - **Web Preprint:** https://crvscience.com/apps/preprints/urban-grid-microscopic-traffic-resilience

2. **CRV Science Public Apps Network Overview (`APP-PREPRINT-01`)**
   - **Title:** The CRV Science App Network: Eighteen Open-Source Rust Programs for Science, Data, and Simulation
   - **Files:** `crv-science-public-apps-network.tex`, `crv-science-public-apps-network.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222398
   - **Figshare Mirror:** https://figshare.com/articles/preprint/The_CRV_Science_App_Network_Seventeen_Open-Source_Rust_Programs_for_Science_Data_and_Simulation/35013736
   - **Web Preprint:** https://crvscience.com/apps/preprints/crv-science-public-apps-network

## Compiling Standalone LaTeX

Both `.tex` files inline their bibliography so each compiles standalone with standard `pdflatex`:

```bash
pdflatex urban-grid-microscopic-traffic-resilience.tex
pdflatex crv-science-public-apps-network.tex
```
