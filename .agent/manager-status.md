# Agent Manager Status

## Session History
- **2025-12-19:** Initialized agent management framework from paper draft "kSZ_DESI_LRG_Y3_x_ACT_DR6". Created `project-context.md` with paper overview, methodology, and outstanding TODOs.

## Current Milestone
Paper finalization - addressing reviewer comments and internal TODOs marked in the draft.

## Outstanding Tasks

### Figures
- [x] Fig 4: ~~Check units, extend galaxy dots, add legend~~ **DONE** - Replaced density field with scatter plot of individual galaxy positions
- [ ] Fig 5: Consider reducing to rho/x/beta only; add HOD variation plot
- [ ] Fig 7: Add PTE values for best-fit and Battaglia curves
- [ ] Fig 9: Add combined sample null test; consider 90-150 or y-map null
- [ ] Fig 10: Clarify colorbar as noise description
- [ ] Fig 12: Reduce horizontal data point offset
- [ ] Fig 13: Specify fiducial; add r(z) degeneracy caveat

### Tables
- [ ] Table 1: Specify HOD source
- [ ] Table 2: Specify HOD source for marginalization

### Text
- [ ] Section 4.2: Describe HOD used and add citations
- [ ] Section 5.0.2: Clarify mass dependence caption
- [ ] Section 5.1.2: Clarify imperfect photo-z consistency
- [ ] Section 6 (Conclusion): Check SNR scaling claim; rephrase future work

### Analysis
- [ ] Verify SNR improvement factor (sqrt(N) vs N scaling)
- [ ] Configuration space vs Fourier space consistency check (if CAP modeling code available)

## Completed Tasks
- Initialized `.agent/project-context.md` with comprehensive paper summary
- Initialized `.agent/manager-status.md` (this file)
- **Fig 1:** Fixed alpha transparency in footprint map (`LRG_Y3.ipynb`, `create_full_footprint_map_custom_alpha`). Implemented Porter-Duff alpha compositing for NGC/SGC/ACT overlap regions.
- **Fig 4 (2025-12-25):** Replaced pixelized density field with scatter plot of individual galaxy positions (`plot_no_vel.py`). Shows RA 120-160°, Dec 0-12.5° with ~128k galaxies as filled circles (s=1, alpha=0.5, edgecolors='none'). Updated caption to describe cosmic web structure visible in galaxy distribution.
- **Mask robustness test (2026-01-08):** Created `survey_mask_local_nbar.py` implementing equation 8 local number density mask. Outputs α_r × random counts per pixel instead of binary Heaviside. Commit c0101de.

## Notes
- Paper is in late-stage draft with internal comments (ES:, FJQ:) marking items to address
- Key scientific results (18-sigma detection, GNFW constraints) appear finalized
- Main remaining work is editorial: figure polish, caption clarification, cross-referencing
