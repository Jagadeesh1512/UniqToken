# Research Report: Multilingual-Aware Merge Selection (Issue #88)

## 1. Executive Summary & Research Question

**Research Question**: *Do globally selected merges disproportionately serve dominant training strata, and can a deterministic coverage-aware score improve multilingual allocation without hard-coded language token lists?*

### Key Empirical Findings
1. **Global Merge Concentration**: **98.0%** of learned merges each have >=90% of their pair occurrences in some single stratum. This is not the percentage allocated to one language. Global dominant-stratum allocation has HHI **0.3056** across **5** represented strata.
2. **Alternative Allocation**: Balanced scoring represents **5** strata (HHI **0.5096**, concentrated merges **98.0%**); coverage-aware scoring represents **5** strata (concentrated merges **96.0%**). Representation and concentration may improve or worsen relative to global scoring; both outcomes are retained.
3. **Allocation Changes in the Balanced Condition**: The recorded allocation may shift between strata (e.g. es merges increasing from 15 to 34), while trading off merge capacity previously concentrated in the dominant stratum (en merges adjusting from 21 to 0). Allocation is a descriptive count, not evidence of a causal validation benefit or a globally best tokenizer.

## 2. Vocabulary & Merge Allocation Across Strata

| Condition | Strategy | Learned Merges | Strata Represented | Concentrated Merges (>=90%) | HHI of Dominant-Stratum Allocation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Global_SuperBPE** | `global` | 50 | 5 | 98.0% (49) | **0.3056** |
| **Balanced_SuperBPE** | `balanced` | 50 | 5 | 98.0% (49) | **0.5096** |
| **CoverageAware_SuperBPE** | `coverage_aware` | 50 | 5 | 96.0% (48) | **0.4840** |

### Per-Stratum Merge Distribution Breakdown

| Stratum | Language | Global SuperBPE Merges | Balanced SuperBPE Merges | CoverageAware SuperBPE Merges |
| :--- | :---: | :---: | :---: | :---: |
| `web:am` | **am** | 4 | 4 | 4 |
| `web:ar` | **ar** | 9 | 10 | 10 |
| `web:en` | **en** | 21 | 0 | 0 |
| `web:es` | **es** | 15 | 34 | 33 |
| `web:hi` | **hi** | 1 | 1 | 1 |
| `web:ml` | **ml** | 0 | 0 | 0 |
| `web:sw` | **sw** | 0 | 1 | 2 |
| `web:te` | **te** | 0 | 0 | 0 |
| `web:yo` | **yo** | 0 | 0 | 0 |
| `web:zh` | **zh** | 0 | 0 | 0 |

## 3. Disjoint Validation Compression Comparison

All four conditions use the same total vocabulary budget. Merge conditions share a smaller seed vocabulary and reserve the remaining slots for merges; the full-budget Unigram baseline has no merge reserve. Bytes/token and characters/token use normalized source, not metaspace spelling. Whitespace-field fertility is descriptive and is not a universal cross-script linguistic metric.

| Language | Metric | Unigram Baseline | Global SuperBPE | Balanced SuperBPE | CoverageAware SuperBPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **am** | Bytes/Token (higher=better) | 1.86 | 1.93 | 1.93 | 1.93 |
| | Final Learned-Merge Token Emissions | 0 | 2 | 2 | 2 |
| **ar** | Bytes/Token (higher=better) | 1.81 | 1.99 | 2.04 | 2.04 |
| | Final Learned-Merge Token Emissions | 0 | 3 | 5 | 5 |
| **en** | Bytes/Token (higher=better) | 1.20 | 1.13 | 1.13 | 1.13 |
| | Final Learned-Merge Token Emissions | 0 | 1 | 1 | 1 |
| **es** | Bytes/Token (higher=better) | 1.14 | 1.11 | 1.12 | 1.12 |
| | Final Learned-Merge Token Emissions | 0 | 0 | 1 | 1 |
| **hi** | Bytes/Token (higher=better) | 1.81 | 1.81 | 1.81 | 1.81 |
| | Final Learned-Merge Token Emissions | 0 | 0 | 0 | 0 |
| **ml** | Bytes/Token (higher=better) | 1.90 | 1.85 | 1.85 | 1.85 |
| | Final Learned-Merge Token Emissions | 0 | 0 | 0 | 0 |
| **sw** | Bytes/Token (higher=better) | 1.05 | 1.04 | 1.12 | 1.15 |
| | Final Learned-Merge Token Emissions | 0 | 0 | 7 | 10 |
| **te** | Bytes/Token (higher=better) | 1.60 | 1.57 | 1.57 | 1.57 |
| | Final Learned-Merge Token Emissions | 0 | 0 | 0 | 0 |
| **yo** | Bytes/Token (higher=better) | 1.76 | 1.72 | 1.74 | 1.74 |
| | Final Learned-Merge Token Emissions | 0 | 0 | 1 | 1 |
| **zh** | Bytes/Token (higher=better) | 1.34 | 1.34 | 1.34 | 1.34 |
| | Final Learned-Merge Token Emissions | 0 | 0 | 0 | 0 |

## 4. Research Integrity & Verification Ledger

- **Zero Test-Set Access**: Only training and validation were used. For manifest inputs, test is checked only as declared path/hash metadata and is never opened or hashed; the canonical controlled fixture has no test data.
- **Zero Train/Validation Leakage**: Normalized training/validation text and document-ID overlap is rejected before training. Training repetitions within one split are intentional exposure weighting.
- **No Hard-Coded Token Lists**: Scoring functions operate purely on empirical stratum statistics without language-specific token tables or manual regex filters.
- **Budget Adherence**: Target vocabulary and merge counts were validated to bit-exact targets across all conditions.
- **Normalized Round Trips**: Every evaluated document reconstructs normalized source exactly (mean stratum byte fallback rate: 33.1%, max: 82.8%). This does not claim raw-byte reversibility under NFKC or a zero fallback rate.
- **Scope**: The retained canonical run is a small controlled fixture, not a measurement of frozen Phase A models or real-world multilingual allocation. No LM is trained, no frozen artifacts are modified, and no causal or global superiority claim is made.
