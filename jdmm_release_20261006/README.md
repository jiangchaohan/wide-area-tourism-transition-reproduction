# Xinjiang tourism route-product data and reproducibility materials

Versioned materials for the revised manuscript, **Context-dependent attraction connectivity in destination route products: Evidence from Xinjiang, China**, prepared on 6 October 2026. The author approved the publication checklist before this update. Root-level materials remain a legacy archive and must not be used as the current revised results.

The S2 processed workbook contains 33,269 route records. Analysis excludes three January 1970 records, retaining 33,266 routes, 1,034 attraction states and 736,883 successor events. These administrative route products are not observed individual tourist trajectories. The unused `route_title` column has been replaced throughout by `OMITTED`; modeling fields are preserved. S3 is unchanged. No original administrative export is included.

The data provider authorized public release of these processed tables. No additional open reuse license, ethics exemption or author-approval certification is asserted. Review residual disclosure risks and the old repository's raw files and history before release; this proposal does not certify the old archive as privacy-screened.

## Reproduction

Use Python 3.12 and install `requirements.txt`. PyTorch 2.7.1 with a suitable CUDA 12.6 wheel or CPU build was used. From this directory:

```text
python code/strengthening/prepare_strengthening.py
python code/strengthening/statistical_strengthening.py
python code/strengthening/neural_strengthening.py
python code/strengthening/plot_strengthened.py
```

Preparation creates an intentionally omitted cache from the workbooks. Preserve supplied outputs before rerunning. Results include deterministic counts, annual and filtered sensitivities, paired cluster intervals, validation-selected local GRU/causal-transformer trials over three seeds, learning curves, checkpoints and predictions. These are local implementations, not official benchmark replications. `code/run_route_experiments.py` is included because the strengthening script imports its historical tie-convention helpers; historical five-epoch results are not substituted for the revised neural comparisons. The protocol and supplementary experiment report document the analysis boundaries.

`SHA256_MANIFEST.json` records the release files for comparison against the submission supplement. A hash manifest verifies file identity, not scientific correctness or a privacy guarantee. The public README has been updated from the reviewed local proposal; data, code and result files are unchanged. No legacy-file deletion, Git-history rewrite or repository license change is included in this update.
