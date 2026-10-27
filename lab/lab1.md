# Lab 1 — Verification and storage choice

Questions 1–8 and the original setup are answered in `../lab-answers.md`.

The storage workaround adopted is the subject's recommended **local DVC remote**. The full dataset is retained; no reduced sample replaces the raw data. Storage lives in `../dvc-storage`, outside the Git repository. The machine-specific setting is private in `.dvc/config.local`; the original shared DagsHub configuration stays available.

```powershell
dvc remote add --local lab-local ../dvc-storage
dvc push -r lab-local --all-commits
```

Another clone on this machine must configure the absolute path or its own relative path to that folder, then use `dvc pull -r lab-local`. An off-machine clone requires copying that storage or completing a shared remote upload; a local remote does not make files available over the internet.

A clean clone initially had no `data/` directory. `dvc pull -r lab-local` fetched 32,034 content objects and reconstructed all 36,578 original files. Switching to the raw-only commit required fetching its historical directory manifest once, since pulling the latest commit alone does not populate every historical manifest. The raw-only checkout retained all 16,643 raw files and removed both processed folders. Switching back to `main` restores the processed snapshot using `dvc checkout` from cache. On a machine without cached historical objects, use `dvc pull -r lab-local` after switching commits.

Lab 5's contract exposed 611 duplicate processed images across the supplied splits. The raw 16,643-image snapshot is preserved. The corrected generated dataset contains 16,032 full processed images (9,866 training, 3,098 evaluation, 3,068 validation) and 3,292 mini images (1,100/1,096/1,096), with no cross-split byte-identical images. Mini selection refills classes from the remaining unique files and retains the at-most-100 rule. This produces a new DVC pointer, preserving the original snapshots in Git/DVC history.
