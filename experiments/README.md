# Experiment Records

Every scientific run receives a directory named:

`YYYYMMDD-HHMM-short-hypothesis`

Each directory must contain `record.md` with:

1. Hypothesis
2. Source commit
3. Configuration and random seed
4. Training and calibration split
5. Features and model
6. Local low-FPR results
7. Pseudo-OoD results
8. Runtime and hardware
9. Artifact and submission checksums
10. Conclusion and next decision

A Codabench submission must reference exactly one experiment directory.
