---
title: "SSFL Paper Reproduction — Semisupervised Federated-Learning IDS on N-BaIoT"
status: draft
version: "1.0"
---

# Product Requirements Document

## Validation Checklist

### CRITICAL GATES (Must Pass)

- [x] All required sections are complete
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Problem statement is specific and measurable
- [x] Every feature has testable acceptance criteria (Gherkin format)
- [x] No contradictions between sections

### QUALITY CHECKS (Should Pass)

- [x] Problem is validated by evidence (not assumptions)
- [x] Context → Problem → Solution flow makes sense
- [x] Every persona has at least one user journey
- [x] All MoSCoW categories addressed (Must/Should/Could/Won't)
- [x] Every metric has corresponding tracking events
- [x] No technical implementation details included
- [x] A new team member could understand this PRD
- [x] **MECE: Personas** — each persona is distinct, all user types represented
- [x] **MECE: Journeys** — each journey is a unique path, all paths covered
- [x] **MECE: Features** — no overlapping user stories, no capability gaps
- [x] **MECE: Acceptance Criteria** — each criterion tests a unique condition, all paths covered

---

## Product Overview

### Vision
A faithful, runnable reproduction of Zhao et al., *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things"* (IEEE IoT-J 10(10), 2023), regenerating every result artifact of the paper (Tables II–IV, Figures 3–6) from the raw N-BaIoT data with one command per experiment.

### Problem Statement
The paper reports strong results (SSFL: 87.40% / 86.70% / 84.22% accuracy across three non-IID scenarios) but publishes no code, no seeds, and leaves several implementation details implicit. Without a reproduction, the researcher cannot verify the claims, cannot build on the method, and cannot compare future variants against a trusted baseline. The consequence of not solving this: any follow-up research on SSFL/pseudo-labeling rests on unverified numbers.

### Value Proposition
Unlike ad-hoc reimplementations, this reproduction (a) covers the *full* experiment suite including all ablations, (b) regenerates the paper's tables/figures programmatically so paper-vs-ours deltas are visible at a glance, and (c) yields a reusable federated-simulation codebase for follow-up research.

## User Personas

### Primary Persona: The Researcher
- **Demographics:** Graduate-level ML/security researcher (the project owner), comfortable with Python/PyTorch, working on a MacBook (Apple M4, 16 GB) with optional access to cloud CUDA machines.
- **Goals:** Verify the paper's claims end-to-end; obtain a trusted SSFL baseline and codebase for future extensions (e.g., alternative pseudo-labeling strategies).
- **Pain Points:** No reference code or seeds; long-running experiments (~30 runs, hours each) that must not be lost to crashes; a 7.6 GB raw dataset that is slow to handle naively; ambiguous paper details that must be resolved by documented judgment calls.

### Secondary Personas
None — single-user research project. (Future readers of the code are served by the same artifacts the primary persona needs.)

### MECE Check: Personas
- [x] Each persona has distinct goals and pain points (no overlap)
- [x] All user types who interact with this feature are represented (no gaps)

## User Journey Maps

### Primary User Journey: Full reproduction campaign
1. **Awareness:** Researcher decides to reproduce the paper before extending it.
2. **Consideration:** Alternatives — trust the paper, partially reproduce (SSFL only), or reimplement from scratch without structure. Criteria: fidelity, total compute time, reusability.
3. **Adoption:** One-time setup: create the environment, build the cached mini-N-BaIoT dataset once from raw CSVs.
4. **Usage:** Run a timing pilot (SSFL, Scenario 1) → decide local vs. cloud → execute the ~30-run suite (methods × scenarios, then ablations) → generate the report comparing every number/figure to the paper.
5. **Retention:** The codebase becomes the baseline harness for follow-up experiments (new methods drop in as additional strategies).

### Secondary User Journeys

#### Single-experiment iteration (debug/develop)
1. Researcher modifies or inspects one method.
2. Runs the smoke test suite (minutes) to catch breakage.
3. Runs a single short experiment (reduced rounds) to sanity-check learning curves.
4. Promotes to a full 200-round run only when the short run looks sane.

#### Interrupted-run recovery (error path)
1. A multi-hour run dies (crash, sleep, OOM) partway through.
2. Researcher restarts the run; completed per-round metrics up to the failure point are not lost.
3. The run either resumes or restarts cleanly with the same seed, and the results directory never ends up in a corrupt half-written state that poisons the report.

### MECE Check: Journeys
- [x] Each journey describes a distinct path (no two journeys cover the same actions for the same persona)
- [x] All primary, secondary, and error/recovery paths are mapped (no gaps)
- [x] Every persona has at least one journey

## Feature Requirements

### Must Have Features

#### Feature 1: Mini-N-BaIoT dataset builder
- **User Story:** As the researcher, I want the paper's mini-N-BaIoT dataset built once from the raw CSVs and cached, so that every experiment starts from identical, fast-loading data.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given the raw `data/*.csv` files (89 device-category files), When the builder runs, Then it produces a cached dataset with 1000 samples per device-category subset (89 subsets), min-max normalized to [0,1], and each sample reshaped to the paper's 23×5 layout.
  - [ ] Given the cached dataset, When splits are inspected, Then private/open/test are 70%/10%/20% per subset, mutually disjoint, and the open split carries no labels into training.
  - [ ] Given a fixed seed, When the builder runs twice, Then both runs produce byte-identical splits and partitions.
  - [ ] Given Scenario 1/2/3 is requested, When partitioning runs, Then client counts are 27/89/89 respectively, Scenario 1 uses label-sorted shards (2 per client), Scenario 2 gives each client one class, and Scenario 3 uses a Dirichlet(α=0.1) allocation.
  - [ ] Given devices 3 and 7 (no mirai traffic), When the dataset is built, Then their subsets contain exactly 6 classes and no run crashes on the missing categories.

#### Feature 2: Model zoo per the paper
- **User Story:** As the researcher, I want the paper's CNN (classifier and discriminator variants), MLP, and LSTM models available, so that all Table II rows can be produced.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given a 23×5 input batch, When the CNN classifier runs forward, Then layer output shapes match Table I exactly and the head has 11 outputs (discriminator variant: 2 outputs).
  - [ ] Given any model and any available device (cpu/mps/cuda), When one training step executes, Then it completes without device-specific errors.

#### Feature 3: Four federated training methods
- **User Story:** As the researcher, I want FL (FedAvg), FD, DS-FL, and SSFL runnable under one federated simulation harness, so that the paper's method comparison is apples-to-apples.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given any of the four methods and any scenario, When a run executes, Then all clients train per the method's protocol and the server aggregates per the paper's equations (Eq. 1 FL; Eqs. 2–4 FD; Eqs. 5–10 DS-FL; Eqs. 11–18 SSFL).
  - [ ] Given an SSFL run, When a round completes, Then each client has: trained its classifier on private data, trained its discriminator using the median-confidence threshold, uploaded hard labels with unfamiliar samples excluded, and distilled on the open set with the server's majority-voted global labels.
  - [ ] Given an SSFL run, When evaluation happens, Then the evaluated model is the server classifier trained on open data with global hard labels, measured on the held-out test set every round.
  - [ ] Given identical seed, method, and scenario, When a run repeats, Then per-round accuracy is reproducible.

#### Feature 4: Experiment runner
- **User Story:** As the researcher, I want a single CLI that runs any method × model × scenario × seed with the paper's hyperparameters as defaults, so that the ~30-run suite is scriptable.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given no overrides, When a run starts, Then hyperparameters equal the paper's (Adam, lr 1e-4, batch 80, 5 local epochs, 200 rounds).
  - [ ] Given a running experiment, When each round completes, Then per-round test metrics are appended to durable per-run storage (not held only in memory), and an interrupted run leaves previously written rounds readable.
  - [ ] Given a completed run, When its results directory is inspected, Then it contains the full config (method, model, scenario, seed, flags), per-round accuracy, and final accuracy/F1/precision/confusion-matrix.

#### Feature 5: SSFL ablation variants
- **User Story:** As the researcher, I want flag-controlled SSFL variants (no voting, no discriminating, neither, simply-filtering, fixed thresholds 0.7/0.8/0.9, soft labels rounded to 8/6/4/2 decimals), so that Figures 4–6 can be reproduced.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given `--no-discriminating`, When a round runs, Then clients upload predictions for all open samples without discriminator filtering.
  - [ ] Given `--no-voting`, When aggregation runs, Then global labels are produced without the majority-vote mechanism (direct aggregation of client predictions).
  - [ ] Given `--simply-filtering`, When a round runs, Then samples are marked unfamiliar purely by confidence threshold with no discriminator model.
  - [ ] Given `--threshold 0.8` (or 0.7/0.9), When discriminator sets are built, Then the fixed value replaces the per-client median.
  - [ ] Given a soft-label strategy with precision *x* ∈ {8,6,4,2}, When clients upload, Then payloads are soft-label vectors rounded to *x* decimals and the per-round communication cost reflects that payload size.

#### Feature 6: Communication-cost accounting
- **User Story:** As the researcher, I want per-round upload/download costs computed analytically for every method, so that Table IV and Figure 6's cost curves can be reproduced.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given a completed run, When costs are computed, Then cumulative MB at each round is derived from the method's actual payload sizes, and C@50 / C@75 / C@Top-Acc are reported from the run's accuracy curve.
  - [ ] Given an accuracy target a run never reaches (e.g., FD never hits 75%), When C@75 is requested, Then the report marks it as unreached rather than fabricating a value.

#### Feature 7: Report generation
- **User Story:** As the researcher, I want one command that regenerates Tables II–IV and Figures 3–6 from the results directory with the paper's values side-by-side, so that reproduction quality is assessable at a glance.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given a results directory with completed runs, When the report runs, Then it emits Table II (accuracy/F1/precision, ours vs. paper, with deltas), Table III (top-1 accuracy at rounds 10/50/100/150/200), Table IV (communication costs), confusion-matrix figures per scenario, and the ablation/threshold/label-strategy curves.
  - [ ] Given some runs are missing, When the report runs, Then it renders available results and explicitly lists missing runs instead of failing.

#### Feature 8: Smoke test suite
- **User Story:** As the researcher, I want a fast automated test suite covering data invariants and a tiny end-to-end run per method, so that hours-long experiments are never launched on broken code.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given the test suite, When it runs on the cached dataset, Then it verifies split disjointness, shapes, normalization range, per-scenario client counts, and class counts for devices 3/7.
  - [ ] Given each of the four methods plus one ablation variant, When a 2-round micro-run on a tiny subset executes, Then it completes and produces well-formed metrics.
  - [ ] Given the full suite, When it runs, Then it finishes in under ~5 minutes on the laptop.

### Should Have Features

#### Feature 9: Timing pilot & device selection
- **User Story:** As the researcher, I want a timing mode that measures wall-clock per round on the current machine (cpu vs. mps), so that I can decide local vs. cloud before committing to the suite.
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given a timing pilot (SSFL, Scenario 1, small number of rounds), When it completes, Then it reports measured seconds/round and a projected duration for each planned run.

#### Feature 10: Suite orchestration
- **User Story:** As the researcher, I want a script that runs the full ~30-run campaign sequentially with progress visibility, so that the suite can run unattended (e.g., overnight).
- **Acceptance Criteria (Gherkin Format):**
  - [ ] Given the campaign script, When it runs, Then completed runs are skipped on re-invocation and progress (done/remaining) is visible.

### Could Have Features

- **Multi-seed replication:** repeat headline runs over ≥3 seeds and report mean±std (paper reports single numbers; useful for judging tolerance).
- **Parallel client execution tuning:** configuration to exploit the 10 CPU cores for concurrent client training when MPS is not the bottleneck.

### Won't Have (This Phase)

- Real multi-machine/distributed deployment — everything is single-machine simulation.
- New methods, method improvements, or hyperparameter search beyond the paper's grid — this phase verifies, it does not extend.
- Other datasets (only N-BaIoT / mini-N-BaIoT).
- Privacy hardening (differential privacy, secure aggregation) — out of the paper's scope too.
- A serving/production IDS pipeline — offline experiments only.
- Exact numeric replication of the paper — seeds are unpublished; the target is tolerance-based match (see Success Metrics).

### MECE Check: Features
- [x] No two user stories describe the same capability (no overlap across MoSCoW categories)
- [x] All capabilities needed to solve the problem for every persona are present (no gaps)
- [x] Every feature has testable acceptance criteria
- [x] "Won't Have" explicitly accounts for capabilities that could be confused with in-scope features

## Detailed Feature Specifications

### Feature: SSFL federated round (Feature 3, most complex)
**Description:** One communication round of the proposed method, executed by every client in parallel plus a server aggregation step, repeated for 200 rounds.

**User Flow (per round t):**
1. Client distills its classifier on the shared unlabeled open set using the global hard labels voted in round t−1 (skipped at t=1).
2. Client trains its classifier on its private labeled data (5 local epochs).
3. Client computes a confidence score (max softmax) for every open-set sample; samples below the client's median confidence are labeled "unfamiliar", all private samples "familiar", and the discriminator trains on this two-class set.
4. Client predicts labels for all open samples, discards those its discriminator deems unfamiliar (marks them −1), and uploads the resulting hard-label vector.
5. Server collects all clients' label vectors, majority-votes per sample (ignoring −1 votes), broadcasts the global hard labels, and trains its own server classifier on the open set with those labels.
6. Server evaluates the server classifier on the held-out test set and records round metrics.

**Business Rules:**
- Rule 1: The confidence threshold defaults to each client's own median score for that round (per-client, per-round), not a global constant.
- Rule 2: Unfamiliar samples (−1) never enter a voting set; a sample with zero votes has no global label that round and is excluded from distillation.
- Rule 3: Uploaded labels are hard labels (integers), not logits — the communication cost accounting depends on this.
- Rule 4: The model evaluated for all reported metrics is the server classifier, never an individual client.

**Edge Cases:**
- A client's discriminator marks all open samples unfamiliar → client contributes no votes that round; the round must still complete.
- All clients mark the same sample unfamiliar → sample has no global label; distillation skips it.
- Tie in majority voting → deterministic tie-break (documented, seed-stable).
- A client has only one class locally (Scenario 2) → classifier is trained on one class; training and confidence computation must not degenerate (this is precisely the situation the discriminator exists for).

## Success Metrics

### Key Performance Indicators

- **Fidelity (primary):** SSFL final accuracy within ±3 percentage points of the paper per scenario (87.40 / 86.70 / 84.22); F1 and precision within ±4 points.
- **Ordering preserved:** SSFL outperforms FL, FD, DS-FL, MLP, and LSTM in every scenario, and the qualitative ablation findings hold (removing the discriminator hurts most; median threshold ≥ fixed thresholds; hard labels ≈ soft-label accuracy at a fraction of the cost).
- **Communication:** SSFL per-round upload cost is orders of magnitude below FL's, with C@Top-Acc in the same order of magnitude as Table IV (~0.5 MB).
- **Completeness:** all ~30 planned runs completed and every paper artifact (Tables II–IV, Figs 3–6) regenerated.
- **Efficiency:** full suite completes within ~2 weeks of wall-clock on available hardware.

### Tracking Requirements

| Event | Properties | Purpose |
|-------|------------|---------|
| Round completed | run id, round, top-1 test accuracy, wall-clock | Table III curves, convergence analysis, timing projections |
| Run completed | config (method/model/scenario/seed/flags), final accuracy/F1/precision, confusion matrix | Table II, Fig 3 |
| Payload uploaded/downloaded | round, bytes per client, direction | Table IV, Fig 6 cost curves |
| SSFL label quality | round, # unfamiliar per client, # unlabeled-after-vote, vote agreement | Diagnosing divergence from the paper's ablation behavior |
| Test suite run | pass/fail per invariant | Gate before long runs |

---

## Constraints and Assumptions

### Constraints
- Single researcher, single MacBook (Apple M4, 10 cores, 16 GB, MPS) as the default compute; cloud CUDA optional. Measured baseline: ~2.7–9 h per 200-round run on MPS → suite is ~1 week sequential.
- Stack fixed by prior decision: Flower pinned at 1.32.1 (Message API only — the ecosystem has heavy version churn), PyTorch, fresh uv-managed Python 3.12 environment.
- Raw data is fixed: the 7.6 GB N-BaIoT CSVs already in `data/` (devices 3 and 7 lack mirai traffic).
- Paper hyperparameters are non-negotiable defaults (Adam 1e-4, batch 80, 5 local epochs, 200 rounds, temperature-softmax for DS-FL).

### Assumptions
- The paper's reported setup is complete enough that undocumented details (e.g., voting tie-breaks, optimizer state across rounds, exact distillation loss) can be resolved by reasonable documented choices without invalidating the reproduction.
- "First 1000 records per device-category" is an acceptable reading of the paper's mini-N-BaIoT sampling ("We select 1000 traffic data records"); the seed-fixed builder makes the choice auditable.
- ±3–4 points is an acceptable reproduction tolerance given unpublished seeds and single-run reporting in the paper.
- 16 GB RAM suffices for 89 simulated clients with compact per-client state (verified feasible by research).

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Ambiguous paper details lead to results outside tolerance | High | Medium | Log every judgment call in the spec; use SSFL diagnostic tracking (label quality events) to localize divergence; ablation curves give intermediate checkpoints to compare against |
| Flower 1.32 Message API friction (no distillation-FL reference exists) | Medium | Medium | Smoke tests with 2-round micro-runs before any long run; the harness isolates method logic from transport |
| Suite runtime exceeds patience (~1 week MPS) | Medium | Medium | Timing pilot first; cloud CUDA escape hatch; campaign script resumable so runs spread over nights |
| Crash mid-run loses hours | Medium | Medium | Durable per-round metric writes; resumable campaign; runs restartable with same seed |
| RAM pressure with 89 clients | Medium | Low | Compact per-client state, disk checkpoints keyed by client id as fallback |
| Scenario 3 (Dirichlet α=0.1) instability mirrors the paper's hardest case | Low | Medium | Expected — the paper itself shows degraded, noisier curves there; compare qualitatively |

## Open Questions

- [ ] None blocking. (Resolved during brainstorm/research: full scope confirmed; stack fixed; compute deferred to timing pilot by design.)

---

## Supporting Research

### Competitive Analysis
No official code release for the paper exists (checked; the Flower ecosystem also has no FedMD/distillation-style baseline to reuse). Community FedMD implementations target removed Flower APIs and are unusable. This reproduction is built from the paper's equations directly.

### User Research
Stakeholder (the researcher) interviewed via brainstorm session: goal = full-fidelity reproduction (all tables, all ablations) as a foundation for future pseudo-labeling research; stack preference = Flower + PyTorch (explicit); compute = undecided pending timing pilot.

### Market Data
N-BaIoT is a standard IDS benchmark (9 IoT devices, gafgyt/mirai botnet traffic, 115 statistical features). The paper is IEEE IoT-J 2023 (Zhao et al., DOI 10.1109/JIOT.2022.3175918), with FD (Jeong et al.) and DS-FL (Itahara et al.) as the distillation-FL lineage being compared against.
