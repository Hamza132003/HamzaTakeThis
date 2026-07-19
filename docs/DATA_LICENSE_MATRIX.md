# DATA LICENSE MATRIX — AEGIS-X PRIME

Status: **no evaluation or training data is currently in the repository or acquired.** Every row below
is a candidate requiring legal review and operator approval before download or use. Nothing here is an
acquisition decision. Private, intercepted, classified, or otherwise unauthorized audio is prohibited
outright (spec §1); nothing in this plan uses it.

Statement required by spec §16.2: **military content and military channel are different things.** No
openly available corpus listed below is represented as genuine Hebrew battlefield communications, and
none may be described that way unless provenance and license actually establish it.

## Hebrew — clean / domain proxies

| Source | Reported license | Commercial | Verification status | Notes |
|---|---|---|---|---|
| ivrit.ai corpora | custom ivrit.ai license (versioned; terms vary by release) | TBD per release | UNVERIFIED | Largest open Hebrew speech effort; check per-dataset terms + attribution |
| HebDB | research release (reported permissive-with-attribution) | TBD | UNVERIFIED | |
| VoxKnesset | parliamentary-recording derivative; license per publication | TBD | UNVERIFIED | Speaker metadata useful for disjoint splits |
| Common Voice (he) | CC0 | yes | KNOWN (verify snapshot) | Read speech only — domain mismatch, useful for clean baseline |
| FLEURS (he) | CC-BY-4.0 | yes (attribution) | KNOWN (verify snapshot) | |
| Authorized institutional recordings | owner agreement | per agreement | N/A | References must be independently produced (spec §16.3) |

## Persian — clean / domain proxies

| Source | Reported license | Commercial | Verification status | Notes |
|---|---|---|---|---|
| PSRB public subset | research benchmark terms | TBD | UNVERIFIED | Benchmark of record for fa ASR |
| Common Voice (fa) | CC0 | yes | KNOWN (verify snapshot) | Large fa coverage |
| FLEURS (fa) | CC-BY-4.0 | yes | KNOWN (verify snapshot) | |
| Licensed Persian corpora (commercial vendors) | paid license | per contract | N/A | Only if operator procures |

## Degraded-channel corpora

| Source | Reported license | Verification status | Notes |
|---|---|---|---|
| LDC RATS (Persian/Farsi partitions) | LDC membership/fee license | UNVERIFIED / operator procurement | The canonical degraded-radio corpus; usable for LID/SAD/keyword robustness where the license permits |
| SPINE / public-safety radio corpora | LDC / varied | UNVERIFIED | **Channel/noise robustness only — English content; never counted toward Hebrew/Persian accuracy** |
| Synthetic RATS-style degradation of clean Hebrew/Persian | inherits source license + noise/IR licenses | Depends on components | Degradation recipes must be versioned; augmented copies are **never** counted as independent real-world sources (spec §16.8) |

## Annotation & governance requirements (binding on any acquisition)

- Gold sets: two independent native annotators per language + third-expert adjudication; branch-blind first pass; inter-annotator WER/CER reported; locked normalization guide (spec §16.3).
- Speaker-, session-, and source-disjoint splits; grouped splitting for recurring speakers (spec §16.4).
- No production model output as initial reference; no evaluation items used for tuning of any kind.
- Every accepted dataset gets a row in `DATA_LICENSES.yaml` + a manifest entry (ID, hash, license, split, annotation status) before first use.
- Nothing containing personal audio enters Git; retention and deletion policy per SECURITY.md/PRIVACY.md (Phase 0).
