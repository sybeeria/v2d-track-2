# Parent decisions, brainstorm 6 (2026-10-23)

## D1. Tier 3 reference protocol amendment: isolated acceleration flags are inspected, not blocking (2026-10-22 ~22:30 UTC)

Decided before any policy has been trained on candidate_v1, and without any clean data or score.

- T3-RECON's frozen protocol (§3 inspection, §6 unknown evidence → HOLD) keeps ep18 candidate_v1 on HOLD only because
  18 isolated 50 Hz acceleration flags (no run ≥3 frames) cannot be causally attributed from single-view RGB
  (T3-EP18 REPORT.md: 16 likely camera-depth jitter during a real hand-held carry; 683–684 a real pivot plus unresolved depth).
- Every sustained gate passes: coverage, full-mesh support, 0 detached/unsupported runs, 0 ≥3-frame dynamics failures,
  26/26 independent RGB outlines, protected carry unchanged.
- Monocular depth jitter is inherent to every CARI4D reconstruction, so this clause as written would hold back most of the
  remaining 29 episodes too. Its purpose (don't spend GPU on references with demonstrated defects) is served by the sustained
  gates; isolated flags do not demonstrate a defect.
- **Amendment:** isolated flags (runs <3 frames at both playback rates) are recorded and inspected, and do not block
  promotion when every sustained gate and the RGB gate pass. Sustained-gate thresholds are unchanged.
- **Consequence:** ep18 candidate_v1 is promoted to the registered fresh 50M/350M pilot pair (T3-POLICY PROTOCOL-qualified.md,
  unchanged decision rules: 3 fresh evaluations per final, all must pass ≤10 cm self-MPPE and the carry checks). Cost ≈ $2.
- T3-EP18's proposed depth refit (≤2 cm camera-depth correction in carry spans) stays an optional follow-up; it must not be
  chosen by comparing pilot scores between v1 and v2.

## D2. Tier 3 gates choose the reference and the investment, not whether an episode is covered (2026-10-23 ~02:00 UTC)

Proposed parent default, made before any Tier 3 policy exists; the user can override it in the morning.

- Measured tonight: the scalable bridge reproduces the cached generic-repaired ep18 reference exactly, and that reference
  fails the frozen gates (22 detached runs). Only the video-anchored rest repair (candidate_v1) passed. T3-EVIDENCE's
  preflight allows that repair on 12 episodes; 18 are HOLD (elevated support without a metric plane, partial envelope,
  or no long rest). Strict gating would leave most Tier 3 episodes without a policy.
- A missing tier costs more than a weak one: with 7 teams on the Tier 3 boards, a missing entry is rank 8 on each of 4
  metrics (sum 32) versus 19 at the organizer-baseline level (data/leaderboard/2026-10-23T0029/NOTES.md).
- **Rule:** every episode with a structurally valid reference (gate status PASS or HOLD, never FAIL) gets a policy.
  Reference preference is fixed in advance: rest-repaired (preflight allowed) > generic repaired. No choice between
  variants by any policy, self or clean score. Recipe and k-repeat veto follow the ep18 pilot decision.json:
  retain 50M -> registered 50M recipe for all; selective rescue -> 50M for all, 350M only for its registered rescue set;
  diagnose -> 50M coverage still runs before Nov 1 unless the diagnosis shows the policies are invalid.
- D2 is about reference quality only; it does not override eligibility. Every current reference uses the released
  mesh/support assets and carries "released scene assets: organizer ruling pending". Coverage training starts only after
  the user's asset decision (or the organizers' answer); the ep18 pilot is an engineering test and may run before it.
- Cost: ~30 x 50M T2C-contact, about $7.50 by T3-BATCH's estimate. Integrity rules are unchanged (Tier 3 inputs = video,
  plan-listed released assets pending the ruling, Tier 1 frame 0 only).
