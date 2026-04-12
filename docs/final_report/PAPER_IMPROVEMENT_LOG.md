# Paper Improvement Log

Auto-review loop on `report_groupID24.tex` (CS5242 course project, Group ID24).
Reviewer: GPT-5.4 (xhigh reasoning) via Codex MCP.
Review framing: course TA / Master's course grader, **not** top-venue submission.
User guidance: "we don't have to beat sota or baseline, just follow the introduction as the goal of this project, it's just a course project".

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (original) | 6/10 (B) | Needs major fixes | Baseline |
| Round 1 | — | — | CRITICAL + MAJOR fixes implemented (see below) |
| Round 2 | 8/10 (A-) | Needs minor fixes | Minor cleanup (loss vs reward, policy-evidence tie, eq overflow) |

Net change: **+2 points (B → A-)** across 2 rounds.

## Round 1 Review (Baseline = Round 0)

<details>
<summary>GPT-5.4 xhigh Review — Round 1 (full text)</summary>

**Overall Score:** `6/10 (B)`

**Summary:** The underlying project looks like a real from-scratch implementation effort, and the paper does show some genuine learning through environment design, debugging, and comparison of DQN vs. DDQN. But as a report, it only partially delivers on the stated pedagogical goal: the write-up is not yet self-consistent, several results are incomplete, and the "from first principles" understanding is implied more than clearly demonstrated.

**Strengths:**
- The project goal is appropriate and well-framed for a course report: learning DRL by implementing DQN/DDQN from scratch, not chasing SOTA.
- There is clear engineering substance: custom environment, replay buffer, target network, epsilon schedule, reward shaping, and evaluation pipeline.
- The `Challenges` section is one of the strongest parts; it shows real debugging and iterative refinement rather than a polished-only story.
- The report is honest about underperforming buy-and-hold and gives a reasonable market-regime explanation instead of overselling.
- The appendix ideas are good for a course audience, especially the episode walkthrough.

**Weaknesses:**

*CRITICAL*
- The results section is not submission-ready. The main table still contains `---` entries and a literal TODO note ("ARIMA and agent numbers will be updated..."), even though the paper claims comparison against multiple baselines.
  - Fix: either fill in every reported baseline result, or remove the unreported baselines everywhere (Baselines section, table, figure caption, results discussion). Delete all placeholder/TODO language.
- The experimental protocol is internally inconsistent on whether shorting is allowed. Environment Design says `pos_t ∈ {-1,0,+1}` and Challenges mentions short-selling, but the ARIMA baseline says it matches the "long-only constraint used during RL training."
  - Fix: choose one setting and make it consistent across Environment Design, Baselines, Challenges, and Contributions.

*MAJOR*
- Data provenance and scope are inconsistent. Paper says kagglehub/alincijov dataset, but Contributions says yfinance. Also introduces 30 DJIA tickers, yet only reports AAPL.
- The report only partially delivers on "understand DRL from first principles." Lists components and hyperparameters but does not clearly explain what each mechanism is doing.
- Citation practice is below expected standard. Bibliography exists, but none of the listed papers are cited in the body text.
- Some claims overreach: text says DQN may "avoid the worst drawdowns," but the table shows worse max drawdown than buy-and-hold. Low correlation doesn't "confirm" distinct signals.
- Important methodology details missing or ambiguous: γ not reported, TD target omits terminal-state mask, transaction cost underspecified.
- Section-level inconsistencies: says "two baselines" but lists four.

*MINOR*
- Tighten notation, ensure figure/table cross-references are consistent, describe correlation analysis as AAPL-only case study.

**Verdict:** `Needs major fixes`

</details>

### Round 1 Fixes Implemented

1. **Pruned baselines to match introduction.** Removed Momentum and Random Forest baselines entirely (no quantitative runs existed for them). Kept ARIMA(1,0,0) as a qualitative reference visible in Fig. 2. Table reduced from 6 columns (with `---` placeholders) to 3 columns (DQN / DDQN / B&H) containing real numbers. Removed the "ARIMA and agent numbers will be updated..." caption note.

2. **Resolved shorting inconsistency.** Environment Design now explicitly states "Shorting is allowed throughout training and evaluation." The ARIMA baseline description was rewritten as "a deliberately long-only directional bet... unlike the RL agent (which may also go short)", removing the misleading "matching the long-only constraint used during RL training" phrase.

3. **Data provenance made consistent.** Contributions section was corrected from `yfinance` to `kagglehub (alincijov/trading)`. Data section now explicitly frames the scope as "AAPL as a single-asset case study" with a note that the pipeline loads a 10-ticker DJIA subset.

4. **Added pedagogical explanation paragraph.** New "What each component is for" subsection in Deep Learning Methods walks through replay buffer, target network, exponential ε-decay, reward clipping + Huber loss, and DDQN target decoupling in plain conceptual language, grounding each in an observation from the actual runs.

5. **Added inline citations.** `\cite{mnih2015}` at DRL / DQN, `\cite{hasselt2016}` at DDQN, `\cite{yang2020}` at RL-for-trading. The bibliography is now actually cited.

6. **Softened overclaims.** The "DQN partially learns to avoid worst drawdowns" sentence was replaced with a passage that explicitly acknowledges DQN's max drawdown is *worse* than buy-and-hold. Correlation claim softened from "confirming distinct signals" to "suggesting limited linear redundancy on this ticker."

7. **Filled in missing methodology.** Added γ = 0.99, terminal-state mask `(1 - d_t)` in both DQN and DDQN TD targets, explicit statement that the 10 bps flat transaction cost is what was actually used (proportional mode exists in code but is not used).

8. **Fixed reward equation.** The previous form `log(V_t/V_{t-1}) - c_t·I[...]` was mathematically inconsistent with the code (cost is applied inside the portfolio update, not additively on the log-return). Rewritten as a portfolio-update equation plus the log-return of that update, with drawdown penalty and final clipping, matching the implementation in `src/env.py`.

9. **Introduction rewritten** to explicitly state the pedagogical goal and that we do not aim to beat the passive benchmark.

---

## Round 2 Review

<details>
<summary>GPT-5.4 xhigh Review — Round 2 (full text)</summary>

**Overall Score:** `8/10 (A-)`

**Summary:** This revision is substantially stronger. It now mostly delivers on the stated pedagogical goal: the paper clearly frames itself as a from-scratch DRL case study, explains what the main DQN/DDQN components are doing, and interprets the results honestly rather than defensively. The remaining issues are mostly report-quality cleanup, not structural problems with the project.

**Strengths:**
- The pedagogical framing is now clear and appropriate for a course project.
- The paper is much more self-consistent across data source, scope, environment design, baselines, and results.
- The new "What each component is for" paragraph materially improves the "understand DRL from first principles" goal.
- The results discussion is now aligned with the actual numbers, especially on drawdown.
- The methodology is reported more cleanly: γ, terminal mask, and actual transaction-cost setting are now specified.
- The `Challenges` section remains a strong part of the report because it shows real debugging and learning.
- Reframing AAPL as a single-asset case study is a good fix; it makes the scope honest and defensible.

**Remaining Weaknesses:**

*CRITICAL:* None remaining.

*MAJOR:*
- The `Training and Results` text says DQN shows "training-loss reduction," but the figure caption describes training reward, not loss. Claim/evidence mismatch.
- The `Contributions` section appears to list only two members in a three-student group. For a course report, authorship/accountability should be complete.

*MINOR:*
- ARIMA is still presented in Baselines but only shown qualitatively. Acceptable if intentional, but reads awkwardly.
- The "DQN more willing to stay long, DDQN more cautious" statement would be stronger if tied explicitly to evidence such as the walkthrough figure or trade frequency.
- Clean up any drafting placeholders before submission.

**Actionable Fixes for MAJOR issues:**
- Training-loss vs reward: change the sentence to describe the metric actually plotted, or replace the left panel with a true loss curve.
- Incomplete contributions: add the third student explicitly with a short sentence describing their concrete contributions.

**Verdict:** `Needs minor fixes`

</details>

### Round 2 Fixes Implemented

1. **Fixed training-loss vs training-reward mismatch.** The sentence "DQN shows a clear reduction in training loss over time" was replaced with a description of the left panel as per-episode training reward (20-episode MA), correctly matching the figure caption and what is actually plotted.

2. **Tied policy characterisation to concrete evidence.** The "DQN more willing to stay long / DDQN more cautious" claim was rewritten to explicitly cite the episode walkthrough appendix figure (for DQN's long commitment during extended stretches) and the test-portfolio-value trajectory in Fig. 2 (for DDQN staying closer to the zero-position line), grounding both halves of the comparison in visible evidence.

3. **Fixed overfull hbox in DDQN equation.** The single-line DDQN target equation (12.6pt overfull in one-column) was converted to a `multline` environment and split across two lines. 0 overfull hboxes remain.

### Round 2 Items NOT Fixed (by design)

- **Contributions section third member.** The reviewer flagged that only 2 of 3 group members are listed. The third member's row was explicitly removed from the file by the user (before Round 1 started) with an intentional-change marker, so this round did not re-add it. **Surface this to the user** — if the removal was accidental, the row should be restored; if intentional (e.g., group composition change), the paper is correct as-is.

- **ARIMA qualitative-only presentation.** Acknowledged as acceptable by the reviewer; not changed because producing ARIMA numbers would require re-running `scripts/baselines.py` and saving results — outside the scope of a writing-only improvement loop.

- **`[cite ...]` / `R^10` / `Eq. ref:reward` placeholders** flagged as minor. These were artifacts of the representational text sent in the review prompt (plain-text paraphrase), not real placeholders in the actual `.tex`, which uses proper `\cite{}`, `\mathbb{R}^{10}`, `Eq.~\ref{eq:reward}`. False positive.

## Format Check

- **Pages:** 8 (within typical 6--8 page course-report range)
- **Overfull hbox:** 0 (fixed during Round 2)
- **Underfull hbox:** 17 (all minor badness, none affecting readability — typical for two-column layout)
- **Undefined references:** 0
- **Missing images:** 0

## PDF Artifacts

- `report_groupID24_round0_original.pdf` — Original, pre-review
- `report_groupID24_round1.pdf` — After Round 1 fixes (CRITICAL + MAJOR)
- `report_groupID24_round2.pdf` — Final, after Round 2 minor cleanup
- `report_groupID24.pdf` — Current (identical to round2)
