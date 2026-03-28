# CS5242 Group Project Requirements

**Course:** CS5242 Neural Networks and Deep Learning, Semester 2 2025/26
**Instructor:** Xavier Bresson, NUS

---

## Project Philosophy

- Focus on **understanding deep learning fundamentals** and practical skills to develop a data analysis project
- Design from scratch, debug, understand, and train learning algorithms
- Understand **why it works and why it does not**
- NOT about: copying GitHub code, winning Kaggle, using 3 lines of Keras, running long GPU experiments, or achieving 90% accuracy

## Project Goals

1. Download or prepare a dataset (novel or existing)
2. Implement deep learning techniques with simple model(s) as baseline
3. Propose improvements (motivation, description, equations, implementation, results, discussion)
4. Demonstrate initiatives (own scraper, dynamic visualization, new data insights, etc.)

## Pre-trained Models

- Primary goal is assessing understanding, not achieving high accuracy
- Building from first principles / "from scratch" is expected
- Pre-trained models (e.g. HuggingFace) are allowed but must be **well justified**

## Project Development Steps

| Step | Description |
|------|-------------|
| 1 | Identify a data analysis problem solvable with deep learning |
| 2 | Dataset collection (existing or new) |
| 3 | Data exploration (statistics, visualization) |
| 4 | Pre-processing (cleaning, normalization) |
| 5 | Data analysis with deep learning (compare different models) |
| 6 | Numerical results (analysis, interpretation, conclusion) |
| 7 | Report (Word/LaTeX or Python Notebook + Markdown) |
| 8 | Video presentation with slides |

## Team Formation

- **Group size:** 2-5 members
- Team contract required: distribute tasks equally, all members sign
- Teammates can be from different tutorial groups
- All teammates receive the **same grade**

## Deliverables & Deadlines

| Deliverable | Deadline | Notes |
|-------------|----------|-------|
| Group formation (`group_project_formation.txt`) | Sun Feb 22, 2026 11:59pm (Week 6) | Canvas > Assignments > Group Project Formation |
| Project plan + team contract (`project_plan_contract_groupIDXX.pdf`) | Sun Mar 8, 2026 11:59pm (Week 7) | One-page plan (strict limit) + signed contract. Does not count toward grade. |
| Final deliveries (`project_groupID.zip`) | **Sun Apr 26, 2026 11:59pm (Week 14)** | Canvas > Assignment > Group project deliveries |

### Late Penalties

- Project plan: **10%** of group grade per late day
- Final deliveries: **25%** of group grade per late day

## Project Plan Requirements

- **Strictly one page** (grade = 0 if exceeded; references allowed as extra pages)
- Any style/format (single/double column, etc.)
- Suggested template: project motivation, description, proposed solution, milestones
- Must include signed team contract on an additional page
- **Missing signed contract = grade of 0**
- TA feedback provided by Fri Mar 13, 2026 (Week 8)

## Final Submission Contents

The `.zip` file (max 500MB) must include:

1. **Working/reproducible Python notebook** (code demo)
2. **Project report** (can be merged with notebook)
3. **Presentation slides**
4. **Video recording**

File naming: `project_groupID.zip` (e.g. `project_group31.zip`)

## Video Presentation

- **Maximum 10 minutes** (grade = 0 if beyond 11 min)
- Each teammate must present their contribution (grade = 0 if not)
- ~3-4 min/member for group of 3, ~2 min/member for group of 5
- Must cover: motivation, data acquisition, exploration, pre-processing, DL solutions, results analysis, future development
- Use slides (1 slide ~ 1-2 min)

## Weekly Monitoring & Zoom Meeting

- **Weekly progress updates** to your group TA: every Friday by 6pm, Week 8 to Week 13
- At least **one Zoom meeting** with TA (ideally by Week 11)
- Weekly updates + Zoom meeting = **10 pts** of project grade (awarded independently of content)

## Marking Scheme

| Component | Points |
|-----------|--------|
| Project plan & team contract | 0 pts (required but ungraded) |
| Weekly updates + 1 Zoom meeting | 10 pts |
| Steps 1-8 (core project work) | 65 pts |
| Initiative bonus | up to 25 pts |
| **Total** | **100 pts** |

## GPU Resources

- Project should NOT require extensive top-tier GPU experimentation
- Options: SoC Compute Cluster (free, queue-based), Google Colab (free, 12hr/day), Google Cloud ($300 free credit), Google Education Program ($50/student)

## Dataset Sources

- UCI, Kaggle, Paperswithcode, GitHub
- Or scrape/collect new data via APIs (Twitter, Meta, etc.)
