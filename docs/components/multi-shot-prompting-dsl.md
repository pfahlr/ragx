# Multi-Shot Prompting  DSL

RAGX is a clean-slate, DSL-driven orchestration system for multi-step AI workflows that blend LLMs with MCP tools and retrieval—under strict policy and budget guards. Teams describe flows as graphs (units, decisions, loops, transforms), then run them with deterministic caching, pricing-aware cost control, and rich observability (logs/metrics/traces). A static linter and JSON Schemas enforce correctness up front, while a modular runtime (Toolpacks, REST/STDIO, optional C++ vector index) delivers portability and performance. The result is a transparent, auditable way to design, execute, and scale complex research and writing pipelines—without giving up safety, cost control, or reproducibility

### What this system does (layperson’s view)

Think of it like a **director’s script for AI conversations**.
Instead of just asking an AI one question and getting one answer, this system lets you **plan out a whole sequence of steps**—with rules, limits, and safety checks—so the AI works more like a team of specialists following a playbook.

---

### The key ideas

* **Steps (nodes):** Each step is a job to do. Sometimes it’s the AI writing text, sometimes it’s checking its own work, sometimes it’s searching the web or looking up notes in a database.

* **Connections (edges):** The steps are linked together so that the output from one becomes the input for the next.

* **Policies (tool rules):** You can tell the system “At this point, only use Google-like search” or “Don’t use the internet here, just rely on our notes.” It’s like setting guardrails for what tools are allowed.

* **Budgets (limits):** Each run has a “budget” in terms of time, money, and effort. You can say “Don’t spend more than $3,” “Don’t use more than 10,000 words worth of AI processing,” or “Stop after 3 rounds of refinement.”

* **Loops:** If you want the AI to improve on its answer (like editing a draft several times), you can tell it “repeat this feedback-and-refine step up to 3 times, but stop early if it looks good enough.”

* **Decisions:** Sometimes the AI has to choose a path. For example, if it thinks more research is needed, it can search; if it already has enough info, it can skip ahead to giving an answer.

* **Transforms:** These are little helpers that clean up or reformat text between steps—like splitting a big document into chunks or merging several answers into a summary.

* **Observability (watching what happens):** Every step is logged: what went in, what came out, how long it took, and how much it cost. You get a play-by-play record of the run, so you can check where things went right (or wrong).

---

### Why it matters

* **Control:** You’re not just throwing a question at an AI—you’re controlling *how* it thinks, what it’s allowed to use, and when to stop.

* **Safety:** Built-in limits keep costs, time, and risks under control.

* **Flexibility:** You can design new “strategies” for problem-solving, from simple Q&A to complex multi-agent research teams.

* **Transparency:** Every step is visible and auditable, like a flight recorder for AI reasoning.

---

### An everyday analogy

Imagine you’re baking a cake:

* One **step** is mixing ingredients, another is baking, another is frosting.
* **Policies** are like: “Don’t use nuts in this recipe.”
* **Budgets** are like: “Don’t spend more than $10 on ingredients.”
* A **loop** might be: “Taste the frosting, adjust sugar, repeat up to 3 times until it’s right.”
* A **decision** could be: “If the cake isn’t cooked in the middle, put it back in the oven, otherwise start decorating.”
* **Observability** is writing down what you did at each stage, so you can repeat it (or fix mistakes) next time.

This system gives you that same kind of **structured recipe**, but for running AI workflows instead of baking.

