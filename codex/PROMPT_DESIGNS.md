# PROMPT SKETCHPAD

---
## Job Runner
---

```
Setup:
- Review AGENTS.md,  `Review  codex/agents/*`
- Find the tasks in  `codex/agents/TASKS/*`
- Note machine readable  specifications are outlined in `codex/specs/ragx_master_spec.yaml`
- Spend as much time as you need, quality of code is the most important thing here. 
- Ensure the code functions according to specs in real-life examples and tests.
- Ensure the creation of relevant tests.
- The best tests will be written before the code they verify and will provide extensive feedback as to the operation 
    of the code they cover to rival the best debuggers or manual debugging breakpoints and debug code. The best part
    is that once they're written, you've automated your debugging system and created a permanent verification of the
    code you created. any, changes to the codebase in the future causing a regression will be immediately apparent,
    preventing the worst kind of bugs.

Request:

-Complete `codex/agents/TASKS/06b_mcp_server_bootstrap.yaml`

Requirements: 
- verify operation using a combination of the feedback from tests and by running the components in the environment and analyzing their outputs
- it is of the greatest importance that tests continue to function and are created to verify everything about the functionality of every component.
- consider your development environment broken if tests are not running, stop everything and focus on getting them working before doing anything else
- consider your code not working if tests are failing, use this output to determine what is wrong with your code
```

---
## Synthesis (use for original set to add additional tasks) 
---

```
review the implementations of:

codex/agents/TASKS/06ab_core_tools_minimal_subset.yaml

in branches: 
codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml
codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-frqyft
codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-u13fvc
codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-whvxfk

you have access to the github token in the secret CODEX_READ_ALL_REPOSITORIES_TOKEN which will give you read access to all repositories and branches within 


to see if there is anything that might be used to improve our own.

add improvement tasks to codex/agents/TASKS/05h.. 05z... start numbering at the next available 05x (currently 05h, I think it is) following the format of the others. 
be considerate of the scope of each task when creating them, don't make them too broad, nor too small. Build each job to match where the currently used codex model 
performs best. 
```



```
PRE-FLIGHT (robust Git in sandbox)
1) Ensure remote and token:
  OWNER=pfahlr
  REPO=ragx
  : "${GITHUB_TOKEN:=$CODEX_READ_ALL_REPOSITORIES_TOKEN:-}"
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://${GITHUB_TOKEN}@github.com/pfahlr/ragx.git"
  git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
  git config --add remote.origin.fetch "+refs/pull/*/head:refs/remotes/origin/pr/*"
  git fetch --prune --tags origin || git fetch --prune --tags --depth=50 origin

2) Fetch candidate branches by refspec (sandbox-safe):

git fetch origin refs/heads/codex/fix-pybind11_dir-environment-variable-issue-ocvfcn:refs/remotes/origin/codex/fix-pybind11_dir-environment-variable-issue

git fetch origin refs/heads/codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml:refs/remotes/origin/codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml

git fetch origin refs/heads/codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-frqyft:refs/remotes/origin/codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-frqyft

git fetch origin refs/heads/codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-u13fvc:refs/remotes/origin/codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-u13fvc

git fetch origin refs/heads/codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-whvxfk:refs/remotes/origin/codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-whvxfk

review the implementations of:

codex/agents/TASKS/06ab_core_tools_minimal_subset.yaml

in branches: 
codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml
codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-frqyft
codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-u13fvc
codex/complete-tasks-in-06ab_core_tools_minimal_subset.yaml-whvxfk

you have access to the github token in the secret CODEX_READ_ALL_REPOSITORIES_TOKEN which will give you read access to all repositories and branches within 

analyze the differences between the implementations. determine which elements represent the most robust and effective code implementations, and using what you've learned plan an implementation that combines the best characteristics from the four branches you reviewed. 
```

---
## Operations 
---

```
Can you please list out the operations I should be able to perform interacting with the software at this point, specificially those related to the current feature branch (tasks from `codex/agents/TASKS/05*.yaml`). I need to verify the functionality beyond what is covered in the test set, as well as familiarize myself with the user interface. 
```

===
## Test Coverage 
---

```
Please review the codebase comparing it against the defined set of tests.
Determine the test coverage
Create any tests that are missing and ensure they pass

```

---
## Documentation Coverage
---

```
Please review the software up until this point. We need to provide extensive documentation, not only of the commands, but also of the many configurable components in this complex package:

- DSL for describing Parallel/Mult-Shot/Multi-Agent/Chaining/Tooling of data processing 
  - The automated processes provided by yaml such as the multi-agent, multi-shot domain spcific language for creating complex processing workflows between agents and MCP tools 
  - how to make simple requests to an LLM 
  - how to define complex workflows involving Models hosted locally and across various LLM providers in the cloud
  - The "flowscript" (working title) language abstraction of the yaml based system that we'll be defining to simplify the process of defining these. 
  - validating yaml workflow designs
  - how to find information helpful in designing workflows for whatever need you can imagine
  - budget contol features

- Vector Database
  - The class and python wrapping of the index builder 
  - how to extend with new vector database implementations
  - how to build vector databases 
  - how to query vector databases directly on the command line or via MCP
  - how to build vector databases with or without GPU
  - how to configure vector databases with or without GPU
  - how to configure parallel processing 

- MCP Tool System 
  - Running the MCP server 
  - Accessing the MCP server as a webservice or as a local process
  - How MCP tools are to be defined
  - exposing 3rd party MCP tools
  - exposing locally running software as an MCP tool
  - exposing existing web services as MCP tools
  - exposing the vector database as an MCP tool
  
- Resarch Collection Scripts
  - Collecting book and article links and metadata 
  - Downloading articles and books
  - Completing article and book metadata
  - Preparing Files for indexing

And whatever else I've forgotten to mention
  
```

---
## Merge Implementation Guidelines Variations
---

### Variation 1

```
**###Instruction###**
You are a **technical analyst and synthesis expert**. Your task is to **analyze and synthesize four versions of implementation guidelines** that were generated by Codex. These versions were created based on a comparison of how a specific task was implemented.

Each version represents a synthesized set of implementation strategies. Now, your job is to:

* **Compare** the approaches across the four versions
* **Identify overlaps, divergences, and strengths**
* **Synthesize a unified development plan** based on the best practices across all four
* Highlight **any missing or weak areas** in the original implementations, if applicable
* Ensure that the final plan is **actionable, scalable, and logically structured**

**###Format###**
Your output must include:

* ✅ Executive Summary of key takeaways (3–5 bullet points)
* 🧠 Comparative Analysis Table (or bullet list): summarizing unique and shared elements across V1–V4
* 🛠 Final Development Plan: clearly structured steps, technologies, or components
* ⚠️ Optional Enhancements: optional improvements, best practices, or tools to improve maintainability or scalability

**###Input###**
Below are the four Codex-generated implementation guideline versions:

**V1:** <output_placeholder>
**V2:** <output_placeholder>
**V3:** <output_placeholder>
**V4:** <output_placeholder>

**###Target Audience###**
Assume the output will be used by **senior developers or technical project leads** looking to standardize or optimize their implementation strategy.
```

### Variation 2: Codex Target


```
###Instruction###
You are a collaborative AI development agent tasked with refining a Codex-generated solution. Codex has analyzed and synthesized four implementation strategies for a specific task. Now, your job is to:

Act as an expert peer reviewing another AI’s output

Critically evaluate the structure, logic, and completeness of each version

Identify:

✅ Shared strengths across versions

🧠 Unique contributions or insights

⚠️ Redundant steps, hallucinations, or conflicting logic

Integrate the best elements from all versions into a final, optimized implementation plan

Offer rationale for key synthesis decisions

###Format###
Your output must follow this format:

🧾 Evaluation Summary: Key similarities, differences, and observations

🔎 Hallucination or Redundancy Check: Any parts that seem inconsistent, unclear, or duplicative

🧩 Synthesis Notes: A brief rationale explaining which parts from which version were selected for the final plan and why

🚀 Final Development Plan: A clear, step-by-step or structured outline

🛠️ Optional Enhancements (tooling, architecture, scalability suggestions)

###Persona###
Act as a senior AI systems engineer collaborating with Codex. Your tone is objective, technical, and forward-looking.

###Constraints###

Do not assume correctness—validate logic and structure

Ensure the final plan is scalable and production-ready

Use clear formatting and include headings or bullet points for readability

If needed, include brief pseudocode or code snippets

###Input###
You’re reviewing the following Codex-generated implementation guideline drafts:

V1: <output_placeholder>
V2: <output_placeholder>
V3: <output_placeholder>
V4: <output_placeholder>

```


