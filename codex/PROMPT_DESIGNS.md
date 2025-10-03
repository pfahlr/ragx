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

-Complete `codex/agents/TASKS/codex/agents/TASKS/06a_core_tools_minimal_subset.yaml`

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

codex/agents/TASKS/05c_toolpacks_loader_spec_alignment.yaml  

in branches: 
 codex/complete-toolpacks-loader-spec-alignment-hk1ntv  
 codex/complete-toolpacks-loader-spec-alignment-ymx51n
 codex/complete-toolpacks-loader-spec-alignment-9r5g15 and  
 codex/complete-toolpacks-loader-spec-alignment-z5pzt0 
 
to see if there is anything that might be used to improve our own.

add improvement tasks to codex/agents/TASKS/05h.. 05z... start numbering at the next available 05x (currently 05h, I think it is) following the format of the others. 
be considerate of the scope of each task when creating them, don't make them too broad, nor too small. Build each job to match where the currently used codex model 
performs best. 
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
