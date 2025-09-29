## Job Runner
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

-Complete `codex/agents/TASKS/05j_toolpacks_loader_execution_validation.yaml`

Requirements: 
- verify operation using a combination of the feedback from tests and by running the components in the environment and analyzing their outputs
- it is of the greatest importance that tests continue to function and are created to verify everything about the functionality of every component.
- consider your development environment broken if tests are not running, stop everything and focus on getting them working before doing anything else
- consider your code not working if tests are failing, use this output to determine what is wrong with your code
```

## Synthesis (use for original set to add additional tasks) 
review the implementations of < list of tasks> in https://github.com/pfahlr/ragx/tree/<competing feature branch> to see if there is anything that might be used to improve our own.
```
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

## Operations 
```
Can you please list out the operations I should be able to perform interacting with the software at this point, specificially those related to the current feature branch (tasks from `codex/agents/TASKS/05*.yaml`). I need to verify the functionality beyond what is covered in the test set, as well as familiarize myself with the user interface. 
```

## Test Coverage 
```
Please review the codebase comparing it against the defined set of tests.
Determine the test coverage
Create any tests that are missing and ensure they pass

```

## Documentation Coverage
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



- [x] codex/agents/TASKS/04a_vectordb_protocol_and_registry.yaml
- [x] codex/agents/TASKS/04b_cpp_stub_backend_scaffold.yaml
- [x] codex/agents/TASKS/04c_python_flat_index_baseline.yaml
- [x] codex/agents/TASKS/04d_cpp_stub_flat_index_feature_parity.yaml
- [x] codex/agents/TASKS/04e_shards_merge_contract_and_tests.yaml
- [x] codex/agents/TASKS/04f_cli_scanner_and_formats_md_pdf.yaml


- [x] codex/agents/TASKS/05a_toolpacks_loader_minimal.yaml
- [x] codex/agents/TASKS/05b_toolpacks_executor_python_only.yaml

- [x] codex/agents/TASKS/05c_toolpacks_loader_spec_alignment.yaml
- [x] codex/agents/TASKS/05d_toolpacks_executor_python_only_plus.yaml
- [x] codex/agents/TASKS/05e_toolpacks_tests_completeness.yaml
- [x] codex/agents/TASKS/05f_toolpacks_docs_sync.yaml
- [x] codex/agents/TASKS/05g_toolpacks_legacy_shim.yaml
- [x] codex/agents/TASKS/05h_toolpacks_loader_metadata_validation.yaml
- [x] codex/agents/TASKS/05i_toolpacks_loader_caps_env_templating.yaml
- [x] codex/agents/TASKS/05j_toolpacks_loader_execution_validation.yaml

- [ ] codex/agents/TASKS/06a_core_tools_minimal_subset.yaml
- [ ] codex/agents/TASKS/06b_mcp_server_bootstrap.yaml
- [ ] codex/agents/TASKS/06c_mcp_envelope_and_schema_validation.yaml
- [ ] codex/agents/TASKS/06d_mcp_toolpacks_transport.yaml

- [ ] codex/agents/TASKS/07a_dsl_policy_engine_completion.yaml
- [ ] codex/agents/TASKS/07b_budget_guards_and_runner_integration.yaml
- [ ] codex/agents/TASKS/07c_transforms_sandbox_minimal_python.yaml

- [ ] codex/agents/TASKS/08a_rest_facade_minimal.yaml
- [ ] codex/agents/TASKS/08b_retrieval_hybrid_and_rerank_stub.yaml

- [ ] codex/agents/TASKS/09a_planner_multiperspective_core.yaml
- [ ] codex/agents/TASKS/09b_hitl_checkpoints_minimal.yaml
- [ ] codex/agents/TASKS/09c_agents_pipeline_minimal.yaml

- [ ] codex/agents/TASKS/10a_observability_trace_metrics.yaml
- [ ] codex/agents/TASKS/10b_ci_conformance_suite.yaml
- [ ] codex/agents/TASKS/10c_adapter_milvus_stub.yaml
- [ ] codex/agents/TASKS/10d_adapter_qdrant_stub.yaml
- [ ] codex/agents/TASKS/10e_runner_templating_and_decisions.yaml
- [ ] codex/agents/TASKS/10f_runner_loops_and_stop_conditions.yaml

- [ ] codex/agents/TASKS/11a_core_tools_full_set.yaml
- [ ] codex/agents/TASKS/11b_mcp_limits_and_errors.yaml
- [ ] codex/agents/TASKS/11c_retrieval_provider_connectors.yaml

- [ ] codex/agents/TASKS/12a_vector_backend_hnsw.yaml
- [ ] codex/agents/TASKS/12b_vector_backend_cuvs_stub.yaml
- [ ] codex/agents/TASKS/12c_serving_node_minimal.yaml

- [ ] codex/agents/TASKS/13a_linter_sarif_output.yaml
- [ ] codex/agents/TASKS/13b_runner_otel_hooks.yaml

- [ ] codex/agents/TASKS/14a_docs_site_scaffolding.yaml
- [ ] codex/agents/TASKS/14b_examples_and_tutorials.yaml

- [ ] codex/agents/TASKS/15a_release_engineering.yaml
- [ ] codex/agents/TASKS/15b_ci_matrix_and_caching.yaml
- [ ] codex/agents/TASKS/15c_security_and_sandbox_policy.yaml
- [ ] codex/agents/TASKS/15d_performance_fixtures_and_bench.yaml
- [ ] codex/agents/TASKS/15e_end_to_end_demo_script.yaml
- [ ] codex/agents/TASKS/15f_quality_bar_exit_checks.yaml

- [ ] codex/agents/TASKS/20a-flowscript-spec-and-grammar.yaml
- [ ] codex/agents/TASKS/20aa_flowscript_scaffold.yaml
- [ ] codex/agents/TASKS/20ab_flowscript_tests_and_ci.yaml
- [ ] codex/agents/TASKS/20ac_rw_dispatch_integration.yaml
- [ ] codex/agents/TASKS/20ad_parser_engine_stub.yaml
- [ ] codex/agents/TASKS/20ae_lints_for_flowscript.yaml
- [ ] codex/agents/TASKS/20af_update_ci_job_explicit.yaml

- [ ] codex/agents/TASKS/20b-flowscript-parser-ast.yaml
- [ ] codex/agents/TASKS/20c-flowscript-compiler-to-yaml.yaml
- [ ] codex/agents/TASKS/20d-runner-execution-core.yaml
- [ ] codex/agents/TASKS/20e-policy-and-budget-engine.yaml
- [ ] codex/agents/TASKS/20f-transforms-sandbox.yaml
- [ ] codex/agents/TASKS/20g-decisions-and-loops.yaml
- [ ] codex/agents/TASKS/20h-adapters-and-costing.yaml
- [ ] codex/agents/TASKS/20i-dsl-linter-and-ci.yaml
- [ ] codex/agents/TASKS/20j-example-flows-and-docs.yaml

- [ ] codex/agents/TASKS/30a-mcp-envelope-and-schema.yaml
- [ ] codex/agents/TASKS/30b-mcp-discovery-and-prompts.yaml
- [ ] codex/agents/TASKS/30c-toolpacks-loader-and-exec.yaml
- [ ] codex/agents/TASKS/30d-core-tools-canonical.yaml

- [ ] codex/agents/TASKS/31a-rest-parity-facade.yaml

- [ ] codex/agents/TASKS/40a-vectordb-faiss-shim-and-serialize.yaml
- [ ] codex/agents/TASKS/40b-vectordb-sharding-and-merge.yaml
- [ ] codex/agents/TASKS/40c-vectordb-md-ingest-frontmatter.yaml

- [ ] codex/agents/TASKS/41a-retrieval-hybrid-and-rerank.yaml

- [ ] codex/agents/TASKS/42a-planner-and-mindmap.yaml
- [ ] codex/agents/TASKS/42b-hitl-outline-and-merge.yaml

- [ ] codex/agents/TASKS/50a-observability-trace-contracts.yaml
- [ ] codex/agents/TASKS/50b-ci-coverage-and-cache.yaml

- [ ] codex/agents/TASKS/60a-codex-bootstrap-and-guardrails.yaml

- [ ] codex/agents/TASKS/70a-docs-foundation.yaml
