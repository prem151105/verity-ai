# Research ledger: a paper-informed Verity runtime

Reviewed 2026-10-06. Scope: 50 paper abstract/metadata pages, plus six primary technical/design sources. This is a targeted architecture scan, not 50 full-paper replications or a systematic literature review. Selected full-text pages for MAST, Chain-of-Verification, GPTSwarm and TradingAgents were also inspected. Results reported by those authors are not measured results for Verity.

## Decisions implemented

1. Replace LangGraph with explicit Python state transitions and a fixed eight-role workflow. Preserve direct Gemini calls and existing source-anchored verification.
2. Add one skeptical review before writing. It inspects retrieved risk excerpts and the analyst summary. It cannot certify claims or override the verifier.
3. Use a deterministic adjudicator: request revision only within the configured limit, stop repeated drafts, and retain unresolved markers.
4. Emit started/completed/failed/cancelled events, use one run identity throughout, and retain snapshots in SQLite. Interrupted runs remain inspectable; automatic resume is not implemented.
5. Present evidence, routing decisions and unresolved claims separately from the research narrative. Never substitute a demo's scripted output for measured live verification.

These are engineering inferences from the sources below. The custom composition is project-specific, not a claim of a novel validated algorithm.

## Paper scan

| # | Primary source | Design implication / disposition |
|---|---|---|
| 1 | [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) | Role-based conversation motivates named handoffs; free-form group chat is not adopted. |
| 2 | [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) | Structured procedures motivate the explicit stage order and review contract; no MetaGPT dependency. |
| 3 | [CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society](https://arxiv.org/abs/2303.17760) | Role specialization informs the skeptical reviewer; role-playing itself does not validate evidence. |
| 4 | [ChatDev: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) | Staged communication informs the observable handoff log; software-development results are not finance results. |
| 5 | [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692) | Model aggregation is an alternative; deferred because additional model outputs add cost and correlated errors. |
| 6 | [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) | Failure taxonomy motivates bounded loops, preserved checkpoints, failure status, and observable termination. |
| 7 | [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) | Debate motivates a challenge pass; majority agreement is deliberately not used as a truth criterion. |
| 8 | [Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://arxiv.org/abs/2305.19118) | Divergent critique motivates the skeptic; one pass limits overhead and does not reproduce MAD. |
| 9 | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | Feedback motivates the writer–verifier revision cycle; cross-run episodic learning is not implemented. |
| 10 | [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | Iterative refinement motivates revisions; unchanged-draft detection prevents repeating the same attempt. |
| 11 | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | Tool-grounded actions motivate visible retrieval and computation steps; no hidden reasoning traces are exposed. |
| 12 | [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | Search over candidate plans is an alternative; deferred until a task-specific quality signal exists. |
| 13 | [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687) | Graph organization informs explicit edges; arbitrary thought-graph search is not implemented. |
| 14 | [AgentScope: A Flexible yet Robust Multi-Agent Platform](https://arxiv.org/abs/2402.14034) | Message-oriented robustness informs event envelopes and failure reporting; no distributed actor runtime. |
| 15 | [Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents](https://arxiv.org/abs/2405.16533) | Real-world tool selection motivates future tool-evaluation cases; current tool routes remain explicit. |
| 16 | [$τ$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) | Interaction evaluation motivates end-to-end state tests; this repository has not run tau-bench. |
| 17 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | Retrieval grounding supports keeping original documents separate from generated reports. |
| 18 | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | Dense retrieval is a baseline for the existing Chroma path; financial retrieval quality is unmeasured. |
| 19 | [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) | Retrieve/generate/critique separation informs review stages; no trained reflection tokens are implemented. |
| 20 | [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | Retrieval quality motivates stopping when no evidence is available; automatic web repair is deferred. |
| 21 | [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251) | Claim-level factual evaluation motivates the evidence ledger; comprehensive atomic decomposition remains absent. |
| 22 | [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627) | Citation quality motivates exact source labels and inspectable source excerpts. |
| 23 | [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495) | Independent checking motivates source-grounded verification; this is not a CoVe reproduction. |
| 24 | [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) | Consistency-based detection is an alternative; repeated generations do not replace original evidence. |
| 25 | [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) | Context-position sensitivity motivates targeted excerpts; truncation still requires evaluation. |
| 26 | [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176) | Cost-aware cascading informs BACE's existing local-first escalation budget; dollar savings unmeasured. |
| 27 | [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665) | Learned routing is deferred pending labeled data; current routing is deterministic. |
| 28 | [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) | Agent evaluation motivates testing interactions and failure states, not just report appearance. |
| 29 | [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983) | Assistant benchmarking motivates realistic tool tasks; GAIA is not evaluated here. |
| 30 | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) | Outcome-based evaluation informs regression testing; code-benchmark success does not transfer to finance. |
| 31 | [A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/abs/2308.11432) | Broad agent architecture taxonomy informs separation of planning, tools, state and observation. |
| 32 | [FinQA: A Dataset of Numerical Reasoning over Financial Data](https://arxiv.org/abs/2109.00122) | Numerical reasoning motivates code-based arithmetic and finance-specific future evaluation. |
| 33 | [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | Tool use motivates a narrow model/tool boundary; no tool-use fine-tuning is performed. |
| 34 | [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435) | Program-aided computation supports keeping arithmetic in Python rather than generated prose. |
| 35 | [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://arxiv.org/abs/2211.12588) | Separating computation from reasoning supports the existing deterministic ratio functions. |
| 36 | [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | Persistent skill learning is an interesting extension; deferred to avoid stale financial memories. |
| 37 | [The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey](https://arxiv.org/abs/2404.11584) | Architecture survey helps compare workflows and autonomous agents; Verity uses explicit workflow control. |
| 38 | [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406) | Tree-search planning is deferred because branching cost and scoring need a measured justification. |
| 39 | [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789) | Large tool-space learning is outside current scope; Verity uses a small fixed tool set. |
| 40 | [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | Judge limitations motivate strict verdict parsing and treating scores as uncalibrated. |
| 41 | [FEVER: a large-scale dataset for Fact Extraction and VERification](https://arxiv.org/abs/1803.05355) | Evidence-labeled fact checking motivates support/abstention semantics and source separation. |
| 42 | [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974) | Scientific claim verification motivates retained evidence; scientific-domain scores do not validate finance. |
| 43 | [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) | RAG evaluation motivates separate retrieval/faithfulness metrics; no Ragas score is claimed. |
| 44 | [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) | Graph representation motivates a small inspectable orchestrator; graph optimization is not implemented. |
| 45 | [FinRobot: An Open-Source AI Agent Platform for Financial Applications using Large Language Models](https://arxiv.org/abs/2405.14767) | Financial role specialization informs retrieval/analysis/report separation. |
| 46 | [FinRobot: AI Agent for Equity Research and Valuation with Large Language Models](https://arxiv.org/abs/2411.08804) | Quantitative and qualitative analysis motivate analyst plus skeptical review, without autonomous valuation claims. |
| 47 | [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138) | Opposing financial perspectives motivate a skeptic; trading actions and backtest-performance claims are excluded. |
| 48 | [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) | Memory architectures inform retained session records; behavioral simulation and long-term belief memory are deferred. |
| 49 | [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/abs/2309.07864) | Agent component taxonomy informs tools/state/roles separation; no general autonomous-agent capability claim. |
| 50 | [CRAG -- Comprehensive RAG Benchmark](https://arxiv.org/abs/2406.04744) | Dynamic factual evaluation motivates time-sensitive financial holdouts and explicit uncertainty. |

## Primary technical and design sources

| Source | Use |
|---|---|
| [physics-intern Space](https://huggingface.co/spaces/huggingface/physics-intern) | User-supplied screenshot and Space: session log beside agent graph, adapted into an original editorial observatory layout. The Space file-tree request failed; no repository code was copied. |
| [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) | Framework comparison. Removing it trades built-in durable orchestration for a smaller project-owned runtime. |
| [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | Prefer explicit, composable workflows for well-defined tasks. |
| [Gemini text generation](https://ai.google.dev/gemini-api/docs/text-generation) | Pass system instructions through the provider configuration, separate from evidence. |
| [Streamlit fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment) | Poll live events without a blocking two-minute loop. |
| [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Primary filings and company facts remain the data foundation. |

## What should be measured next

Use a human-labeled set of claims from held-out companies and periods. Include numeric errors, negation, missing documents, source collisions, stale periods, table extraction, and derived ratios. Compare (a) writer without revision, (b) writer plus verifier, (c) the new skeptic plus bounded revision workflow. Hold model/version, retrieved documents, temperature, and claim budget constant. Record supported-claim precision, unsupported-claim acceptance, abstention, citation recall, review rounds, latency, tokens, and provider cost. Report uncertainty intervals and failures, not only averages.

Runtime tests establish state-machine behavior, not financial accuracy. The synthetic example establishes that the rules catch one numeric mismatch and accept a corrected exact match. It does not establish neural calibration, live SEC availability, model quality, or investment performance.
