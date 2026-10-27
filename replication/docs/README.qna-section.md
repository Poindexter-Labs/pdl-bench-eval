| question answering over a repository | @@QNA@@ | `agent_prompt_qa_v0.1` | `python_qa_2m_v0.1` | `scoring_llm_v0.3`: rubric checkpoints decided by a three-member judge panel (per-item median), weighted mean | reward > 0.95 |

The question-answering tasks are judged, not tested: the hidden suite is a
property gate (file present, headings, size) that must collect but never
scores, and the reward is the panel's checkpoint verdicts over
`answers/answer.md`. The reference panel is `gemini-3.1-pro-preview` (Vertex
AI), `claude-opus-5` (Anthropic API) and `gpt-5.6-sol` (Azure AI Foundry v1
surface); set `OPENAI_BASE_URL` to an endpoint serving `gpt-5.6-sol`, or name
any panel you can reach with `PDL_JUDGES=provider:model[@base_url],...`. The
panel used is recorded on every graded attempt, and a panel you choose is a
different configuration from the reference one. A judge that cannot be
reached leaves the attempt ungraded (`grade` retries it); it never scores 0.
