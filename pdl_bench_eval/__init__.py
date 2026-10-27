"""pdl-bench-eval: the reference evaluator for PDL-Bench.

Three graders, each a port of the code that produced the published numbers:

* ``repo_grader``   — repository tasks (``software_delivery`` executable tasks and
                      ``cyber_operations``): apply a patch, run the public, build and
                      hidden suites in the pinned image, score under ``scoring_v0.2``.
* ``sweqna_grader`` — question-answering-over-a-repository tasks: property gate plus
                      an LLM judge panel deciding rubric checkpoints (``scoring_llm_v0.3``).
* ``qa_grader``     — single-turn Q&A tasks (STEM, Business Operations, Life
                      Administration): final-answer match and/or rubric grading.
"""
__version__ = "0.1.0"
EVALUATOR_ID = f"pdl-bench-eval/{__version__}"
