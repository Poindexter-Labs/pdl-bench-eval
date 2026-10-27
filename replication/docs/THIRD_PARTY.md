# Third-party components

The controller, the agent loop and the evaluator are Poindexter's own code,
released under the Apache License 2.0 (`licenses/pdl-bench-eval-LICENSE`). The
agent loop is exported from Poindexter's PD RL Environment repository with the
provenance in `vendor/SOURCE_MANIFEST.json`.

Provider SDKs (OpenAI, Anthropic, Google Gen AI), `jsonschema`, `PyYAML`,
`python-dotenv`, `pytest`, Debian packages, the Docker CLI and the Docker base
images retain their respective licenses; `requirements.lock` enumerates the
Python dependencies with exact versions. Branding this package as Poindexter
does not claim ownership of those components.

The task fixtures are part of the PDL-Bench dataset (CC BY 4.0, see the
dataset card). The dataset licence is separate from the licences of the code
in this package.
