# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
"""Model-provider adapters (DEC-040): one shared agent loop, thin per-provider
adapters. `make_model_agent` builds a ModelAgent for a given provider family
and endpoint."""

from __future__ import annotations

from pdl_bench_eval._vendor.pd_rl.providers.base import ModelAgent

MODEL_AGENTS = ("gemini", "openai", "anthropic", "maas")


def make_model_agent(name, model=None, prompt=None, endpoint=None):
    """Build a ModelAgent.

    name: provider family — gemini | openai | anthropic.
    endpoint: optional endpoint selector —
      gemini + 'vertex[:loc]'    -> Vertex AI (default europe-west2,
                                    sovereign; 'vertex:global' for preview
                                    models that only serve the global
                                    endpoint — non-sovereign, GCP billing,
                                    labeled google-vertex:global)
      openai + 'azure'           -> Azure OpenAI (UK South, sovereign)
      openai + '<https url>'     -> OpenAI-compatible host at that base_url
      anthropic + 'vertex'       -> Claude on Vertex AI (GCP billing, ADC;
                                    region from ANTHROPIC_VERTEX_REGION)
      maas + '<region>'          -> Vertex MaaS partner model over rawPredict
                                    (model = <publisher>/<model>, DEC-045)
    """
    if name == "gemini":
        from pdl_bench_eval._vendor.pd_rl.providers.google import GoogleAdapter

        vertex = bool(endpoint) and endpoint.split(":", 1)[0] == "vertex"
        location = "europe-west2"
        if vertex and ":" in endpoint:
            # Lowercase matters: the SDK special-cases exactly "global".
            location = endpoint.split(":", 1)[1].strip().lower() or location
        adapter = GoogleAdapter(model_id=model, vertex=vertex, location=location)
    elif name == "openai":
        from pdl_bench_eval._vendor.pd_rl.providers.openai_provider import OpenAIAdapter

        azure = endpoint == "azure"
        base_url = endpoint if (endpoint and not azure) else None
        adapter = OpenAIAdapter(model_id=model, base_url=base_url, azure=azure)
    elif name == "anthropic":
        from pdl_bench_eval._vendor.pd_rl.providers.anthropic_provider import AnthropicAdapter

        if endpoint not in (None, "vertex"):
            raise ValueError(
                "anthropic supports --endpoint vertex only (got %r)" % endpoint
            )
        adapter = AnthropicAdapter(model_id=model, vertex=(endpoint == "vertex"))
    elif name == "maas":
        from pdl_bench_eval._vendor.pd_rl.providers.vertex_maas import VertexMaaSAdapter

        adapter = VertexMaaSAdapter(model_id=model, region=endpoint)
    else:
        raise ValueError(f"unknown model provider '{name}' (available: {', '.join(MODEL_AGENTS)})")
    return ModelAgent(adapter, prompt_id=prompt)
