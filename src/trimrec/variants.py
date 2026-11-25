"""Fixed interventions in the paper's component and architectural studies."""

from dataclasses import replace

from trimrec.config import ExperimentConfig

VARIANTS = {
    "full": {},
    "no_recency": {
        "objective": {
            "recency_start": 1.0,
            "recency_final": 1.0,
            "recency_anneal_epochs": 0,
        }
    },
    "no_refinement": {
        "model": {"refinement": "none"},
        "objective": {"reranker_aux_weight": 0.0},
    },
    "no_item_graph": {"model": {"use_graph": False}},
    "no_candidate_modal_evidence": {"model": {"candidate_modalities": False}},
    "neutral_rating": {"dataset": {"neutral_feedback": True}},
    "no_time_gap": {"model": {"use_time_gap": False}},
    "similarity_only": {"model": {"refinement": "similarity"}},
    "sa": {"model": {"use_return_token": False}},
    "no_transition": {"model": {"use_transition": False}},
    "k1": {"model": {"num_interests": 1}},
    "early_fusion": {
        "model": {"refinement": "none", "early_image_fusion": True},
        "objective": {"reranker_aux_weight": 0.0},
    },
}


def apply_variant(config: ExperimentConfig, name: str) -> ExperimentConfig:
    result = replace(
        config,
        **{
            section: replace(getattr(config, section), **values)
            for section, values in VARIANTS[name].items()
        },
    )
    result.validate()
    return result
