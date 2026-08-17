import torch

from trimrec.config import ModelConfig, ObjectiveConfig
from trimrec.data import TrajectoryDataset, load_corpus
from trimrec.features import FrozenFeatures, load_frozen_features
from trimrec.losses import compute_loss
from trimrec.model.trimrec import TRIMRec
from trimrec.structure import build_item_graph, build_transition_table


def test_forward_and_loss_are_finite(sequence_file, feature_tensors):
    torch.manual_seed(11)
    corpus = load_corpus(sequence_file)
    text_path, image_path = feature_tensors
    features = load_frozen_features(
        text_path,
        image_path,
        corpus.item_count,
        text_dim=8,
        image_dim=8,
    )
    graph = build_item_graph(corpus, min_count=1, top_k=3)
    transition_indices, transition_values = build_transition_table(
        corpus, min_count=1, top_k=3
    )
    config = ModelConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        max_timestep=16,
        dropout=0.0,
        gap_buckets=16,
        text_dim=8,
        image_dim=8,
        graph_layers=1,
        num_interests=2,
        reranker_top_m=4,
        reranker_history_length=3,
        reranker_attention_dim=8,
        reranker_hidden_dim=8,
    )
    model = TRIMRec(
        config,
        corpus.item_count,
        features,
        graph,
        transition_indices,
        transition_values,
    )
    samples = TrajectoryDataset(corpus, 5, "train", gap_buckets=16)
    batch = {key: torch.stack([samples[0][key], samples[1][key]]) for key in samples[0]}
    output = model(batch)
    assert output.logits.shape == (2, 5, 6)
    assert output.candidate_ids.shape == (2, 4)
    losses = compute_loss(output, batch, ObjectiveConfig(), epoch=1)
    assert torch.isfinite(losses.loss)
    losses.loss.backward()
    assert model.item_embedding.weight.grad is not None


def test_external_features_are_not_checkpoint_parameters(
    sequence_file, feature_tensors
):
    corpus = load_corpus(sequence_file)
    features = load_frozen_features(
        *feature_tensors, corpus.item_count, text_dim=8, image_dim=8
    )
    graph = build_item_graph(corpus, min_count=1)
    transition = build_transition_table(corpus, min_count=1, top_k=3)
    model = TRIMRec(
        ModelConfig(
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
            text_dim=8,
            image_dim=8,
            num_interests=2,
            reranker_top_m=4,
            reranker_history_length=3,
            reranker_attention_dim=8,
            reranker_hidden_dim=8,
        ),
        corpus.item_count,
        features,
        graph,
        *transition,
    )
    state = model.state_dict()
    assert "text_features" not in state
    assert "image_features" not in state
    assert "graph_adjacency" not in state


def test_reported_parameter_count_formula():
    item_count = 5
    catalog_size = item_count + 1
    features = FrozenFeatures(
        text=torch.zeros(catalog_size, 1, 512),
        image=torch.zeros(catalog_size, 512),
    )
    diagonal = torch.arange(catalog_size)
    graph = torch.sparse_coo_tensor(
        torch.stack((diagonal, diagonal)),
        torch.ones(catalog_size),
        (catalog_size, catalog_size),
        check_invariants=False,
    ).coalesce()
    transition_indices = torch.zeros(catalog_size, 20, dtype=torch.long)
    transition_values = torch.zeros(catalog_size, 20)
    model = TRIMRec(
        ModelConfig(),
        item_count,
        features,
        graph,
        transition_indices,
        transition_values,
    )
    assert model.parameter_count() == 1_109_318 + catalog_size * 129
