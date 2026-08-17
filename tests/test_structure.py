import torch

from trimrec.data import load_corpus
from trimrec.structure import build_item_graph, build_transition_table


def test_structures_exclude_validation_and_test(sequence_file):
    corpus = load_corpus(sequence_file)
    graph = build_item_graph(corpus, exclude_last=2, min_count=1, top_k=20)
    dense = graph.to_dense()
    assert dense[1, 2] > 0
    assert dense[2, 3] > 0
    assert dense[3, 4] == 0

    indices, values = build_transition_table(
        corpus, exclude_last=2, min_count=1, top_k=3
    )
    assert 3 not in indices[3].tolist()
    assert 4 not in indices[3].tolist()
    assert torch.all(values >= 0)
