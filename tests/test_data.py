from trimrec.data import TrajectoryDataset, load_corpus, rating_to_reward


def test_strict_split_counts(sequence_file):
    corpus = load_corpus(sequence_file)
    assert corpus.user_count == 2
    assert corpus.item_count == 5
    assert corpus.split_counts == {"train": 3, "validation": 2, "test": 2}
    assert len(TrajectoryDataset(corpus, 5, "train")) == 3
    assert len(TrajectoryDataset(corpus, 5, "validation")) == 2
    assert len(TrajectoryDataset(corpus, 5, "test")) == 2


def test_validation_target_and_padding(sequence_file):
    corpus = load_corpus(sequence_file)
    sample = TrajectoryDataset(corpus, 5, "validation")[0]
    length = int(sample["attention_mask"].sum())
    assert sample["states"][:length].tolist() == [1, 2, 3]
    assert sample["actions"][:length].tolist() == [2, 3, 4]
    assert sample["states"][length:].eq(0).all()


def test_reward_mapping():
    assert rating_to_reward(5.0) == 1.0
    assert rating_to_reward(3.0) == 0.5
    assert rating_to_reward(2.0) == 0.0
    assert rating_to_reward(1.0) == -0.2
