"""Tests for Rust-accelerated classification — NLTK compatibility."""


from fastnltk.classify import NaiveBayesClassifier


class TestNaiveBayesClassifier:
    def test_train_and_classify_basic(self):
        train_data = [
            ({"feature": "good"}, "pos"),
            ({"feature": "good"}, "pos"),
            ({"feature": "bad"}, "neg"),
            ({"feature": "bad"}, "neg"),
            ({"feature": "good"}, "pos"),
        ]
        classifier = NaiveBayesClassifier.train(train_data)
        result = classifier.classify({"feature": "good"})
        assert result == "pos"
        result = classifier.classify({"feature": "bad"})
        assert result == "neg"

    def test_labels(self):
        train_data = [
            ({"f": "a"}, "X"),
            ({"f": "b"}, "Y"),
        ]
        classifier = NaiveBayesClassifier.train(train_data)
        labels = set(classifier.labels())
        assert labels == {"X", "Y"}

    def test_prob_classify(self):
        train_data = [
            ({"f": "a"}, "X"),
            ({"f": "a"}, "X"),
            ({"f": "b"}, "Y"),
        ]
        classifier = NaiveBayesClassifier.train(train_data)
        probs = classifier.prob_classify({"f": "a"})
        assert isinstance(probs, dict)
        assert "X" in probs

    def test_show_most_informative_features(self):
        train_data = [
            ({"f": "a"}, "X"),
            ({"f": "b"}, "Y"),
        ]
        classifier = NaiveBayesClassifier.train(train_data)
        # Should not crash
        features = classifier.show_most_informative_features(5)
        assert isinstance(features, (list, str))
