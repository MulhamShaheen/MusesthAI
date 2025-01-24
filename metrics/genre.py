import numpy as np


def top_k_genre_accuracy(y_true, y_pred, k):
    """
    Calculate top@k genre accuracy
    :param y_true: true labels
    :param y_pred: predicted labels
    :param k: top k
    :return: top@k accuracy
    """
    correct = [1 if y_true[i] in y_pred[i] else 0 for i in range(len(y_true))]
    return sum(correct) * k / len(correct)


def map(y_true, y_pred):
    """
    Calculate mean average precision
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: mean average precision
    """

    def apk(actual, predicted, k=10):
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    return np.mean([apk(y_true[i], y_pred[i]) for i in range(len(y_true))])
