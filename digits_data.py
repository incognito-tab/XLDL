"""Shared Digits split for the Mixer-SNN and Dual Memory Pathways runs."""


def load_digits_split(seed=42):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    return train_test_split(
        digits.images,
        digits.target,
        test_size=0.2,
        random_state=seed,
        stratify=digits.target,
    )
