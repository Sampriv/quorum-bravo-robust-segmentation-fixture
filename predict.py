def predict(frame):
    labels = list(frame["labels"])
    # Baseline: copy labels with one deterministic robustness mistake.
    if frame["id"].endswith("rain"):
        labels[1] = 0
    return labels
