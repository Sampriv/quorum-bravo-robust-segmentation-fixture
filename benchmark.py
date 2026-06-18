#!/usr/bin/env python3
import importlib.util, json, pathlib

root = pathlib.Path(__file__).resolve().parent
data = json.loads((root / "fixtures" / "labels.json").read_text())
spec = importlib.util.spec_from_file_location("predict", root / "predict.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def iou_for_class(pred, true, cls):
    inter = sum(1 for p, t in zip(pred, true) if p == cls and t == cls)
    union = sum(1 for p, t in zip(pred, true) if p == cls or t == cls)
    return 1.0 if union == 0 else inter / union

scores = []
for frame in data["frames"]:
    pred = list(mod.predict(frame))
    true = frame["labels"]
    if len(pred) != len(true):
        raise SystemExit("prediction length mismatch for " + frame["id"])
    scores.append(sum(iou_for_class(pred, true, c) for c in range(len(data["classes"]))) / len(data["classes"]))

miou = sum(scores) / len(scores)
out = {"score": round(miou, 6), "miou": round(miou, 6), "valid": True, "metric": "fixture_mean_iou", "perFrame": [round(s, 6) for s in scores]}
(root / "score.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, sort_keys=True))
