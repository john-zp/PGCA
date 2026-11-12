"""Fetch and hash official pretrained weights without changing the old env."""
import json
import sys
from query_model import HERE
from strong_models import WEIGHTS
from train_one import sha, save_json


def main():
    sys.path.insert(0, str(HERE / "supplement_packages"))
    import catboost
    import torchvision
    records = {}
    for name in ["ResNet50-I", "DenseNet121-I"]:
        weight = WEIGHTS[name]
        state = weight.get_state_dict(progress=False, check_hash=True)
        path = HERE / "pretrained_cache/checkpoints" / weight.url.rsplit("/", 1)[-1]
        records[name] = {"enum": str(weight), "url": weight.url, "sha256": sha(path), "bytes": path.stat().st_size,
                         "state_keys": len(state)}
    records["ResNet50-simple-IFP"] = records["ResNet50-I"]
    save_json(HERE / "pretrained_provenance.json", {"status": "PASS", "weights": records,
        "torchvision": torchvision.__version__, "catboost": catboost.__version__,
        "catboost_installation": str(HERE / "supplement_packages"), "base_environment_modified": False})
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
