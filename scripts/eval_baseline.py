"""Evaluate VGGFace2-pretrained embeddings without CelebA fine-tuning."""
import argparse
import json
import csv
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from face_reco.data.dataset import make_loader
    from face_reco.device import resolve_device
    from face_reco.metrics import extract_embeddings, tpr_at_fpr
    from face_reco.models import FaceEmbedder

    if args.num_pairs < 2 or args.num_workers < 0:
        parser.error("num-pairs must be >=2 and num-workers must be nonnegative")
    identities = {}
    hashes = {}
    for split in ("train", "val", "test"):
        path = args.manifest_dir / f"{split}.csv"
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as stream:
            identities[split] = {int(row["identity"]) for row in csv.DictReader(stream)}
    meta = json.loads((args.manifest_dir / "meta.json").read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    model = FaceEmbedder(pretrained="vggface2").to(device).eval()
    loader = make_loader(args.manifest_dir / "test.csv", 64, 160, train=False,
                         num_workers=args.num_workers, shuffle=False)
    embeddings, labels = extract_embeddings(model, loader, device)
    result = {"loss": "pretrained_baseline", "fine_tuned": False, "seed": args.seed,
              "manifest_dir": str(args.manifest_dir),
              "data_meta": meta, "manifest_sha256": hashes,
              "identity_protocol": {
                  "train_identities": len(identities["train"]),
                  "val_identities": len(identities["val"]),
                  "test_identities": len(identities["test"]),
                  "train_val_overlap": len(identities["train"] & identities["val"]),
                  "train_test_overlap": len(identities["train"] & identities["test"]),
              },
              "verification": tpr_at_fpr(embeddings, labels, num_pairs=args.num_pairs, seed=args.seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(json.dumps({"ok": True, "summary": str(args.output)}))


if __name__ == "__main__":
    main()
