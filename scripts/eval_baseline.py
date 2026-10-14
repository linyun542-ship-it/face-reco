"""Evaluate VGGFace2-pretrained embeddings without CelebA fine-tuning."""
import argparse
import json
from pathlib import Path

from face_reco.data.dataset import make_loader
from face_reco.device import resolve_device
from face_reco.metrics import extract_embeddings, tpr_at_fpr
from face_reco.models import FaceEmbedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = FaceEmbedder(pretrained="vggface2").to(device).eval()
    loader = make_loader(args.manifest_dir / "test.csv", 64, 160, train=False,
                         num_workers=args.num_workers, shuffle=False)
    embeddings, labels = extract_embeddings(model, loader, device)
    result = {"loss": "pretrained_baseline", "fine_tuned": False, "seed": args.seed,
              "manifest_dir": str(args.manifest_dir),
              "verification": tpr_at_fpr(embeddings, labels, num_pairs=args.num_pairs, seed=args.seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
