"""Compare two already cropped/aligned faces with a trained checkpoint."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-a", required=True)
    parser.add_argument("--image-b", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, help="Cosine threshold calibrated on independent validation data")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from face_reco.data.dataset import build_transforms
    from face_reco.device import resolve_device
    from face_reco.eval import load_model_from_checkpoint
    if args.threshold is not None and not -1 <= args.threshold <= 1:
        parser.error("--threshold must be between -1 and 1")
    device = resolve_device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    embedder = model.embedder if hasattr(model, "embedder") else model
    transform = build_transforms(int(checkpoint.get("cfg", {}).get("image_size", 160)), train=False)
    faces = []
    for path in (args.image_a, args.image_b):
        with Image.open(path) as image:
            faces.append(transform(image.convert("RGB")))
    with torch.inference_mode():
        embeddings = embedder(torch.stack(faces).to(device))
        similarity = torch.nn.functional.cosine_similarity(embeddings[:1], embeddings[1:]).item()
    result = {"cosine_similarity": similarity, "checkpoint_epoch": checkpoint.get("epoch"),
              "note": "Similarity is not a probability. Inputs must already be cropped/aligned faces."}
    if args.threshold is not None:
        result.update(threshold=args.threshold, accepted=similarity >= args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
