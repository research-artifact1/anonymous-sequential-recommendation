"""Extract item-aligned OpenAI CLIP ViT-B/32 text and image features."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterator, Mapping, Sequence, Tuple

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_records(path: Path) -> Iterator[Mapping[str, object]]:
    with open_text(path) as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = ast.literal_eval(line)
            if isinstance(value, Mapping):
                yield value


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(filter(None, (flatten_text(part) for part in value)))
    return str(value).strip()


def item_text(record: Mapping[str, object], asin: str) -> str:
    fields = (
        flatten_text(record.get("title")),
        flatten_text(record.get("brand")),
        flatten_text(record.get("categories")),
        flatten_text(record.get("feature") or record.get("features")),
        flatten_text(record.get("description")),
    )
    text = ". ".join(part for part in fields if part)
    return text or asin


class ImageDataset(Dataset[Tuple[int, torch.Tensor]]):
    def __init__(self, rows: Sequence[Tuple[int, Path]], preprocess) -> None:
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        item_id, path = self.rows[index]
        with Image.open(path) as image:
            return item_id, self.preprocess(image.convert("RGB"))


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--image-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    with args.sequences.open("rb") as handle:
        sequence_payload = pickle.load(handle)
    itemmap: Dict[str, int] = sequence_payload["itemmap"]
    item_count = len(itemmap)

    metadata: Dict[str, Mapping[str, object]] = {}
    for record in iter_records(args.metadata):
        asin = record.get("asin") or record.get("parent_asin")
        if asin is not None and str(asin) in itemmap:
            metadata[str(asin)] = record

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(device)
    output_dim = int(model.text_projection.shape[-1])

    texts = [""] * (item_count + 1)
    image_rows = []
    for asin, item_id in sorted(itemmap.items(), key=lambda pair: pair[1]):
        texts[item_id] = item_text(metadata.get(asin, {}), asin)
        image_path = args.image_dir / f"{asin}.jpg"
        if image_path.is_file():
            image_rows.append((item_id, image_path))

    text_features = torch.zeros(item_count + 1, output_dim)
    for start in tqdm(range(1, item_count + 1, args.batch_size), desc="CLIP text"):
        item_ids = list(range(start, min(start + args.batch_size, item_count + 1)))
        tokens = tokenizer([texts[item_id] for item_id in item_ids]).to(device)
        with torch.inference_mode(), autocast(device, args.amp):
            encoded = F.normalize(model.encode_text(tokens).float(), dim=-1)
        text_features[item_ids] = encoded.cpu()

    image_features = torch.zeros(item_count + 1, output_dim)
    loader = DataLoader(
        ImageDataset(image_rows, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    for item_ids, images in tqdm(loader, desc="CLIP image"):
        with torch.inference_mode(), autocast(device, args.amp):
            encoded = F.normalize(
                model.encode_image(images.to(device, non_blocking=True)).float(),
                dim=-1,
            )
        image_features[item_ids.long()] = encoded.cpu()

    save_tensor(text_features, args.text_output)
    save_tensor(image_features, args.image_output)
    print(
        json.dumps(
            {
                "items": item_count,
                "text_coverage": item_count,
                "image_coverage": len(image_rows),
                "text_output": str(args.text_output.resolve()),
                "image_output": str(args.image_output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
