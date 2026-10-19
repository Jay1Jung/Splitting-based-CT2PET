import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import HecktorCTPETDataset
from losses import gaussian_blur, make_soft_gate_target
from models import CT2PETModel


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config, split):
    data_cfg = config["data"]

    split_file = Path(data_cfg["split_dir"]) / f"{split}.txt"

    return HecktorCTPETDataset(
        data_root=data_cfg["root"],
        split_file=split_file,
        context_depth=data_cfg["context_depth"],
        num_orthogonal_slices=data_cfg["num_orthogonal_slices"],
        ct_min=data_cfg["ct_min"],
        ct_max=data_cfg["ct_max"],
        pet_max=data_cfg["pet_max"],
        include_empty_slices=data_cfg["include_empty_slices"],
    )


def make_loader(dataset, config, shuffle):
    train_cfg = config["training"]

    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train_cfg["num_workers"] > 0,
    )


def build_model(config):
    model_cfg = config["model"]
    data_cfg = config["data"]

    return CT2PETModel(
        num_orthogonal_slices=data_cfg["num_orthogonal_slices"],
        base_channels=model_cfg["base_channels"],
        latent_channels=model_cfg["latent_channels"],
        context_dim=model_cfg["context_dim"],
        gate_channels=model_cfg["gate_channels"],
    )


def expected_loss(prediction, pet, tumor_mask, config):
    loss_cfg = config["loss"]

    target = gaussian_blur(
        pet,
        kernel_size=loss_cfg["blur_kernel"],
        sigma=loss_cfg["blur_sigma"],
    )

    gate_target = make_soft_gate_target(
        tumor_mask,
        dilation=loss_cfg["gate_dilation"],
        sigma=loss_cfg["gate_sigma"],
    )

    weights = 1.0 - gate_target + 0.05

    error = torch.abs(prediction - target) * weights

    return error.sum() / (weights.sum() + 1e-8)


def train_epoch(model, loader, optimizer, device, config):
    model.train()

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc="train expected")

    for batch in progress:
        axial_ct = batch["axial_ct"].to(device)
        pet = batch["pet"].to(device)
        tumor_mask = batch["tumor_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        expected_feature = model.expected_encoder(axial_ct)
        expected_pet = model.expected_decoder(expected_feature)

        loss = expected_loss(
            expected_pet,
            pet,
            tumor_mask,
            config,
        )

        loss.backward()
        optimizer.step()

        batch_size = axial_ct.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / total_samples


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="val expected"):
        axial_ct = batch["axial_ct"].to(device)
        pet = batch["pet"].to(device)
        tumor_mask = batch["tumor_mask"].to(device)

        expected_feature = model.expected_encoder(axial_ct)
        expected_pet = model.expected_decoder(expected_feature)

        loss = expected_loss(
            expected_pet,
            pet,
            tumor_mask,
            config,
        )

        batch_size = axial_ct.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "expected_encoder": model.expected_encoder.state_dict(),
            "expected_decoder": model.expected_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/prototype.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    seed = config["training"]["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device: {device}")

    train_dataset = make_dataset(config, "train")
    val_dataset = make_dataset(config, "val")

    train_loader = make_loader(
        train_dataset,
        config,
        shuffle=True,
    )

    val_loader = make_loader(
        val_dataset,
        config,
        shuffle=False,
    )

    model = build_model(config).to(device)

    parameters = list(model.expected_encoder.parameters())
    parameters += list(model.expected_decoder.parameters())

    optimizer = AdamW(
        parameters,
        lr=config["training"]["expected_lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    checkpoint_dir = Path(
        config["training"]["checkpoint_dir"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    epochs = config["training"]["expected_epochs"]

    for epoch in range(1, epochs + 1):
        print(f"\nepoch {epoch}/{epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
        )

        val_loss = validate(
            model,
            val_loader,
            device,
            config,
        )

        print(f"train loss: {train_loss:.5f}")
        print(f"val loss:   {val_loss:.5f}")

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            checkpoint_dir / "expected_last.pt",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                checkpoint_dir / "expected_best.pt",
            )

            print("saved expected_best.pt")


if __name__ == "__main__":
    main()