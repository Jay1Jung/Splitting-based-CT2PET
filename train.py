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
from losses import CT2PETLoss
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

    return HecktorCTPETDataset(
        data_root=data_cfg["root"],
        split_file=Path(data_cfg["split_dir"]) / f"{split}.txt",
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


def build_loss(config):
    loss_cfg = config["loss"]

    return CT2PETLoss(
        expected_weight=loss_cfg["expected_weight"],
        deviation_weight=loss_cfg["deviation_weight"],
        gate_weight=loss_cfg["gate_weight"],
        gradient_weight=loss_cfg["gradient_weight"],
        lesion_weight=loss_cfg["lesion_weight"],
        blur_kernel=loss_cfg["blur_kernel"],
        blur_sigma=loss_cfg["blur_sigma"],
        gate_dilation=loss_cfg["gate_dilation"],
        gate_sigma=loss_cfg["gate_sigma"],
    )


def load_expected_checkpoint(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.expected_encoder.load_state_dict(
        checkpoint["expected_encoder"]
    )

    model.expected_decoder.load_state_dict(
        checkpoint["expected_decoder"]
    )

    print(f"loaded expected checkpoint: {path}")


def make_optimizer(model, config):
    training_cfg = config["training"]

    full_lr = training_cfg["full_lr"]

    expected_lr_scale = training_cfg.get(
        "expected_lr_scale",
        0.25,
    )

    expected_params = (
        list(model.expected_encoder.parameters())
        + list(model.expected_decoder.parameters())
    )

    expected_ids = {
        id(parameter)
        for parameter in expected_params
    }

    other_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in expected_ids
    ]

    return AdamW(
        [
            {
                "params": expected_params,
                "lr": full_lr * expected_lr_scale,
            },
            {
                "params": other_params,
                "lr": full_lr,
            },
        ],
        weight_decay=training_cfg["weight_decay"],
    )


def move_batch(batch, device):
    return (
        batch["axial_ct"].to(device),
        batch["coronal_ct"].to(device),
        batch["sagittal_ct"].to(device),
        batch["pet"].to(device),
        batch["tumor_mask"].to(device),
    )


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    running = {
        "loss": 0.0,
        "final": 0.0,
        "expected": 0.0,
        "deviation": 0.0,
        "gate": 0.0,
        "gradient": 0.0,
    }

    total_samples = 0

    progress = tqdm(
        loader,
        desc="train full",
    )

    for batch in progress:
        (
            axial_ct,
            coronal_ct,
            sagittal_ct,
            pet,
            tumor_mask,
        ) = move_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            axial_ct,
            coronal_ct,
            sagittal_ct,
        )

        losses = criterion(
            outputs,
            pet,
            tumor_mask,
        )

        losses["loss"].backward()

        optimizer.step()

        batch_size = axial_ct.shape[0]
        total_samples += batch_size

        for key in running:
            running[key] += (
                losses[key].item()
                * batch_size
            )

        progress.set_postfix(
            loss=f"{losses['loss'].item():.4f}",
            final=f"{losses['final'].item():.4f}",
            dev=f"{losses['deviation'].item():.4f}",
        )

    for key in running:
        running[key] /= total_samples

    return running


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running = {
        "loss": 0.0,
        "final": 0.0,
        "expected": 0.0,
        "deviation": 0.0,
        "gate": 0.0,
        "gradient": 0.0,
    }

    total_samples = 0

    for batch in tqdm(
        loader,
        desc="val full",
    ):
        (
            axial_ct,
            coronal_ct,
            sagittal_ct,
            pet,
            tumor_mask,
        ) = move_batch(
            batch,
            device,
        )

        outputs = model(
            axial_ct,
            coronal_ct,
            sagittal_ct,
        )

        losses = criterion(
            outputs,
            pet,
            tumor_mask,
        )

        batch_size = axial_ct.shape[0]
        total_samples += batch_size

        for key in running:
            running[key] += (
                losses[key].item()
                * batch_size
            )

    for key in running:
        running[key] /= total_samples

    return running


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    path,
):
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def print_losses(name, losses):
    print(
        f"{name}: "
        f"total={losses['loss']:.5f} | "
        f"final={losses['final']:.5f} | "
        f"expected={losses['expected']:.5f} | "
        f"deviation={losses['deviation']:.5f} | "
        f"gate={losses['gate']:.5f} | "
        f"gradient={losses['gradient']:.5f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/prototype.yaml",
    )

    parser.add_argument(
        "--expected_checkpoint",
        default=None,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    set_seed(
        config["training"]["seed"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device: {device}")

    train_dataset = make_dataset(
        config,
        "train",
    )

    val_dataset = make_dataset(
        config,
        "val",
    )

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

    model = build_model(
        config
    ).to(device)

    checkpoint_dir = Path(
        config["training"]["checkpoint_dir"]
    )

    expected_checkpoint = (
        Path(args.expected_checkpoint)
        if args.expected_checkpoint
        else checkpoint_dir / "expected_best.pt"
    )

    if not expected_checkpoint.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found: "
            f"{expected_checkpoint}"
        )

    load_expected_checkpoint(
        model,
        expected_checkpoint,
        device,
    )

    criterion = build_loss(
        config
    )

    optimizer = make_optimizer(
        model,
        config,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    epochs = config["training"]["full_epochs"]

    for epoch in range(1, epochs + 1):
        print(
            f"\nepoch {epoch}/{epochs}"
        )

        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_losses = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print_losses(
            "train",
            train_losses,
        )

        print_losses(
            "val  ",
            val_losses,
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_losses["loss"],
            checkpoint_dir / "full_last.pt",
        )

        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_losses["loss"],
                checkpoint_dir / "full_best.pt",
            )

            print("saved full_best.pt")


if __name__ == "__main__":
    main()