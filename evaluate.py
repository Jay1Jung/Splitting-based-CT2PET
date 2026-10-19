import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from data import HecktorCTPETDataset
from models import CT2PETModel
from utils import (
    save_prediction_figure,
    save_branch_figure,
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_dataset(config):
    data_cfg = config["data"]

    return HecktorCTPETDataset(
        data_root=data_cfg["root"],
        split_file=Path(data_cfg["split_dir"]) / "test.txt",
        context_depth=data_cfg["context_depth"],
        num_orthogonal_slices=data_cfg["num_orthogonal_slices"],
        ct_min=data_cfg["ct_min"],
        ct_max=data_cfg["ct_max"],
        pet_max=data_cfg["pet_max"],
        include_empty_slices=data_cfg["include_empty_slices"],
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/prototype.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device: {device}")

    dataset = make_dataset(
        config
    )

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["training"]["num_workers"] > 0,
    )

    model = build_model(
        config
    ).to(device)

    checkpoint_dir = Path(
        config["training"]["checkpoint_dir"]
    )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else checkpoint_dir / "full_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    output_dir = Path(
        config["training"]["output_dir"]
    )

    figure_dir = (
        output_dir
        / "test_figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    abs_error_sum = 0.0
    squared_error_sum = 0.0
    voxel_count = 0

    lesion_error_sum = 0.0
    lesion_voxel_count = 0

    nonlesion_error_sum = 0.0
    nonlesion_voxel_count = 0

    gradient_error_sum = 0.0
    gradient_count = 0

    peak_errors = []

    saved_figures = 0

    pet_max = config["data"]["pet_max"]

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="test",
        ):
            axial_ct = batch[
                "axial_ct"
            ].to(device)

            coronal_ct = batch[
                "coronal_ct"
            ].to(device)

            sagittal_ct = batch[
                "sagittal_ct"
            ].to(device)

            pet = batch[
                "pet"
            ].to(device)

            tumor_mask = batch[
                "tumor_mask"
            ].to(device)

            outputs = model(
                axial_ct,
                coronal_ct,
                sagittal_ct,
            )

            prediction = outputs[
                "pet"
            ]

            error = torch.abs(
                prediction - pet
            )

            squared_error = (
                prediction - pet
            ) ** 2

            abs_error_sum += (
                error.sum().item()
            )

            squared_error_sum += (
                squared_error.sum().item()
            )

            voxel_count += (
                pet.numel()
            )

            lesion = (
                tumor_mask > 0.5
            )

            nonlesion = ~lesion

            if lesion.any():
                lesion_error_sum += (
                    error[lesion]
                    .sum()
                    .item()
                )

                lesion_voxel_count += (
                    lesion.sum().item()
                )

            if nonlesion.any():
                nonlesion_error_sum += (
                    error[nonlesion]
                    .sum()
                    .item()
                )

                nonlesion_voxel_count += (
                    nonlesion.sum().item()
                )

            pred_dx = (
                prediction[:, :, :, 1:]
                - prediction[:, :, :, :-1]
            )

            pred_dy = (
                prediction[:, :, 1:, :]
                - prediction[:, :, :-1, :]
            )

            pet_dx = (
                pet[:, :, :, 1:]
                - pet[:, :, :, :-1]
            )

            pet_dy = (
                pet[:, :, 1:, :]
                - pet[:, :, :-1, :]
            )

            gradient_error_sum += (
                torch.abs(
                    pred_dx - pet_dx
                ).sum().item()
            )

            gradient_error_sum += (
                torch.abs(
                    pred_dy - pet_dy
                ).sum().item()
            )

            gradient_count += (
                pred_dx.numel()
                + pred_dy.numel()
            )

            for i in range(
                pet.shape[0]
            ):
                current_mask = (
                    tumor_mask[i] > 0.5
                )

                if current_mask.any():
                    gt_peak = (
                        pet[i][
                            current_mask
                        ].max()
                    )

                    pred_peak = (
                        prediction[i][
                            current_mask
                        ].max()
                    )

                    peak_errors.append(
                        abs(
                            pred_peak.item()
                            - gt_peak.item()
                        )
                    )

                if (
                    saved_figures
                    < args.num_visualizations
                    and batch["has_lesion"][i]
                ):
                    patient_id = batch[
                        "patient_id"
                    ][i]

                    z = int(
                        batch["z"][i]
                    )

                    prefix = (
                        f"{patient_id}_z{z}"
                    )

                    save_prediction_figure(
                        axial_ct=axial_ct[i],
                        target_pet=pet[i],
                        prediction=prediction[i],
                        tumor_mask=tumor_mask[i],
                        gate=outputs["gate"][i],
                        save_path=(
                            figure_dir
                            / f"{prefix}_prediction.png"
                        ),
                    )

                    save_branch_figure(
                        target_pet=pet[i],
                        expected_pet=outputs[
                            "expected_pet"
                        ][i],
                        deviation_pet=outputs[
                            "deviation_pet"
                        ][i],
                        prediction=prediction[i],
                        gate=outputs["gate"][i],
                        save_path=(
                            figure_dir
                            / f"{prefix}_branches.png"
                        ),
                    )

                    saved_figures += 1

    mae = (
        abs_error_sum
        / voxel_count
    )

    mse = (
        squared_error_sum
        / voxel_count
    )

    rmse = math.sqrt(
        mse
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = (
            10
            * math.log10(
                1.0 / mse
            )
        )

    lesion_mae = (
        lesion_error_sum
        / lesion_voxel_count
        if lesion_voxel_count > 0
        else None
    )

    nonlesion_mae = (
        nonlesion_error_sum
        / nonlesion_voxel_count
        if nonlesion_voxel_count > 0
        else None
    )

    gradient_mae = (
        gradient_error_sum
        / gradient_count
    )

    peak_error = (
        sum(peak_errors)
        / len(peak_errors)
        if peak_errors
        else None
    )

    peak_error_suv = (
        peak_error * pet_max
        if peak_error is not None
        else None
    )

    results = {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "lesion_mae": lesion_mae,
        "nonlesion_mae": nonlesion_mae,
        "gradient_mae": gradient_mae,
        "lesion_peak_error": peak_error,
        "lesion_peak_error_suv": peak_error_suv,
        "num_test_slices": len(dataset),
        "num_lesion_slices": len(peak_errors),
    }

    print("\nTest results")

    for name, value in results.items():
        if isinstance(value, float):
            print(
                f"{name}: {value:.6f}"
            )
        else:
            print(
                f"{name}: {value}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "test_metrics.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\nresults saved to {output_dir}"
    )


if __name__ == "__main__":
    main()