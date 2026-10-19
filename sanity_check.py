import argparse

import torch
import yaml

from models import CT2PETModel
from losses import CT2PETLoss, gaussian_blur


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_shape(name, tensor):
    print(f"{name:<22} {list(tensor.shape)}")


def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def check_gradients(model):
    modules = {
        "expected_encoder": model.expected_encoder,
        "orthogonal_encoder": model.orthogonal_encoder,
        "deviation_encoder": model.deviation_encoder,
        "gate": model.gate,
        "glue": model.glue,
        "decoder": model.decoder,
        "expected_decoder": model.expected_decoder,
        "deviation_decoder": model.deviation_decoder,
    }

    print("\nGradients")

    for name, module in modules.items():
        grads = [
            p.grad.abs().mean().item()
            for p in module.parameters()
            if p.grad is not None
        ]

        if grads:
            mean_grad = sum(grads) / len(grads)
            print(f"{name:<22} {mean_grad:.8f}")
        else:
            print(f"{name:<22} no gradient")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/prototype.yaml",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device: {device}")

    batch_size = args.batch_size
    image_size = 96
    num_views = data_cfg["num_orthogonal_slices"]

    model = CT2PETModel(
        num_orthogonal_slices=num_views,
        base_channels=model_cfg["base_channels"],
        latent_channels=model_cfg["latent_channels"],
        context_dim=model_cfg["context_dim"],
        gate_channels=model_cfg["gate_channels"],
    ).to(device)

    criterion = CT2PETLoss(
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

    axial_ct = torch.randn(
        batch_size,
        1,
        image_size,
        image_size,
        device=device,
    )

    coronal_ct = torch.randn(
        batch_size,
        num_views,
        image_size,
        image_size,
        device=device,
    )

    sagittal_ct = torch.randn(
        batch_size,
        num_views,
        image_size,
        image_size,
        device=device,
    )

    pet = torch.rand(
        batch_size,
        1,
        image_size,
        image_size,
        device=device,
    )

    tumor_mask = torch.zeros(
        batch_size,
        1,
        image_size,
        image_size,
        device=device,
    )

    tumor_mask[
        :,
        :,
        35:55,
        40:60,
    ] = 1.0

    print("\nInputs")
    print_shape("axial_ct", axial_ct)
    print_shape("coronal_ct", coronal_ct)
    print_shape("sagittal_ct", sagittal_ct)
    print_shape("pet", pet)
    print_shape("tumor_mask", tumor_mask)

    print("\nExpected branch")

    expected_feature = model.expected_encoder(
        axial_ct
    )

    expected_pet = model.expected_decoder(
        expected_feature
    )

    print_shape(
        "expected_feature",
        expected_feature,
    )

    print_shape(
        "expected_pet",
        expected_pet,
    )

    expected_target = gaussian_blur(
        pet,
        kernel_size=loss_cfg["blur_kernel"],
        sigma=loss_cfg["blur_sigma"],
    )

    if expected_pet.shape != pet.shape:
        raise RuntimeError(
            f"Expected PET shape mismatch: "
            f"{expected_pet.shape} vs {pet.shape}"
        )

    if expected_target.shape != pet.shape:
        raise RuntimeError(
            f"Gaussian blur changed image size: "
            f"{expected_target.shape} vs {pet.shape}"
        )

    print("\nFull model")

    outputs = model(
        axial_ct,
        coronal_ct,
        sagittal_ct,
    )

    for key in [
        "pet",
        "expected_pet",
        "deviation_pet",
        "gate",
        "expected_feature",
        "deviation_feature",
        "glued_feature",
        "orthogonal_context",
    ]:
        print_shape(
            key,
            outputs[key],
        )

        check_finite(
            key,
            outputs[key],
        )

    expected_latent_shape = (
        batch_size,
        model_cfg["latent_channels"],
        24,
        24,
    )

    if tuple(outputs["expected_feature"].shape) != expected_latent_shape:
        raise RuntimeError(
            "Unexpected expected feature shape: "
            f"{outputs['expected_feature'].shape}"
        )

    if (
        outputs["expected_feature"].shape
        != outputs["deviation_feature"].shape
    ):
        raise RuntimeError(
            "Expected and deviation features "
            "do not have the same shape"
        )

    if (
        outputs["expected_feature"].shape
        != outputs["glued_feature"].shape
    ):
        raise RuntimeError(
            "Glued feature has the wrong shape"
        )

    if outputs["pet"].shape != pet.shape:
        raise RuntimeError(
            f"PET output shape mismatch: "
            f"{outputs['pet'].shape} vs {pet.shape}"
        )

    if outputs["deviation_pet"].shape != pet.shape:
        raise RuntimeError(
            "Deviation decoder output shape mismatch"
        )

    expected_context_shape = (
        batch_size,
        model_cfg["context_dim"],
    )

    if tuple(outputs["orthogonal_context"].shape) != expected_context_shape:
        raise RuntimeError(
            "Orthogonal context shape mismatch: "
            f"{outputs['orthogonal_context'].shape}"
        )

    if outputs["gate"].min() < 0 or outputs["gate"].max() > 1:
        raise RuntimeError(
            "Gate is outside [0, 1]"
        )

    if outputs["pet"].min() < 0 or outputs["pet"].max() > 1:
        raise RuntimeError(
            "Final PET is outside [0, 1]"
        )

    if not (
        outputs["deviation_pet"].min() < 0
        or outputs["deviation_pet"].max() > 0
    ):
        raise RuntimeError(
            "Deviation decoder appears constant"
        )

    print("\nLoss")

    losses = criterion(
        outputs,
        pet,
        tumor_mask,
    )

    for name, value in losses.items():
        print(
            f"{name:<22} "
            f"{value.item():.6f}"
        )

        check_finite(
            name,
            value,
        )

    print("\nBackward")

    model.zero_grad(set_to_none=True)

    losses["loss"].backward()

    check_gradients(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    optimizer.step()

    print("\nOptimizer step: OK")

    with torch.no_grad():
        outputs_after = model(
            axial_ct,
            coronal_ct,
            sagittal_ct,
        )

        check_finite(
            "pet after optimizer",
            outputs_after["pet"],
        )

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()