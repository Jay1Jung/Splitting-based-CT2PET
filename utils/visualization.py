from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def to_numpy(tensor):
    return tensor.detach().cpu().squeeze().numpy()


def resize_gate(gate, output_size):
    if gate.shape[-2:] == output_size:
        return gate

    return F.interpolate(
        gate,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )


def save_prediction_figure(
    axial_ct,
    target_pet,
    prediction,
    tumor_mask,
    gate,
    save_path,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gate = resize_gate(
        gate,
        target_pet.shape[-2:],
    )

    ct = to_numpy(axial_ct)
    target = to_numpy(target_pet)
    pred = to_numpy(prediction)
    mask = to_numpy(tumor_mask)
    gate = to_numpy(gate)

    error = abs(pred - target)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
    )

    axes[0, 0].imshow(ct, cmap="gray")
    axes[0, 0].set_title("CT")

    axes[0, 1].imshow(target, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title("GT PET")

    axes[0, 2].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title("Predicted PET")

    axes[1, 0].imshow(error, cmap="hot")
    axes[1, 0].set_title("Absolute Error")

    axes[1, 1].imshow(gate, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Predicted Gate")

    axes[1, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Tumor Label")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_branch_figure(
    target_pet,
    expected_pet,
    deviation_pet,
    prediction,
    gate,
    save_path,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gate = resize_gate(
        gate,
        target_pet.shape[-2:],
    )

    target = to_numpy(target_pet)
    expected = to_numpy(expected_pet)
    deviation = to_numpy(deviation_pet)
    prediction = to_numpy(prediction)
    gate = to_numpy(gate)

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4),
    )

    axes[0].imshow(target, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("GT PET")

    axes[1].imshow(expected, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Expected")

    axes[2].imshow(deviation, cmap="coolwarm")
    axes[2].set_title("Deviation")

    axes[3].imshow(gate, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Gate")

    axes[4].imshow(prediction, cmap="hot", vmin=0, vmax=1)
    axes[4].set_title("Final PET")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)