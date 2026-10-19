import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def mae(prediction, target):
    return torch.abs(prediction - target).mean().item()


@torch.no_grad()
def rmse(prediction, target):
    return torch.sqrt(
        F.mse_loss(prediction, target)
    ).item()


@torch.no_grad()
def psnr(prediction, target, max_value=1.0):
    mse = F.mse_loss(prediction, target).item()

    if mse == 0:
        return float("inf")

    return 20 * math.log10(max_value) - 10 * math.log10(mse)


@torch.no_grad()
def lesion_mae(prediction, target, tumor_mask):
    mask = tumor_mask > 0.5

    if not mask.any():
        return None

    return torch.abs(
        prediction[mask] - target[mask]
    ).mean().item()


@torch.no_grad()
def nonlesion_mae(prediction, target, tumor_mask):
    mask = tumor_mask <= 0.5

    if not mask.any():
        return None

    return torch.abs(
        prediction[mask] - target[mask]
    ).mean().item()


@torch.no_grad()
def lesion_peak_error(prediction, target, tumor_mask):
    errors = []

    for i in range(target.shape[0]):
        mask = tumor_mask[i] > 0.5

        if not mask.any():
            continue

        pred_peak = prediction[i][mask].max()
        target_peak = target[i][mask].max()

        errors.append(
            torch.abs(pred_peak - target_peak)
        )

    if not errors:
        return None

    return torch.stack(errors).mean().item()


@torch.no_grad()
def lesion_peak_error_suv(
    prediction,
    target,
    tumor_mask,
    pet_max,
):
    error = lesion_peak_error(
        prediction,
        target,
        tumor_mask,
    )

    if error is None:
        return None

    return error * pet_max


@torch.no_grad()
def gradient_mae(prediction, target):
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]

    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    dx_error = torch.abs(
        pred_dx - target_dx
    ).mean()

    dy_error = torch.abs(
        pred_dy - target_dy
    ).mean()

    return (dx_error + dy_error).item() / 2