import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(kernel_size, sigma, device, dtype):
    x = torch.arange(kernel_size, device=device, dtype=dtype)
    x = x - (kernel_size - 1) / 2

    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    return kernel[:, None] * kernel[None, :]


def gaussian_blur(image, kernel_size=15, sigma=3.0):
    kernel = gaussian_kernel(
        kernel_size,
        sigma,
        image.device,
        image.dtype,
    )

    kernel = kernel[None, None]
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)

    return F.conv2d(
        image,
        kernel,
        padding=kernel_size // 2,
        groups=image.shape[1],
    )


def make_soft_gate_target(tumor_mask, dilation=7, sigma=2.0):
    if dilation > 1:
        tumor_mask = F.max_pool2d(
            tumor_mask,
            kernel_size=dilation,
            stride=1,
            padding=dilation // 2,
        )

    kernel_size = int(2 * round(3 * sigma) + 1)

    soft_mask = gaussian_blur(
        tumor_mask,
        kernel_size=kernel_size,
        sigma=sigma,
    )

    return soft_mask.clamp(0, 1)


def weighted_l1(prediction, target, weight):
    error = torch.abs(prediction - target) * weight
    return error.sum() / (weight.sum() + 1e-8)


def image_gradient(image):
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]

    return dx, dy


class CT2PETLoss(nn.Module):
    def __init__(
        self,
        expected_weight=0.5,
        deviation_weight=0.5,
        gate_weight=0.1,
        gradient_weight=0.1,
        lesion_weight=4.0,
        blur_kernel=15,
        blur_sigma=3.0,
        gate_dilation=7,
        gate_sigma=2.0,
    ):
        super().__init__()

        self.expected_weight = expected_weight
        self.deviation_weight = deviation_weight
        self.gate_weight = gate_weight
        self.gradient_weight = gradient_weight
        self.lesion_weight = lesion_weight

        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

        self.gate_dilation = gate_dilation
        self.gate_sigma = gate_sigma

    def forward(self, outputs, pet, tumor_mask):
        prediction = outputs["pet"]
        expected_prediction = outputs["expected_pet"]
        deviation_prediction = outputs["deviation_pet"]
        predicted_gate = outputs["gate"]

        expected_target = gaussian_blur(
            pet,
            kernel_size=self.blur_kernel,
            sigma=self.blur_sigma,
        )

        deviation_target = pet - expected_target

        gate_target = make_soft_gate_target(
            tumor_mask,
            dilation=self.gate_dilation,
            sigma=self.gate_sigma,
        )

        lesion_weights = 1.0 + self.lesion_weight * gate_target

        final_loss = weighted_l1(
            prediction,
            pet,
            lesion_weights,
        )

        normal_weights = 1.0 - gate_target + 0.05

        expected_loss = weighted_l1(
            expected_prediction,
            expected_target,
            normal_weights,
        )

        deviation_loss = weighted_l1(
            deviation_prediction,
            deviation_target,
            lesion_weights,
        )

        gate_target_small = F.interpolate(
            gate_target,
            size=predicted_gate.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        gate_loss = F.binary_cross_entropy(
            predicted_gate,
            gate_target_small,
        )

        pred_dx, pred_dy = image_gradient(prediction)
        target_dx, target_dy = image_gradient(pet)

        gradient_loss = (
            F.l1_loss(pred_dx, target_dx)
            + F.l1_loss(pred_dy, target_dy)
        )

        total_loss = (
            final_loss
            + self.expected_weight * expected_loss
            + self.deviation_weight * deviation_loss
            + self.gate_weight * gate_loss
            + self.gradient_weight * gradient_loss
        )

        return {
            "loss": total_loss,
            "final": final_loss.detach(),
            "expected": expected_loss.detach(),
            "deviation": deviation_loss.detach(),
            "gate": gate_loss.detach(),
            "gradient": gradient_loss.detach(),
        }