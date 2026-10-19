import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ExpectedEncoder(nn.Module):
    def __init__(self, base_channels=32, latent_channels=128):
        super().__init__()

        self.block1 = ConvBlock(1, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.block3 = ConvBlock(base_channels * 2, latent_channels, stride=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x


class ViewEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        output_dim=64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=2),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Linear(
            base_channels * 4,
            output_dim,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)

        return self.projection(x)


class OrthogonalEncoder(nn.Module):
    def __init__(
        self,
        num_views=1,
        base_channels=32,
        context_dim=128,
    ):
        super().__init__()

        view_dim = context_dim // 2

        self.coronal_encoder = ViewEncoder(
            in_channels=num_views,
            base_channels=base_channels,
            output_dim=view_dim,
        )

        self.sagittal_encoder = ViewEncoder(
            in_channels=num_views,
            base_channels=base_channels,
            output_dim=view_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, coronal, sagittal):
        coronal_context = self.coronal_encoder(coronal)
        sagittal_context = self.sagittal_encoder(sagittal)

        context = torch.cat(
            [coronal_context, sagittal_context],
            dim=1,
        )

        return self.fusion(context)


class FiLM(nn.Module):
    def __init__(self, context_dim, channels):
        super().__init__()

        self.projection = nn.Linear(
            context_dim,
            channels * 2,
        )

    def forward(self, feature, context):
        scale, shift = self.projection(context).chunk(2, dim=1)

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        return feature * (1.0 + scale) + shift


class DeviationEncoder(nn.Module):
    def __init__(
        self,
        base_channels=32,
        latent_channels=128,
        context_dim=128,
    ):
        super().__init__()

        self.block1 = ConvBlock(1, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.block3 = ConvBlock(base_channels * 2, latent_channels, stride=2)

        self.condition = FiLM(
            context_dim,
            latent_channels,
        )

        self.refine = ConvBlock(
            latent_channels,
            latent_channels,
        )

    def forward(self, axial_ct, context):
        x = self.block1(axial_ct)
        x = self.block2(x)
        x = self.block3(x)

        x = self.condition(x, context)
        x = self.refine(x)

        return x


class SoftGate(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()

        self.network = nn.Sequential(
            ConvBlock(1, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels * 2, stride=2),
            ConvBlock(hidden_channels * 2, hidden_channels * 2, stride=2),
            nn.Conv2d(
                hidden_channels * 2,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, axial_ct):
        return torch.sigmoid(
            self.network(axial_ct)
        )


class LatentGlue(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels * 3 + 1,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, channels),
            nn.SiLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        expected_feature,
        deviation_feature,
        gate,
    ):
        if gate.shape[-2:] != expected_feature.shape[-2:]:
            gate = F.interpolate(
                gate,
                size=expected_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        weighted_feature = (
            (1.0 - gate) * expected_feature
            + gate * deviation_feature
        )

        combined = torch.cat(
            [
                expected_feature,
                deviation_feature,
                weighted_feature,
                gate,
            ],
            dim=1,
        )

        correction = self.refine(combined)

        return weighted_feature + correction


class PETDecoder(nn.Module):
    def __init__(
        self,
        latent_channels=128,
        base_channels=32,
        output_activation=True,
    ):
        super().__init__()

        self.block1 = ConvBlock(
            latent_channels,
            latent_channels,
        )

        self.block2 = ConvBlock(
            latent_channels,
            base_channels * 2,
        )

        self.block3 = ConvBlock(
            base_channels * 2,
            base_channels,
        )

        self.output = nn.Conv2d(
            base_channels,
            1,
            kernel_size=1,
        )

        self.output_activation = output_activation

    def forward(self, x):
        x = self.block1(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.block2(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.block3(x)
        x = self.output(x)

        if self.output_activation:
            x = torch.sigmoid(x)

        return x