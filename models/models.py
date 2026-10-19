import torch.nn as nn

from .components import (
    ExpectedEncoder,
    OrthogonalEncoder,
    DeviationEncoder,
    SoftGate,
    LatentGlue,
    PETDecoder,
)


class CT2PETModel(nn.Module):
    def __init__(
        self,
        num_orthogonal_slices=1,
        base_channels=32,
        latent_channels=128,
        context_dim=128,
        gate_channels=32,
    ):
        super().__init__()

        self.expected_encoder = ExpectedEncoder(
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

        self.orthogonal_encoder = OrthogonalEncoder(
            num_views=num_orthogonal_slices,
            base_channels=base_channels,
            context_dim=context_dim,
        )

        self.deviation_encoder = DeviationEncoder(
            base_channels=base_channels,
            latent_channels=latent_channels,
            context_dim=context_dim,
        )

        self.gate = SoftGate(
            hidden_channels=gate_channels,
        )

        self.glue = LatentGlue(
            channels=latent_channels,
        )

        self.decoder = PETDecoder(
            latent_channels=latent_channels,
            base_channels=base_channels,
            output_activation=True,
        )

        self.expected_decoder = PETDecoder(
            latent_channels=latent_channels,
            base_channels=base_channels,
            output_activation=True,
        )

        self.deviation_decoder = PETDecoder(
            latent_channels=latent_channels,
            base_channels=base_channels,
            output_activation=False,
        )

    def forward(
        self,
        axial_ct,
        coronal_ct,
        sagittal_ct,
    ):
        expected_feature = self.expected_encoder(
            axial_ct
        )

        context = self.orthogonal_encoder(
            coronal_ct,
            sagittal_ct,
        )

        deviation_feature = self.deviation_encoder(
            axial_ct,
            context,
        )

        gate = self.gate(
            axial_ct
        )

        glued_feature = self.glue(
            expected_feature,
            deviation_feature,
            gate,
        )

        pet = self.decoder(
            glued_feature
        )

        expected_pet = self.expected_decoder(
            expected_feature
        )

        deviation_pet = self.deviation_decoder(
            deviation_feature
        )

        return {
            "pet": pet,
            "expected_pet": expected_pet,
            "deviation_pet": deviation_pet,
            "gate": gate,
            "expected_feature": expected_feature,
            "deviation_feature": deviation_feature,
            "glued_feature": glued_feature,
            "orthogonal_context": context,
        }