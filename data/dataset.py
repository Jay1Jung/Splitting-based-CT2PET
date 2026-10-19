from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def load_patient_ids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


class HecktorCTPETDataset(Dataset):
    def __init__(
        self,
        data_root,
        split_file,
        context_depth=96,
        num_orthogonal_slices=1,
        ct_min=-1000.0,
        ct_max=1000.0,
        pet_max=25.0,
        include_empty_slices=True,
    ):
        self.data_root = Path(data_root)
        self.patient_ids = load_patient_ids(split_file)

        self.context_depth = context_depth
        self.num_orthogonal_slices = num_orthogonal_slices

        self.ct_min = ct_min
        self.ct_max = ct_max
        self.pet_max = pet_max

        self.include_empty_slices = include_empty_slices

        if ct_max <= ct_min:
            raise ValueError("ct_max must be larger than ct_min")

        if pet_max <= 0:
            raise ValueError("pet_max must be positive")

        self.samples = []
        self._build_index()

    def _get_paths(self, patient_id):
        patient_dir = self.data_root / patient_id

        return {
            "ct": patient_dir / f"{patient_id}__CT.nii.gz",
            "pet": patient_dir / f"{patient_id}__PT.nii.gz",
            "label": patient_dir / f"{patient_id}.nii.gz",
        }

    def _build_index(self):
        missing = []

        for patient_id in self.patient_ids:
            paths = self._get_paths(patient_id)

            if not all(path.exists() for path in paths.values()):
                missing.append(patient_id)
                continue

            ct_img = nib.load(str(paths["ct"]))
            pet_img = nib.load(str(paths["pet"]))
            label_img = nib.load(str(paths["label"]))

            if not (ct_img.shape == pet_img.shape == label_img.shape):
                raise ValueError(
                    f"Shape mismatch for {patient_id}: "
                    f"CT {ct_img.shape}, PET {pet_img.shape}, label {label_img.shape}"
                )

            label = np.asarray(label_img.dataobj, dtype=np.float32)

            for z in range(label.shape[2]):
                has_lesion = np.any(label[:, :, z] > 0)

                if not self.include_empty_slices and not has_lesion:
                    continue

                self.samples.append(
                    {
                        "patient_id": patient_id,
                        "z": z,
                        "has_lesion": bool(has_lesion),
                    }
                )

        if missing:
            print(f"Skipped {len(missing)} patients with missing files")

        lesion_count = sum(sample["has_lesion"] for sample in self.samples)

        print(f"{len(self.patient_ids)} patients")
        print(f"{len(self.samples)} slices ({lesion_count} with lesion)")

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=2)
    def _load_patient(self, patient_id):
        paths = self._get_paths(patient_id)

        ct = nib.load(str(paths["ct"])).get_fdata(dtype=np.float32)
        pet = nib.load(str(paths["pet"])).get_fdata(dtype=np.float32)
        label = nib.load(str(paths["label"])).get_fdata(dtype=np.float32)

        return ct, pet, label

    def _normalize_ct(self, image):
        image = np.clip(image, self.ct_min, self.ct_max)
        image = (image - self.ct_min) / (self.ct_max - self.ct_min)
        return (image * 2.0 - 1.0).astype(np.float32)

    def _normalize_pet(self, image):
        image = np.clip(image, 0.0, self.pet_max)
        return (image / self.pet_max).astype(np.float32)

    def _get_z_window(self, volume, z):
        depth = self.context_depth

        start = z - depth // 2
        end = start + depth

        pad_before = max(0, -start)
        pad_after = max(0, end - volume.shape[2])

        start = max(0, start)
        end = min(volume.shape[2], end)

        window = volume[:, :, start:end]

        if pad_before or pad_after:
            window = np.pad(
                window,
                ((0, 0), (0, 0), (pad_before, pad_after)),
                mode="edge",
            )

        return window

    def _orthogonal_positions(self, size):
        positions = np.linspace(
            0,
            size - 1,
            self.num_orthogonal_slices + 2,
        )[1:-1]

        return np.round(positions).astype(int)

    def _get_orthogonal_views(self, ct_window):
        x_positions = self._orthogonal_positions(ct_window.shape[0])
        y_positions = self._orthogonal_positions(ct_window.shape[1])

        coronal = np.stack(
            [ct_window[:, y, :] for y in y_positions],
            axis=0,
        )

        sagittal = np.stack(
            [ct_window[x, :, :] for x in x_positions],
            axis=0,
        )

        return coronal, sagittal

    def __getitem__(self, index):
        sample = self.samples[index]

        patient_id = sample["patient_id"]
        z = sample["z"]

        ct, pet, label = self._load_patient(patient_id)

        axial_ct = ct[:, :, z]
        axial_pet = pet[:, :, z]
        tumor_mask = (label[:, :, z] > 0).astype(np.float32)

        ct_window = self._get_z_window(ct, z)
        coronal_ct, sagittal_ct = self._get_orthogonal_views(ct_window)

        axial_ct = self._normalize_ct(axial_ct)
        coronal_ct = self._normalize_ct(coronal_ct)
        sagittal_ct = self._normalize_ct(sagittal_ct)
        axial_pet = self._normalize_pet(axial_pet)

        return {
            "axial_ct": torch.from_numpy(axial_ct).float().unsqueeze(0),
            "coronal_ct": torch.from_numpy(coronal_ct).float(),
            "sagittal_ct": torch.from_numpy(sagittal_ct).float(),
            "pet": torch.from_numpy(axial_pet).float().unsqueeze(0),
            "tumor_mask": torch.from_numpy(tumor_mask).float().unsqueeze(0),
            "has_lesion": torch.tensor(sample["has_lesion"], dtype=torch.bool),
            "patient_id": patient_id,
            "z": torch.tensor(z, dtype=torch.long),
        }