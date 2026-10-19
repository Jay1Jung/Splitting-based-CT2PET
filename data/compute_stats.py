import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from dataset import load_patient_ids


def sample_values(array, n, rng):
    values = array.reshape(-1)

    if len(values) <= n:
        return values

    idx = rng.choice(len(values), size=n, replace=False)
    return values[idx]


def print_percentiles(name, values):
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 100]

    print(f"\n{name}")

    for p in percentiles:
        value = np.percentile(values, p)
        print(f"p{p:<5}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--samples_per_patient", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    patient_ids = load_patient_ids(args.split_file)
    data_root = Path(args.data_root)

    ct_samples = []
    pet_samples = []
    lesion_pet_samples = []
    normal_pet_samples = []

    for i, patient_id in enumerate(patient_ids):
        patient_dir = data_root / patient_id

        ct = nib.load(
            str(patient_dir / f"{patient_id}__CT.nii.gz")
        ).get_fdata(dtype=np.float32)

        pet = nib.load(
            str(patient_dir / f"{patient_id}__PT.nii.gz")
        ).get_fdata(dtype=np.float32)

        label = nib.load(
            str(patient_dir / f"{patient_id}.nii.gz")
        ).get_fdata(dtype=np.float32)

        ct_samples.append(
            sample_values(ct, args.samples_per_patient, rng)
        )

        pet_samples.append(
            sample_values(pet, args.samples_per_patient, rng)
        )

        lesion_pet = pet[label > 0]
        normal_pet = pet[label == 0]

        if len(lesion_pet):
            lesion_pet_samples.append(
                sample_values(lesion_pet, args.samples_per_patient, rng)
            )

        if len(normal_pet):
            normal_pet_samples.append(
                sample_values(normal_pet, args.samples_per_patient, rng)
            )

        print(f"{i + 1}/{len(patient_ids)} {patient_id}")

    print_percentiles("CT", np.concatenate(ct_samples))
    print_percentiles("PET", np.concatenate(pet_samples))

    if lesion_pet_samples:
        print_percentiles(
            "Lesion PET",
            np.concatenate(lesion_pet_samples),
        )

    if normal_pet_samples:
        print_percentiles(
            "Non-lesion PET",
            np.concatenate(normal_pet_samples),
        )


if __name__ == "__main__":
    main()