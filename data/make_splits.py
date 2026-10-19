import argparse
import random
from pathlib import Path


def find_patients(data_root):
    patients = []

    for patient_dir in sorted(Path(data_root).iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        ct = patient_dir / f"{patient_id}__CT.nii.gz"
        pet = patient_dir / f"{patient_id}__PT.nii.gz"
        label = patient_dir / f"{patient_id}.nii.gz"

        if ct.exists() and pet.exists() and label.exists():
            patients.append(patient_id)

    return patients


def save_ids(ids, path):
    with open(path, "w") as f:
        for patient_id in ids:
            f.write(patient_id + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must add up to 1")

    patients = find_patients(args.data_root)

    random.seed(args.seed)
    random.shuffle(patients)

    n = len(patients)

    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    train_ids = patients[:train_end]
    val_ids = patients[train_end:val_end]
    test_ids = patients[val_end:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_ids(train_ids, output_dir / "train.txt")
    save_ids(val_ids, output_dir / "val.txt")
    save_ids(test_ids, output_dir / "test.txt")

    print(f"train: {len(train_ids)}")
    print(f"val:   {len(val_ids)}")
    print(f"test:  {len(test_ids)}")


if __name__ == "__main__":
    main()