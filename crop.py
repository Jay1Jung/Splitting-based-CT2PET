import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def get_patient_files(patient_dir):
    patient_id = patient_dir.name

    ct_path = patient_dir / f"{patient_id}__CT.nii.gz"
    pet_path = patient_dir / f"{patient_id}__PT.nii.gz"
    label_path = patient_dir / f"{patient_id}.nii.gz"

    if not (ct_path.exists() and pet_path.exists() and label_path.exists()):
        return None

    return patient_id, ct_path, pet_path, label_path


def get_image_center(shape):
    return shape[0] // 2, shape[1] // 2


def get_body_center(ct, threshold=-800):
    # Use a subset of z slices so this does not create another huge 3D mask.
    step = max(1, ct.shape[2] // 64)
    sampled = ct[:, :, ::step]

    occupancy = (sampled > threshold).mean(axis=2)

    body = occupancy > 0.05

    if not body.any():
        return get_image_center(ct.shape)

    x, y = np.where(body)
    weights = occupancy[body]

    center_x = int(np.round(np.average(x, weights=weights)))
    center_y = int(np.round(np.average(y, weights=weights)))

    return center_x, center_y


def get_label_center(label):
    coords = np.argwhere(label > 0)

    if len(coords) == 0:
        return get_image_center(label.shape)

    center = coords[:, :2].mean(axis=0)

    return int(round(center[0])), int(round(center[1]))


def get_crop_center(ct, label, mode, body_threshold):
    if mode == "image":
        return get_image_center(ct.shape)

    if mode == "body":
        return get_body_center(ct, body_threshold)

    if mode == "label":
        return get_label_center(label)

    raise ValueError(f"Unknown center mode: {mode}")


def crop_xy(volume, center_x, center_y, crop_size, pad_value):
    half = crop_size // 2

    x0 = center_x - half
    y0 = center_y - half

    x1 = x0 + crop_size
    y1 = y0 + crop_size

    out = np.full(
        (crop_size, crop_size, volume.shape[2]),
        pad_value,
        dtype=volume.dtype,
    )

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)

    src_x1 = min(volume.shape[0], x1)
    src_y1 = min(volume.shape[1], y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0

    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    out[
        dst_x0:dst_x1,
        dst_y0:dst_y1,
        :
    ] = volume[
        src_x0:src_x1,
        src_y0:src_y1,
        :
    ]

    return out, x0, y0


def shifted_affine(affine, x0, y0):
    offset = np.array(
        [x0, y0, 0, 1],
        dtype=np.float64,
    )

    new_affine = affine.copy()
    new_affine[:3, 3] = (affine @ offset)[:3]

    return new_affine


def save_nifti(data, reference_img, path, x0, y0):
    new_affine = shifted_affine(
        reference_img.affine,
        x0,
        y0,
    )

    header = reference_img.header.copy()

    output = nib.Nifti1Image(
        data,
        new_affine,
        header=header,
    )

    nib.save(output, str(path))


def crop_patient(
    patient_dir,
    output_root,
    crop_size,
    center_mode,
    body_threshold,
):
    files = get_patient_files(patient_dir)

    if files is None:
        return None

    patient_id, ct_path, pet_path, label_path = files

    ct_img = nib.load(str(ct_path))
    pet_img = nib.load(str(pet_path))
    label_img = nib.load(str(label_path))

    if not (
        ct_img.shape
        == pet_img.shape
        == label_img.shape
    ):
        raise ValueError(
            f"{patient_id}: shape mismatch "
            f"CT={ct_img.shape}, "
            f"PET={pet_img.shape}, "
            f"label={label_img.shape}"
        )

    ct = ct_img.get_fdata(dtype=np.float32)
    pet = pet_img.get_fdata(dtype=np.float32)
    label = label_img.get_fdata(dtype=np.float32)

    center_x, center_y = get_crop_center(
        ct,
        label,
        center_mode,
        body_threshold,
    )

    cropped_ct, x0, y0 = crop_xy(
        ct,
        center_x,
        center_y,
        crop_size,
        pad_value=-1000.0,
    )

    cropped_pet, _, _ = crop_xy(
        pet,
        center_x,
        center_y,
        crop_size,
        pad_value=0.0,
    )

    cropped_label, _, _ = crop_xy(
        label,
        center_x,
        center_y,
        crop_size,
        pad_value=0.0,
    )

    output_dir = output_root / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    save_nifti(
        cropped_ct,
        ct_img,
        output_dir / f"{patient_id}__CT.nii.gz",
        x0,
        y0,
    )

    save_nifti(
        cropped_pet,
        pet_img,
        output_dir / f"{patient_id}__PT.nii.gz",
        x0,
        y0,
    )

    save_nifti(
        cropped_label,
        label_img,
        output_dir / f"{patient_id}.nii.gz",
        x0,
        y0,
    )

    lesion_before = int(np.count_nonzero(label > 0))
    lesion_after = int(np.count_nonzero(cropped_label > 0))

    retained = (
        lesion_after / lesion_before
        if lesion_before > 0
        else 1.0
    )

    return {
        "patient_id": patient_id,
        "original_shape": str(ct.shape),
        "cropped_shape": str(cropped_ct.shape),
        "center_x": center_x,
        "center_y": center_y,
        "x0": x0,
        "y0": y0,
        "lesion_voxels_before": lesion_before,
        "lesion_voxels_after": lesion_after,
        "lesion_retained": retained,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        required=True,
    )

    parser.add_argument(
        "--output_root",
        default="/workspace/data/hecktor2026_crop96",
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--center_mode",
        choices=["body", "image", "label"],
        default="body",
    )

    parser.add_argument(
        "--body_threshold",
        type=float,
        default=-800.0,
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_dirs = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir()
    )

    results = []

    for i, patient_dir in enumerate(patient_dirs, start=1):
        result = crop_patient(
            patient_dir,
            output_root,
            args.crop_size,
            args.center_mode,
            args.body_threshold,
        )

        if result is None:
            continue

        results.append(result)

        print(
            f"[{i}/{len(patient_dirs)}] "
            f"{result['patient_id']} "
            f"{result['original_shape']} -> "
            f"{result['cropped_shape']} "
            f"lesion retained: "
            f"{result['lesion_retained']:.3f}"
        )

    if not results:
        raise RuntimeError("No patients were processed")

    csv_path = output_root / "crop_info.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nProcessed {len(results)} patients")
    print(f"Crop info saved to {csv_path}")


if __name__ == "__main__":
    main()