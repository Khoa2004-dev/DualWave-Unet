import os
import pathlib
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
from sklearn.model_selection import KFold
import monai.transforms as transforms
from monai.transforms import NormalizeIntensity
from .preprocessing import (
    pad_or_crop_image, 
    pad_or_crop_image_label,
    zscore_normalise,
    irm_min_max_preprocess
)

# Global paths
BRATS_TRAIN_FOLDERS = "path/to/Training"
VALI_PATH = 'path/to/Validation'

def get_brats_folder(on="train"):
    if on == "train":
        return BRATS_TRAIN_FOLDERS
    else:
        return VALI_PATH

class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="zscore", normal=False):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.normal = normal
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        self.target_spacing = [1.0, 1.0, 1.0]
        self.roi_size = [128, 128, 128]
        
        if self.training:
            self.transform = transforms.Compose([
                transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    k_divisible=self.roi_size,
                )
            ])
        
        if not no_seg:
            self.patterns += ["_seg"]
            
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [patient_dir / f"{patient_id}{value}.nii" for value in self.patterns]
            
            patient = dict(
                id=patient_id,
                t1=paths[0], t1ce=paths[1], t2=paths[2], flair=paths[3],
                seg=paths[4] if not no_seg else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
            et = patient_label == 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)
            et_present = 0

        if self.normal == False: 
            if self.normalisation == "minmax":
                patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
            elif self.normalisation == "zscore":
                patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        else:
            normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)
            patient_image = {key: normalize_intensity(patient_image[key]) for key in patient_image}

        patient_image = np.stack([patient_image[key] for key in patient_image])

        if self.training:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]              
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))

            if self.normal == True:
                data_dict = {
                    "image": patient_image,
                    "label": patient_label,
                }
                transformed = self.transform(data_dict)
                patient_image = transformed["image"]
                patient_label = transformed["label"]
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]  
            patient_image, patient_label = pad_or_crop_image_label(patient_image, patient_label, target_size=(128, 128, 128))

        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]

        return dict(
            patient_id=_patient["id"],
            image=patient_image, 
            label=patient_label,
            seg_path=str(_patient["seg"]) if self.training else str(_patient["t1"]),
            crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            et_present=et_present,
            supervised=True,
            idx=idx,
        )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)

def get_datasets(seed, on="train", fold_number=0, normalisation="zscore", use_fold=False):
    base_folder_train = pathlib.Path(get_brats_folder(on)).resolve()
    base_folder_vali = pathlib.Path(get_brats_folder("a")).resolve()
    assert base_folder_train.exists()
    patients_dir_train = sorted([x for x in base_folder_train.iterdir() if x.is_dir()])
    patients_dir_vali = sorted([x for x in base_folder_vali.iterdir() if x.is_dir()])

    if use_fold == True:
        kfold = KFold(5, shuffle=True, random_state=seed)
        splits = list(kfold.split(patients_dir_train))
        train_idx, test_idx = splits[fold_number]
        train = [patients_dir_train[i] for i in train_idx]
        test = [patients_dir_train[i] for i in test_idx]
        train_dataset = Brats(train, training=True, normalisation=normalisation, normal=True)
        test_dataset = Brats(test, training=False, benchmarking=True, normalisation=normalisation, normal=True)
    else:
        train_dataset = Brats(patients_dir_train, training=True, normalisation=normalisation, normal=True)
        test_dataset = Brats(patients_dir_vali, training=False, benchmarking=True, normalisation=normalisation, normal=True)

    return train_dataset, test_dataset