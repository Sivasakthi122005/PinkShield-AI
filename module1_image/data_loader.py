import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(), _NORM])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), _NORM])
    return train_tf, val_tf


def patient_id(path):
    """
    BreakHis names look like SOB_B_A-14-22549AB-40-001.png ->  patient '14-22549AB'.
    Returns None if the name does not follow that pattern.
    """
    m = re.search(r'SOB_[BM]_[A-Z]+-(\d+-[0-9A-Za-z]+)-\d+-\d+', os.path.basename(path))
    return m.group(1) if m else None


def split_indices(samples, val_frac=0.2, seed=42):
    """
    Patient-level split when BreakHis filenames are recognised (no patient
    appears in both train and validation - otherwise near-identical slides leak
    and accuracy is inflated). Falls back to a random split otherwise.
    """
    rng = np.random.default_rng(seed)
    pids = [patient_id(p) for p, _ in samples]
    if all(p is not None for p in pids):
        uniq = sorted(set(pids))
        rng.shuffle(uniq)
        val_p = set(uniq[:max(1, int(len(uniq) * val_frac))])
        val_idx = [i for i, p in enumerate(pids) if p in val_p]
        train_idx = [i for i, p in enumerate(pids) if p not in val_p]
        return train_idx, val_idx, 'patient-level'
    idx = rng.permutation(len(samples)).tolist()
    n_val = int(len(idx) * val_frac)
    return idx[n_val:], idx[:n_val], 'random (filenames not BreakHis-style)'


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    train_tf, val_tf = get_transforms()
    # TWO dataset objects so augmentation applies to training only.
    # (The old code mutated the shared dataset, which switched augmentation off
    #  for training as well.)
    train_full = datasets.ImageFolder(data_dir, transform=train_tf)
    val_full = datasets.ImageFolder(data_dir, transform=val_tf)
    class_names = train_full.classes

    tr_idx, va_idx, how = split_indices(train_full.samples)
    print(f"Classes: {class_names} | images: {len(train_full)} | split: {how}")
    print(f"Train: {len(tr_idx)} | Val: {len(va_idx)}")

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(Subset(train_full, tr_idx), shuffle=True, **kw)
    val_loader = DataLoader(Subset(val_full, va_idx), shuffle=False, **kw)
    train_loader.split_description = val_loader.split_description = how
    return train_loader, val_loader, class_names
