from pathlib import Path

RAW_ROOT = Path(r"D:\Alzhimers\ADNI DATA")

MODALITIES = {
    'MRI': RAW_ROOT / 'MRI' / 'ADNI',
    'PET': RAW_ROOT / 'PET' / 'ADNI',
    'DTI': RAW_ROOT / 'DTI' / 'ADNI'
}


def get_subject_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {p.name for p in sorted(path.iterdir()) if p.is_dir()}


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.rglob(pattern) if _.is_file())


def main() -> None:
    print('RAW DATA ROOT:', RAW_ROOT)
    print()

    subjects = {mod: get_subject_set(path) for mod, path in MODALITIES.items()}
    counts = {mod: len(s) for mod, s in subjects.items()}
    print('Subject counts by modality:')
    for mod, count in counts.items():
        print(f'  {mod}: {count}')

    overlap_mri_pet = subjects['MRI'] & subjects['PET']
    overlap_mri_dti = subjects['MRI'] & subjects['DTI']
    overlap_pet_dti = subjects['PET'] & subjects['DTI']
    overlap_all = subjects['MRI'] & subjects['PET'] & subjects['DTI']

    print('\nOverlap:')
    print(f'  MRI+PET: {len(overlap_mri_pet)}')
    print(f'  MRI+DTI: {len(overlap_mri_dti)}')
    print(f'  PET+DTI: {len(overlap_pet_dti)}')
    print(f'  MRI+PET+DTI: {len(overlap_all)}')

    print('\nExample subjects:')
    print('  MRI sample:', sorted(list(subjects['MRI']))[:10])
    print('  PET sample:', sorted(list(subjects['PET']))[:10])
    print('  DTI sample:', sorted(list(subjects['DTI']))[:10])

    print('\nRaw file format summary:')
    if MODALITIES['MRI'].exists():
        print('  MRI DICOM files:', count_files(MODALITIES['MRI'], '*.dcm'))
        print('  MRI NIfTI files:', count_files(MODALITIES['MRI'], '*.nii*'))
    if MODALITIES['PET'].exists():
        print('  PET DICOM files:', count_files(MODALITIES['PET'], '*.dcm'))
        print('  PET NIfTI files:', count_files(MODALITIES['PET'], '*.nii*'))
    if MODALITIES['DTI'].exists():
        print('  DTI DICOM files:', count_files(MODALITIES['DTI'], '*.dcm'))
        print('  DTI NIfTI files:', count_files(MODALITIES['DTI'], '*.nii*'))

    print('\nNote: Available raw data is DICOM-only for MRI, PET, and DTI in the current environment.')


if __name__ == '__main__':
    main()
