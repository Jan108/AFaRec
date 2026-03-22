from pathlib import Path

import numpy as np
import pandas as pd


def afarec_splits(data_root: Path) -> None:
    """
    Creates the splits used for AFaRec: bird, cat, dog, small_animals

    :param data_root: Directory of the PetFace dataset
    :return: None
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Directory {data_root} does not exist. Please provide the PetFace dataset.")
    mapping = {
        'all': ['cat', 'chinchilla', 'degus', 'dog', 'ferret', 'guineapig', 'hamster',
                'hedgehog', 'javasparrow', 'parakeet', 'rabbit'],
        'bird': ['javasparrow', 'parakeet'],
        'small_animals': ['chinchilla', 'degus', 'ferret', 'guineapig', 'hamster', 'hedgehog', 'rabbit'],
    }
    for cls, species in mapping.items():
        cls_path = data_root / 'split' / cls
        cls_path.mkdir(parents=True, exist_ok=True)

        # Train
        amount_individuals = 0
        data = []

        for animal in species:
            animal_csv = data_root / 'split' / animal / 'train.csv'
            if not animal_csv.exists():
                raise FileNotFoundError(f'The train file {animal_csv} does not exist.')
            df = pd.read_csv(animal_csv, sep=',', index_col=0)
            df['label'] = df['label'] + amount_individuals
            amount_individuals += len(df['label'].unique())
            data.append(df)

        df_species = pd.concat(data)
        df_species.to_csv(cls_path / 'train.csv')

        # Test
        data = []

        for animal in species:
            animal_csv = data_root / 'split' / animal / 'test.txt'
            if not animal_csv.exists():
                raise FileNotFoundError(f'The test file {animal_csv} does not exist.')
            data.append(pd.read_csv(animal_csv, sep=',', header=None))

        df_species = pd.concat(data)
        df_species.to_csv(cls_path / 'test.txt', header=None, index=False)

        # Verification
        data = []
        for animal in species:
            animal_csv = data_root / 'split' / animal / 'verification.csv'
            if not animal_csv.exists():
                raise FileNotFoundError(f'The verification file {animal_csv} does not exist.')
            df = pd.read_csv(animal_csv, sep=',', index_col=0)
            data.append(df)

        df_species = pd.concat(data)
        df_species.to_csv(cls_path / 'verification.csv')

    for cls in ['all', 'bird', 'cat', 'dog', 'small_animals']:
        cls_path = data_root / 'split' / cls

        # Identification
        df_test = pd.read_csv(cls_path / 'test.txt', header=None)
        df_test = df_test.rename(columns={0: 'filename'})

        df_test['extracted_id'] = df_test['filename'].str.extract(r'([a-z]+[/\\]\d+)')[0]
        unique_strings = df_test['extracted_id'].unique()
        string_to_index = {s: i for i, s in enumerate(unique_strings)}
        df_test['individual'] = df_test['extracted_id'].map(string_to_index)

        df_test['pool'] = (~df_test['filename'].str.endswith('00.png')).astype(int)
        ids = df_test['individual'].unique()
        non_pool_ids = np.random.choice(ids, size=len(ids)//2, replace=False)

        df_img = df_test[~((df_test['individual'].isin(non_pool_ids)) & (df_test['pool'] == 1))]
        df_img[['individual', 'filename', 'pool']].to_csv(cls_path / 'identification_img.csv', index=False)

        df_label = df_test[['individual']]
        df_label.loc[:, 'true_label'] = df_label['individual']
        df_label.loc[df_label['true_label'].isin(non_pool_ids), 'true_label'] = -1
        df_label.to_csv(cls_path / 'identification_label.csv', index=False)


if __name__ == '__main__':
    afarec_splits(Path('/mnt/data/afarec/data/PetFace'))
