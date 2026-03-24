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

    for cls in ['bird', 'cat', 'dog', 'small_animals']:
        cls_path = data_root / 'split' / cls

        # Identification

        # Load and prepare the data
        df_test = pd.read_csv(cls_path / 'test.txt', header=None)
        df_test = df_test.rename(columns={0: 'filename'})
        df_test['extracted_id'] = df_test['filename'].str.extract(r'([a-z]+[/\\]\d+)')[0]
        unique_strings = df_test['extracted_id'].unique()
        string_to_index = {s: i for i, s in enumerate(unique_strings)}
        df_test['individual'] = df_test['extracted_id'].map(string_to_index)

        # Select 400 unique individuals
        ids = df_test['individual'].unique()
        print(f'{cls} has {len(ids)} unique individuals')
        selected_ids = np.random.choice(ids, size=400, replace=False)

        # For each selected individual, pick one random image for the feature set (pool=0)
        df_feature = df_test[df_test['individual'].isin(selected_ids)].groupby('individual').apply(
            lambda x: x.sample(1)
        ).reset_index(drop=True)
        df_feature['pool'] = 0

        # For 200 of these individuals, pick one additional image for the pool set (pool=1),
        # ensuring it is different from the feature image
        pool_ids = np.random.choice(selected_ids, size=200, replace=False)
        df_pool = pd.DataFrame()
        for individual in pool_ids:
            individual_images = df_test[df_test['individual'] == individual]
            feature_image = df_feature[df_feature['individual'] == individual]['filename'].values[0]
            pool_image = individual_images[individual_images['filename'] != feature_image].sample(1)
            df_pool = pd.concat([df_pool, pool_image], ignore_index=True)
        df_pool['pool'] = 1

        # Combine the feature and pool sets
        df_img = pd.concat([df_feature, df_pool], ignore_index=True)

        # Save the image list
        df_img[['individual', 'filename', 'pool']].to_csv(cls_path / 'identification_img.csv', index=False)

        # Prepare the label DataFrame
        df_label = pd.DataFrame({'individual': selected_ids})
        df_label['true_label'] = df_label['individual']
        df_label.loc[~df_label['individual'].isin(pool_ids), 'true_label'] = -1
        df_label = df_label.drop_duplicates()
        df_label.to_csv(cls_path / 'identification_label.csv', index=False)


if __name__ == '__main__':
    afarec_splits(Path('/mnt/data/afarec/data/PetFace'))
