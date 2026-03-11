from pathlib import Path
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


if __name__ == '__main__':
    afarec_splits(Path('/mnt/data/afarec/data/PetFace'))