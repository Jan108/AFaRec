checkpoint = "PekingU/rtdetr_v2_r18vd"
image_size = 480

from datasets import load_dataset

oai_data = load_dataset('imagefolder', data_dir="/mnt/data/afarec/data/OpenAnimalImages")
