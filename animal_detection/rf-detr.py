import torch
from rfdetr import RFDETRSmall



def train_model() -> None:
    model = RFDETRSmall()
    model.train(
        dataset_dir='/mnt/data/afarec/data/OpenAnimalImages_RF-DETR_woBird', dataset_file='yolo',
        epochs=100, batch_size=4, grad_accum_steps=4, checkpoint_interval=1, device="cuda",
        output_dir ='/mnt/data/afarec/code/runs/rfdetr/2026-02-04'
    )


def resume_training() -> None:
    model = RFDETRSmall()
    model.train(
        dataset_dir='/mnt/data/afarec/data/OpenAnimalImages', dataset_file='yolo',
        epochs=100, batch_size=4, grad_accum_steps=4, checkpoint_interval=1, device="cuda",
        output_dir='/mnt/data/afarec/code/runs/rfdetr/2026-02-04',
        resume='????????????????',
    )
    pass


if __name__ == '__main__':
    torch.cuda.memory.set_per_process_memory_fraction(0.9, device=0)
    train_model()
    # resume_training()