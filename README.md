# AFAREC: Animal Face Recognition and Detection

A unified pipeline for **animal face detection and recognition** using adapted human models. This repository contains all code and resources needed to reproduce the results of my master thesis. It serves as **exploratory groundwork** to assess the feasibility and limitations of animal face recognition, rather than a turnkey system.

---

## Repository Structure

```
afarec/
├── code/                              # This Git repository
│   ├── animal_detection/              # Animal detection component
│   ├── face_detection/                # Face detection component
│   │   ├── RetinaFace/                # RetinaFace implementation
│   │   ├── SCRFD/                     # SCRFD implementation
│   │   └── YuNet/                     # YuNet implementation
│   ├── face_recognition/              # Face recognition component
│   │   ├── ArcFace/                   # ArcFace implementation
│   │   ├── GhostFaceNets/             # GhostFaceNets implementation
│   │   └── SphereFace/                # SphereFace implementation
│   ├── data/                          # Scripts for data manipulation and annotation
│   ├── requirements.txt               # Python dependencies
│   └── README.md                      # This file
└── data/                              # Datasets and annotations (not in Git)
```

### External Repositories


| Model         | Repository Link                                   |
| ------------- | ------------------------------------------------- |
| RetinaFace    | [GitHub](https://github.com/Jan108/RetinaFace)    |
| SCRFD         | [GitHub](https://github.com/Jan108/SCRFD)         |
| YuNet         | [GitHub](https://github.com/Jan108/YuNet)         |
| ArcFace       | [GitHub](https://github.com/Jan108/ArcFace)       |
| GhostFaceNets | [GitHub](https://github.com/Jan108/GhostFaceNets) |
| SphereFace    | [GitHub](https://github.com/Jan108/SphereFace)    |


---

## Installation

### Prerequisites

- **Python 3.12**

### Steps

1. Clone this repository:
  ```bash
   git clone https://github.com/Jan108/AFaRec.git code
   cd code
  ```
2. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
3. **Modular Setup**: Each component is self-contained. Install only the external repositories for the models you plan to use. This repository is required for data preparation.

---

## Data Preparation

### Datasets

- **OpenAnimalImages** and **OpenAnimalFaceImages**: Automatically downloaded and prepared using the scripts in `data/`.
- **PetFace Dataset**: Not openly available. Request access from the [original authors](https://github.com/mapooon/PetFace) and download manually.

### Scripts

- `main.py`: Contains `create_all_datasets(data_root: Path)` to generate both OpenAnimalImages and OpenAnimalFaceImages datasets.
- `oai.py`: Scripts for Open Animal Image Dataset.
- `oafi.py`: Scripts for Open Animal Face Images Dataset.
- `oafi_annotations.py`: Custom annotator (can be run as a web service).

### Dataset Viewer

- Use [FiftyOne](https://docs.voxel51.com/) to interact with the datasets. Launch the dataset viewer with:
  ```bash
  python -m data.main launch_app
  ```

---

## Usage

### Component-Specific Instructions

Each model in the `face_detection/` and `face_recognition/` modules includes its own `README.md` with setup and usage instructions.

### Evaluation

- Use `evaluation.py` to generate figures and tables from model results.

### Training

- Training for the **animal detection component** is configured in `training_runs.py`.

---

## Future Plans

- Public release of annotations and trained models (currently available upon request).

---

## License
This project is licensed under the **GNU AGPLv3**. See the [LICENSE](LICENSE) file for details.