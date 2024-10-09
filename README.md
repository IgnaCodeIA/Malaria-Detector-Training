# Malaria Detection Model Training

This project aims to detect malaria in blood cell images using deep learning techniques. The repository is focused on training the model, including data preprocessing, model training, and evaluation.

## Project Structure

```
├── README.md
├── data
│   ├── Test
│   │   ├── Parasitized
│   │   └── Uninfected
│   ├── Train
│   │   ├── Parasitized
│   │   └── Uninfected
│   └── val
│       ├── Parasitized
│       └── Uninfected
├── mlruns
├── models
├── notebooks
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── tests
```

- `data/`: Contains the dataset, divided into training, validation, and test sets.
- `mlruns/`: Stores experiment tracking information.
- `models/`: Directory to store trained models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and testing.
- `src/`: Holds the Python scripts for data preprocessing, model definition, and training.
- `tests/`: Unit tests for the project.
- `requirements.txt`: Python dependencies for the project.

## Prerequisites

- Docker
- Python 3.12
- [DVC](https://dvc.org/) (Data Version Control)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IgnaCodeIA/Malaria-Detector-Training
   cd malaria-detector-training
   ```

2. **Install dependencies:** Use the provided `requirements.txt` to install the Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup DVC:** Initialize DVC in the project.
   ```bash
   dvc init
   ```

4. **Pull the data and models from DVC remote storage if configured:**
   ```bash
   dvc pull
   ```

5. **Configure the environment:** Update the `config.yaml` file with the appropriate paths and parameters.

## Running the Project

### Training the Model
To train the model, use DVC:

```bash
dvc repro
```

The `dvc.yaml` file contains the stages for training, including dependencies and outputs. The training process will automatically generate logs and save the trained model in the `models/` directory.

### Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t malaria-detector-training .
   ```

2. **Run the container:**
   ```bash
   docker run -it --rm -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs -p 6006:6006 malaria-detector-training
   ```

   This command will mount the local `models` and `logs` directories to the container, and expose port 6006 for TensorBoard.

### Running TensorBoard
To view the training logs with TensorBoard, use:

```bash
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006
```

Access TensorBoard at [http://localhost:6006](http://localhost:6006).

## DVC Overview

DVC (Data Version Control) is used in this project to manage datasets and model versions. The `dvc.yaml` file defines the stages of the pipeline, and the `.dvc` files track data files. Key commands include:

- **Initialize DVC:**
  ```bash
  dvc init
  ```

- **Track data files:**
  ```bash
  dvc add data/Train
  dvc add data/val
  ```

- **Reproduce the pipeline:**
  ```bash
  dvc repro
  ```

- **Push data and models to remote storage:**
  ```bash
  dvc push
  ```

- **Pull data and models from remote storage:**
  ```bash
  dvc pull
  ```

## Configuration

The project configuration is managed via `config.yaml`, which includes parameters for data paths, training hyperparameters, and augmentation settings. Make sure to update this file according to your setup before running the pipeline.

## Troubleshooting

- **Common Issues:** Ensure that all paths in `config.yaml` are correct and that DVC is properly set up.
- **Docker Build Errors:** Verify that Docker is installed and running, and that all dependencies in `requirements.txt` are correctly listed.