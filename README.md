# Solar Panel Electroluminescence (EL) Image Analysis

## Project Overview

This project implements an ensemble machine learning system for analyzing electroluminescence (EL) images of solar panels to predict power loss and assess panel health. The system combines a physics-based model with a deep learning ResNet model to provide accurate predictions of solar panel performance.

## Citations 

This project was implemented and trained on the ELPV dataset which is publically available at https://data.fz-juelich.de/dataset.xhtml?persistentId=doi:10.26165/JUELICH-DATA/TVWUUP

## Project Structure

```
h-proj/
├── scripts/
│   ├── run_inference.sh      # Main inference script
│   ├── inference.py          # Python inference implementation
│   └── physics_ensemble_final.ipynb  # Training notebook
├── models/
│   └── best_ensemble_model.pth       # Trained ensemble model 
├── data/
│   ├── data.csv              # Dataset metadata (720 samples)
│   ├── processed.zip         # Compressed processed images (322MB)
│   └── processed/            # Processed EL images (719 images)
└── results/
    └── image_9_results.txt   # Sample inference results
```

## Key Features

### Ensemble Model Architecture
- **Physics Model**: Analyzes inactive pixel percentage using threshold-based calculations
- **ResNet Model**: Modified ResNet18 architecture adapted for grayscale EL images
- **Meta Learner**: Linear combination layer that optimally weights both model predictions

### Model Components

#### Physics Model
- Calculates inactive pixel percentage using configurable threshold
- Applies linear regression: `pred = 1.5752 * inactive_pct - 0.6471`
- Provides interpretable physics-based predictions

#### ResNet Model
- Modified ResNet18 architecture for single-channel (grayscale) input
- First layer adapted to handle grayscale images
- Outputs power prediction values

#### Ensemble Model
- Combines physics and ResNet predictions using learned weights
- Provides final ensemble prediction for power loss assessment

## Dataset

- **Size**: 720 samples with corresponding EL images
- **Images**: 719 processed PNG files (grayscale EL images)
- **Metadata**: CSV file containing:
  - Peak power measurements
  - Nominal power values
  - Pressure readings
  - Excitation class
  - Module type and instance
  - Power group classifications
  - Cross-validation fold assignments

## Usage

### Prerequisites

- Python 3.x with PyTorch
- Conda environment named "solar"
- CUDA-compatible GPU (optional but recommended)

### Running Inference

1. **Configure the script**:
   Edit `scripts/run_inference.sh` to set your parameters:
   ```bash
   # Model Settings
   MODEL_PATH="/home/debonaire/h-proj/models/best_ensemble_model.pth"
   USE_GPU=true
   
   # Image Processing
   IMAGE_SIZE=224
   THRESHOLD=0.2
   
   # Input/Output
   INPUT_FILE="/home/debonaire/h-proj/data/processed/image_9.png"
   OUTPUT_DIR="results"
   ```

2. **Run inference**:
   ```bash
   cd /home/debonaire/h-proj
   chmod +x scripts/run_inference.sh
   ./scripts/run_inference.sh
   ```

### Direct Python Usage

```bash
python scripts/inference.py \
    --image /path/to/el/image.png \
    --model /home/debonaire/h-proj/models/best_ensemble_model.pth \
    --output results/ \
    --gpu true \
    --size 224 \
    --threshold 0.2
```

## Output Format

The system generates comprehensive analysis results including:

- **Physics Model Prediction**: Physics-based power prediction
- **ResNet Model Prediction**: Deep learning model prediction
- **Ensemble Prediction**: Final combined prediction
- **Power Loss**: Calculated as `1 - ensemble_prediction`
- **Relative Power**: Direct ensemble prediction value
- **Inactive Pixel %**: Percentage of inactive pixels detected

### Sample Output
```
Physics Model Prediction: 0.9007
ResNet Model Prediction: 1.3446
Ensemble Prediction: 0.8933
Power Loss: 0.1067
Relative Power: 0.8933
Inactive Pixel %: 0.9007
```

## Configuration Parameters

|  Parameter   |       Description        | Default Value |
|--------------|--------------------------|---------------|
| `IMAGE_SIZE` | Input image resolution   |      224      |
| `THRESHOLD`  | Inactive pixel threshold |      0.2      |
| `USE_GPU`    | Enable GPU acceleration  |      true     |
| `CONDA_ENV`  | Conda environment name   |      solar    |

## Model Performance

The ensemble model combines the interpretability of physics-based analysis with the pattern recognition capabilities of deep learning:

- **Physics Model**: Provides interpretable predictions based on inactive pixel analysis
- **ResNet Model**: Captures complex visual patterns in EL images
- **Ensemble**: Optimally combines both approaches for improved accuracy

## File Descriptions

### Scripts
- `run_inference.sh`: Main bash script for running inference with configuration
- `inference.py`: Python implementation of the inference pipeline
- `physics_ensemble_final.ipynb`: Jupyter notebook containing training and model development

### Models
- `best_ensemble_model.pth`: Trained ensemble model weights (43MB)

### Data
- `data.csv`: Dataset metadata with power measurements and cross-validation folds
- `processed/`: Directory containing 719 processed EL images
- `processed.zip`: Compressed archive of processed images

### Results
- `<image_number>_results.txt`: Sample inference results showing model predictions

## Technical Details

### Model Architecture
- **Input**: Grayscale EL images (224x224 pixels)
- **Physics Model**: Threshold-based inactive pixel analysis
- **ResNet Model**: Modified ResNet18 with grayscale adaptation
- **Output**: Power prediction values (0-1 range)

### Dependencies
- PyTorch
- torchvision
- PIL (Pillow)
- numpy
- argparse

## License

This project is for research and development purposes. Please ensure compliance with relevant data usage agreements when working with solar panel EL images.

## Contact

For questions or issues related to this solar panel EL image analysis system, please refer to the project documentation or contact the development team.

## Credits

**Codebase Creator**: Harini
**Project**: Solar Panel Electroluminescence (EL) Image Analysis  
**Date**: July 2025

<!-- This ensemble machine learning system for solar panel analysis was developed to combine physics-based modeling with deep learning approaches for accurate power loss prediction in solar panel electroluminescence images.  -->
