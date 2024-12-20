# MNIST Lightweight Neural Network

[![ML Pipeline Test](https://github.com/shubhjadhav/tsai-erav3-s6/actions/workflows/test_model.yml/badge.svg?branch=main)](https://github.com/shubhjadhav/tsai-erav3-s6/actions/workflows/test_model.yml)

A lightweight convolutional neural network for MNIST digit classification that achieves >=99.4% accuracy in 20 epoch while maintaining less than 20,000 parameters. The project includes automated testing through GitHub Actions to ensure model efficiency and performance.

## Model Architecture

The MNIST model uses an efficient architecture:
- **Input**: 28x28 grayscale images
- **Convolutional layers** with batch normalization:
  - **Conv1**: 1 → 16 channels (3x3 kernel, padding=0)  
  - **Conv2**: 16 → 32 channels (3x3 kernel, padding=0)  
  - **Conv3**: 32 → 10 channels (1x1 kernel, padding=0)  
  - **Conv4**: 10 → 16 channels (3x3 kernel, padding=0)  
  - **Conv5**: 16 → 16 channels (3x3 kernel, padding=0)  
  - **Conv6**: 16 → 16 channels (3x3 kernel, padding=0)  
  - **Conv7**: 16 → 16 channels (3x3 kernel, padding=1)  
- **Max pooling** after each convolution block (with varying pool sizes)
- **Global Average Pooling** layer to reduce spatial dimensions
- **Single fully connected layer**: 16 → 10 (output classes)
- **Total parameters**: < 20,000


## Performance Metrics
- Accuracy: >=99.4%
- Parameters: ~13,000
- Training time: ~10-13 minutes on CPU

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- torchsummary
- pytest (for testing)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model, run:
```bash
python mnist_model.py
```

This will:
- Download the MNIST dataset automatically
- Train for 20 epoch
- Display progress and metrics
- Show logs of loss and accuracy for each epoch

### Running Tests
To run the test suite:

```bash
pytest tests/ -v --log-cli-level=INFO
```

## GitHub Actions Pipeline

The project includes automated testing that verifies:
1. Model size (< 20,000 parameters)
2. Model has drop out layer
3. Model has Batch Normalization
4. Model has Linear Layer (Fully Connected)
5. Model has GAP
6. Model performance (>= 99.4% test accuracy)

The pipeline runs automatically on:
- Every push to the repository
- Every pull request

## Project Structure

```
.
├── mnist_model.py                  # Main model implementation
├── data                            # MNIST Dataset
├── tests/
│   └── test_mnist.py               # Automated tests
├── .github/
│   └── workflows/                  # GitHub Actions configuration
│       └── test_model.yml          # ML pipeline workflow
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Documentation
```

## Model Details

### Architecture Highlights
- **Uses batch normalization** for stable training and faster convergence.
- **Efficient parameter usage** through strategic kernel sizes, reducing model complexity while maintaining performance.
- **Optimized max pooling placement** to progressively downsample feature maps while retaining critical spatial information.
- **Single fully connected layer** for classification, reducing the number of parameters and complexity.

### Layer-by-Layer Architecture

#### Input Layer
- **Input shape**: 1x28x28 (grayscale MNIST images)

#### Convolutional Block 1
- **Conv2d**: 1 → 16 channels, 3x3 kernel, padding=0 (output shape: 16x26x26)
- **BatchNorm2d**: Normalizes the activations from the convolution layer.
- **ReLU activation**: Non-linear activation function.
- **Dropout**: Applied for regularization to avoid overfitting.

#### Convolutional Block 2
- **Conv2d**: 16 → 32 channels, 3x3 kernel, padding=0 (output shape: 32x24x24)
- **BatchNorm2d**: Normalizes the activations from the convolution layer.
- **ReLU activation**: Non-linear activation function.
- **Dropout**: Applied for regularization.

#### Convolutional Block 3
- **Conv2d**: 32 → 10 channels, 1x1 kernel, padding=0 (output shape: 10x24x24)
- **BatchNorm2d**: Normalizes the activations from the convolution layer.
- **ReLU activation**: Non-linear activation function.

#### Convolutional Block 4
- **Conv2d**: 10 → 16 channels, 3x3 kernel, padding=0 (output shape: 16x22x22)
- **BatchNorm2d**: Normalizes the activations from the convolution layer.
- **ReLU activation**: Non-linear activation function.
- **Dropout**: Applied for regularization.

#### Convolutional Block 5
- **Conv2d**: 16 → 16 channels, 3x3 kernel, padding=0 (output shape: 16x20x20)
- **BatchNorm2d**: Normalizes the activations from the convolution layer.
- **ReLU activation**: Non-linear activation function.
- **Dropout**: Applied for regularization.

#### Convolutional Block 6
- **Conv2d**: 16 → 16 channels, 3x3 kernel, padding=0 (output shape: 16x18x18)
- **BatchNorm2d**: Normalizes the activations from the convolution layer.
- **ReLU activation**: Non-linear activation function.
- **Dropout**: Applied for regularization.

#### Convolutional Block 7
- **Conv2d**: 16 → 16 channels, 3x3 kernel, padding=1 (output shape: 16x18x18)
- **BatchNorm2d**: Normalizes the activations from the convolution layer.
- **ReLU activation**: Non-linear activation function.
- **Dropout**: Applied for regularization.

#### Global Average Pooling (GAP)
- **AdaptiveAvgPool2d**: Reduces the spatial dimensions to 1x1 (output shape: 16x1x1).

#### Classification Head
- **Flatten**: Flattens the output of GAP layer into a vector of 16 features (1,296 features).
- **Fully Connected (Linear)**: 1,296 → 10 classes (output shape: 10).
- **LogSoftmax**: Outputs class probabilities (log of softmax).

### Training Configuration
- **Batch size**: 128
- **Optimizer**: SGD with momentum (0.9)
- **Learning rate**: 0.05
- **Data normalization**:
  - **Mean** = 0.1307
  - **Std** = 0.3081

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST Dataset creators
- PyTorch team
- GitHub Actions for CI/CD support
## Testing Framework

The project includes comprehensive test cases to ensure model quality and performance:

### 1. Parameter Count Test (`test_model_parameters`)
- Verifies model architecture efficiency
- Ensures total parameters stay below 20,000

### 2. Dropout Layer Test (`test_has_dropout_layer`)
- Verifies the presence of a dropout layer in the model

### 3. Batch Normalization Test (`test_has_batch_norm`)
- Ensures batch normalization layers are present in the model

### 4. GAP Layer Test (`test_has_gap`)
- Verifies the presence of a Global Average Pooling (GAP) layer

### 5. Linear Layer Test (`test_has_linear_layer`)
- Verifies the presence of a fully connected linear layer

### 6. Model Accuracy Test (`test_model_accuracy`)
- Trains the model for 20 epochs
- Verifies accuracy exceeds 99.4%


### Running Tests

Execute all tests with detailed logging:
```bash
pytest tests/ -v --log-cli-level=INFO
```

Test artifacts and logs are automatically uploaded to GitHub Actions for each run.