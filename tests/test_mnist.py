import pytest
import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import mnist_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist_model import MNIST, model_train_test

def test_model_parameters():
    logger.info("Starting model parameters test")
    model = MNIST()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params}")
    
    try:
        assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"
        logger.info("✓ Model parameters test passed")
    except AssertionError as e:
        logger.error(f"✗ Model parameters test failed: {str(e)}")
        raise

def test_has_dropout_layer():
    logger.info("Starting model dropout layer test")
    model = MNIST()
    has_dropout_layer = False
    
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            has_dropout_layer = True
            break

    try:
        assert has_dropout_layer , "Model have drop out layer"
        logger.info("✓ Model Dropout Layer test passed")
    except AssertionError as e:
        logger.error(f"✗ Model Dropout Layer test failed: {str(e)}")
        raise

def test_has_batch_norm():
    logger.info("Starting model batch norm layer test")
    model = MNIST()
    has_batch_norm_layer = False

    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            has_batch_norm_layer = True
            break

    try:
        assert has_batch_norm_layer, "Model have batch normalization"
        logger.info("✓ Model Batch Norm test passed")
    except AssertionError as e:
        logger.error(f"✗ Model Batch Norm test failed: {str(e)}")
        raise

def test_has_gap():
    logger.info("Starting model GAP layer test")
    model = MNIST()
    has_gap = False

    for layer in model.modules():
        if isinstance(layer, nn.AdaptiveAvgPool2d):
            has_gap = True
            break

    try:
        assert has_gap, "Model have GAP layer"
        logger.info("✓ Model GAP Layer test passed")
    except AssertionError as e:
        logger.error(f"✗ Model GAP Layer test failed: {str(e)}")
        raise

def test_has_linear_layer():
    logger.info("Starting model linear layer test")
    model = MNIST()
    has_linear_layer = False

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            has_linear_layer = True
            break

    try:
        assert has_linear_layer, "Model have linear layer"
        logger.info("✓ Model Linear Layer test passed")
    except AssertionError as e:
        logger.error(f"✗ Model Linear Layer test failed: {str(e)}")
        raise

def test_model_accuracy():
    logger.info("Starting model accuracy test")
    logger.info("Training model for 20 epoch...")
    
    try:
        metrics = model_train_test()
        logger.info(f"Training completed.")
        logger.info(f"Best Model Train accuracy: {metrics['train_accuracy']}%")
        logger.info(f"Best Model Test accuracy: {metrics['test_accuracy']}%")
        
        assert metrics['test_accuracy'] >= 99.4, \
            f"Model accuracy {metrics['test_accuracy']}% is below 99.4%"
        logger.info("✓ Model accuracy test passed")
    except AssertionError as e:
        logger.error(f"✗ Model accuracy test failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error during training: {str(e)}")
        raise 

