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

def has_dropout():
    logger.info("Starting model dropout layer test")
    model = MNIST()
    has_dropout_layer = False
    
    for layer in model.modules():
        has_dropout_layer = has_dropout_layer or isinstance(layer, nn.Dropout)

    try:
        assert has_dropout_layer == False , "Model does not have drop out layer"
        logger.info("✓ Model Dropout Layer test passed")
    except AssertionError as e:
        logger.error(f"✗ Model Dropout Layer test failed: {str(e)}")
        raise

def has_batch_norm():
    logger.info("Starting model batch norm layer test")
    model = MNIST()
    has_batch_norm_layer = False
    
    for layer in model.modules():
        has_batch_norm_layer = has_batch_norm_layer or isinstance(layer, nn.BatchNorm2d)

    try:
        assert has_batch_norm_layer == False , "Model does not have batch normalization"
        logger.info("✓ Model Batch Norm test passed")
    except AssertionError as e:
        logger.error(f"✗ Model Batch Norm test failed: {str(e)}")
        raise

def test_model_accuracy():
    logger.info("Starting model accuracy test")
    logger.info("Training model for one epoch...")
    
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

