"""
OLy AGI Framework - base_module
Foundation Module

This module is part of the OLy AGI Framework, implementing Base Module functionality.
Created: 2024-10-30 19:24:35
"""

from typing import Dict, List, Optional, Union
import torch
import numpy as np


class BaseModule:
    """Base class for Base Module."""
    
    def __init__(self):
        self.config = None
        self.initialized = False
    
    def initialize(self, config: Dict):
        """Initialize the component with configuration."""
        self.config = config
        self.initialized = True
    
    def validate(self) -> bool:
        """Validate the component configuration."""
        return self.initialized

    def reset(self):
        """Reset the component to its initial state."""
        self.initialized = False
        self.config = None