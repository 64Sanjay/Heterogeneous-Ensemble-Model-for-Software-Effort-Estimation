#!/usr/bin/env python
"""
Clean runner script - suppresses all warnings
"""

import os
import sys
import warnings
import logging

# Suppress ALL warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

# Redirect stderr temporarily to suppress TF messages
import io
old_stderr = sys.stderr
sys.stderr = io.StringIO()

import tensorflow as tf
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0)

# Restore stderr
sys.stderr = old_stderr

# Now run main
from main import main

if __name__ == "__main__":
    main()
