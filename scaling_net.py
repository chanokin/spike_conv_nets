import spynnaker8 as sim
import numpy as np
import mnist
import matplotlib.pyplot as plt
import plotting
import sys
import h5py
import os
import field_encoding as fe
from field_encoding import ROWS_AS_MSB
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

