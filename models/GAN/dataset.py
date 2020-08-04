from GAN.Config import cfg
import os
import pandas as pd
import numpy as np


def load_data(path):
	with open(path) as fp:
		lines = fp.readlines()
		data = np.array([line.split() for line in lines]).astype("float64")
	# data.loc[:, "label"] = 1
	return data

