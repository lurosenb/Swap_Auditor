import fire
import pandas as pd
import pathlib
import sys
import subprocess

from ggplot import *

from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import TAGS

from analysis import run

files = ['education_Race_original.csv',
'education_Race-Sex_original.csv',
'education_Sex_original.csv']

run(dataset = ['education'],  filenames=files)