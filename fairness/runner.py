import fire
import os
import statistics
import sys

import results
from data.objects.list import DATASETS, get_dataset_names
from data.objects.ProcessedData import ProcessedData
from algorithms.list import ALGORITHMS
from metrics.list import get_metrics
from benchmark import run

from algorithms.ParamGridSearch import ParamGridSearch

def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result

algs=['SVM',
  'GaussianNB',
  'LR',
  'DecisionTree',
  'Feldman-SVM',
  'Feldman-GaussianNB',
  'Feldman-LR',
  'Feldman-DecisionTree']

run(num_trials = 1, dataset = ['education'],
        algorithm = algs)