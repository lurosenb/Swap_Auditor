
from algorithms.feldman.FeldmanAlgorithm import FeldmanAlgorithm
from algorithms.baseline.SVM import SVM
from algorithms.baseline.DecisionTree import DecisionTree
from algorithms.baseline.GaussianNB import GaussianNB
from algorithms.baseline.LogisticRegression import LogisticRegression
from algorithms.ParamGridSearch import ParamGridSearch

from metrics.DIAvgAll import DIAvgAll
from metrics.Accuracy import Accuracy
from metrics.MCC import MCC


ALGORITHMS = [
   SVM(), GaussianNB(), LogisticRegression(), DecisionTree(),     # baseline
#   SDBSVM(),                                                      # not yet confirmed to work
   FeldmanAlgorithm(SVM()), FeldmanAlgorithm(GaussianNB()),       # Feldman
   FeldmanAlgorithm(LogisticRegression()), FeldmanAlgorithm(DecisionTree()),
   ParamGridSearch(FeldmanAlgorithm(SVM()), DIAvgAll()),          # Feldman params
   ParamGridSearch(FeldmanAlgorithm(SVM()), Accuracy()),
   ParamGridSearch(FeldmanAlgorithm(GaussianNB()), DIAvgAll()),
   ParamGridSearch(FeldmanAlgorithm(GaussianNB()), Accuracy())
]

def add_algorithm(algorithm):
    ALGORITHMS.append(algorithm)
