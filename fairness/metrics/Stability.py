import numpy
import pandas

from metrics.Metric import Metric

from .swap_auditor import RandomizedGroupSwapAuditor

class Stability(Metric):
    """
    
    """
    def __init__(self):
        Metric.__init__(self)
        self.name = 'Stability'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred, data=None, classifier=None):
        if data is not None:
            randomized_group = RandomizedGroupSwapAuditor(data=data, predictor=classifier, id_column="StudentId", protected_classes=['Sex','Race'], target_col='GradesUndergrad')
            randomized_group.calculate_all_stability(marginal_features=['Tutored','Socio-economicStatusQuartile','FathersWishes'], delta=0.1, epsilon=0.5, t=1) # t=1
            randomized_group._calculate_subgroup_stability()
            print(randomized_group.subgroup_stability)
            total = 0
            for sg in randomized_group.intersectional_classes:
                sg_key = randomized_group._sg_key(sg)
                s, marg = randomized_group._retrieve_stability_subgroup(sg_key)
                total+=s
            # print("\nFor subgroup: " + str(sg))
            # print('Stability: ' + str(s))
            stability = total/len(randomized_group.intersectional_classes)
            print(stability)
            return stability
        else:
            raise ValueError("Missing data")