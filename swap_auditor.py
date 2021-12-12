import pandas as pd
import numpy as np

from itertools import product, combinations
from functools import reduce
from operator import and_

class BaseSwapAuditor():
    """Baseclass for shared functionality between all swap auditors."""

    def __init__(self, data, predictor, id_column, protected_classes, target_col):
        # TODO: Add safety checks beyond data

        # Counter for individuals stability
        if not isinstance(data, pd.DataFrame): 
            raise ValueError("The 'data' field must be a pandas dataframe.")
        self.data = data
        self.predictor = predictor
        self.id_column = id_column
        self.target_col = target_col
        self.protected_classes = protected_classes

        self.individual_stability = {}
        self.subgroup_stability = {}
        self.all_marginals = None

        self.intersectional_classes = None
        self._calculate_intersectional_classes()

        self.subgroup_frames = None
        self._create_subgroup_datasets(data, self.intersectional_classes)

    def calculate_stability_individual(self, id, marginal_features):
        raise NotImplementedError

    def _calculate_intersectional_classes(self):
        # Create list of lists of possible values for each protected class
        values_for_classes = [self.data[c].unique() for c in self.protected_classes]
        self.intersectional_classes = [x for x in product(*values_for_classes)]

    def _calculate_marginal_subsets(self, marginal_features, k=None):
        # Calculates all subsets of features for use as marginals. 
        # TODO: Add a k cap on marginal length
        all_marginals = []
        if k is None:
            for i in range(1, len(marginal_features)+1):
                els = [list(x) for x in combinations(marginal_features, i)]
                all_marginals.extend(els)
        else:
            raise NotImplementedError
        self.all_marginals = all_marginals

    def _retrieve_subgroup_individual(self, sample, protected_classes):
        return tuple(int(sample.iloc[0][c]) for c in protected_classes)

    def _sg_key(self, sg):
        return ''.join([str(x) for x in sg])

    def _marginal_key(self, marginal):
        return ''.join(marginal)

    def _create_subgroup_datasets(self, data, subgroups):
        def apply_conditions(df, cond_list):
            return df[reduce(and_, cond_list)]

        list_subgroup_frames = {}
        for sg in subgroups:
            # For each subgroup, create a conditional list with their
            # protected class values
            conds = []
            for i, pc in enumerate(self.protected_classes): 
                condition = data[pc] == sg[i]
                conds.append(condition)

            subgroup_frame = apply_conditions(data, conds)
            list_subgroup_frames[self._sg_key(sg)] = subgroup_frame

            # Also, create subgroup stability tracker here
            self.subgroup_stability[self._sg_key(sg)] = (0, 0, {})

        self.subgroup_frames = list_subgroup_frames

    def _calculate_stability(self, original, frame):
        return (frame == original).sum()

    def _stringify_marginals(self, marginals, percent=True):
        string = "Marginals:\n\n"
        if percent:
            for m, vals in marginals.items():
                changed, total = vals
                string += m + ": " + str(changed/total)
                string += "\n"
        else:
            for m, vals in marginals.items():
                changed, total = vals
                string += m + ": " + "(" + str(changed) + ", " + str(total) + ")"
                string += "\n"
        return string

    def _calculate_subgroup_stability(self):
        for id, vals in self.individual_stability.items():
            sample = self.data.loc[self.data[self.id_column].isin([id])]
            subgroup = self._retrieve_subgroup_individual(sample, self.protected_classes)
            
            all_non_changes, all_total, marginal_map = self.subgroup_stability[self._sg_key(subgroup)]
            
            ind_non_changes, ind_total, ind_marginal_map = vals
            
            for marginal, m_vals in ind_marginal_map.items():
                a, b = m_vals
                if self._marginal_key(marginal) not in marginal_map:
                    marginal_map[self._marginal_key(marginal)] = (0,0)

                s, p = marginal_map[self._marginal_key(marginal)] 
                marginal_map[self._marginal_key(marginal)] = (s + a, p + b)
            
            self.subgroup_stability[self._sg_key(subgroup)] = (all_non_changes+ind_non_changes, all_total+ind_total, marginal_map)

    def _retrieve_stability(self, x, mappings, percent):
        stability = None
        pretty_print_marginals = ""
        if x not in mappings:
            raise ValueError("Individual not in stability tracker.")

        all_non_changes, all_total, marginal_map = mappings[x] 
        if percent:
            stability = all_non_changes/all_total
            pretty_print_marginals = self._stringify_marginals(marginal_map, percent=True)
        else:
            stability = (all_non_changes, all_total)
            pretty_print_marginals = self._stringify_marginals(marginal_map, percent=False)

        return stability, pretty_print_marginals

    def _retrieve_stability_individual(self, ind, percent=True):
        return self._retrieve_stability(ind, self.individual_stability, percent)

    def _retrieve_stability_subgroup(self, sg, percent=True):
        return self._retrieve_stability(sg, self.subgroup_stability, percent)

    def _track_metrics(self, stability, predictions, marginal, x, mappings, individual=True):
        if individual:
            all_non_changes, all_total, marginal_map = mappings[x] 
        else:
            all_non_changes, all_total, marginal_map = mappings[self._sg_key(x)]

        all_non_changes += stability
        all_total += len(predictions)

        if self._marginal_key(marginal) not in marginal_map:
            marginal_map[self._marginal_key(marginal)] = (0,0)
        s, p = marginal_map[self._marginal_key(marginal)] 
        marginal_map[self._marginal_key(marginal)] = (s + stability, p + len(predictions))
        
        if individual:
            mappings[x] = (all_non_changes, all_total, marginal_map)
        else:
            mappings[self._sg_key(x)] = (all_non_changes, all_total, marginal_map)

class NaiveSwapAuditor(BaseSwapAuditor):
    """Naive swap auditor. 
    Provides functionality to compute all of the swaps for each individual.
    Approx O(n^2*(2^k-1)) time. Slow - n=1000, |pc|=2, |kmarginal|=3 ~45 mins.
    """

    def __init__(self, data, predictor, id_column, protected_classes, target_col):
        # Calculate the intersectional classes and feature subsets for marginals
        super().__init__(data, predictor, id_column, protected_classes, target_col)

    def calculate_stability_individual(self, id):
        # Split data into individual and rest
        sample = self.data.loc[self.data[self.id_column].isin([id])]

        if id not in self.individual_stability:
            self.individual_stability[int(id)] = (0, 0, {})
            # print(int(id))

        #Original prediction
        prediction_cols = sample.columns.difference([self.id_column]+[self.target_col])
        original = self.predictor.predict(sample[prediction_cols])

        # print("Original prediction: " + str(original))

        subgroup = self._retrieve_subgroup_individual(sample, self.protected_classes)

        for sg in self.intersectional_classes:
            if subgroup != sg:
                sg_frame = self.subgroup_frames[self._sg_key(sg)]

                if len(sg_frame) == 0:
                    continue
                
                # Here we create a copy of sgdataframe, and replace all rows values
                # with x's values except marginal. Essentially, we are left
                # with a bunch of duplicate, swapped x's
                for marginal in self.all_marginals:
                    x_copy = sg_frame.copy()
                    
                    x_repeated = sample.loc[sample.index.repeat(len(x_copy))]

                    columns_to_reassign = x_copy.columns.difference(marginal)

                    x_copy.loc[:,columns_to_reassign] = x_repeated.loc[:,columns_to_reassign].values
                    
                    predictions = self.predictor.predict(x_copy[prediction_cols])
                    stability = self._calculate_stability(original, predictions)
                    
                    # Do individual metric tracking
                    self._track_metrics(stability, predictions, marginal, int(id), self.individual_stability, individual=True)
                    
                    # Do groupwise metric tracking
                    # NOTE: Removing this here, as it can be post calculated for Naive but not randomized. Makes
                    # For a fair comparison not to factor in runtime of this operation
                    # self._track_metrics(stability, predictions, marginal, sg, self.subgroup_stability, individual=False)

                    del x_copy

    def calculate_all_stability(self, marginal_features):
        # Calculate the intersectional classes and feature subsets for marginals
        self._calculate_marginal_subsets(marginal_features)

        for _, row in self.data.iterrows():
            self.calculate_stability_individual(row[self.id_column])

class RandomizedSamplingSwapAuditor(BaseSwapAuditor):
    """Randomized swap auditor. 
    Runs randomized swapping process for t iterations to 
    achieve stability estimate within epsilon of the true value
    with 1-delta probability. Defaults to a +-0.1 approximation,
    with 90% success probability.
    """

    def __init__(self, data, predictor, id_column, protected_classes, target_col):
        # Calculate the intersectional classes and feature subsets for marginals
        super().__init__(data, predictor, id_column, protected_classes, target_col)

    def calculate_stability_individual(self, id, t):
        
        # Split data into individual and rest
        sample = self.data.loc[self.data[self.id_column].isin([id])]

        if id not in self.individual_stability:
            self.individual_stability[int(id)] = (0, 0, {})
            # print(int(id))

        #Original prediction
        prediction_cols = sample.columns.difference([self.id_column]+[self.target_col])
        original = self.predictor.predict(sample[prediction_cols])

        # print("Original prediction: " + str(original))

        subgroup = self._retrieve_subgroup_individual(sample, self.protected_classes)

        # Create a frame with all other samples to draw from
        non_sg_frames = []
        for sg in self.intersectional_classes:
            if subgroup != sg:
                sg_frame = self.subgroup_frames[self._sg_key(sg)]
                non_sg_frames.append(sg_frame)
        all_other_frames = pd.concat(non_sg_frames)
        
        # For each marginal 
        for marginal in self.all_marginals:
            # "For t iterations" - equivalent to t samples of other frame
            other_subgroup_samples = all_other_frames.sample(n=t, replace=True)

            x_repeated = sample.loc[sample.index.repeat(len(other_subgroup_samples))]

            columns_to_reassign = other_subgroup_samples.columns.difference(marginal)

            other_subgroup_samples.loc[:,columns_to_reassign] = x_repeated.loc[:,columns_to_reassign].values
            
            predictions = self.predictor.predict(other_subgroup_samples[prediction_cols])
            stability = self._calculate_stability(original, predictions)
            
            # Do individual metric tracking
            self._track_metrics(stability, predictions, marginal, int(id), self.individual_stability, individual=True)

            del other_subgroup_samples


    def calculate_all_stability(self, marginal_features, delta=0.1, epsilon=0.1):
        # Calculate the intersectional classes and feature subsets for marginals
        self._calculate_marginal_subsets(marginal_features)

        n = len(self.data)

        # Calculate number of t iterations with delta, epsilon, n
        # TODO: Verify this iteration calculation
        t = int(np.ceil((np.log((2*n)/delta))/(epsilon**2)))
        print("Iterations: " + str(t))
        
        for _, row in self.data.iterrows():
            self.calculate_stability_individual(row[self.id_column], t)

class RandomizedGroupSwapAuditor(BaseSwapAuditor):
    """Randomized swap auditor. 
    Runs randomized swapping process for t iterations to 
    achieve stability estimate within epsilon of the true value
    with 1-delta probability. Defaults to a +-0.1 approximation,
    with 90% success probability.
    """

    def __init__(self, data, predictor, id_column, protected_classes, target_col):
        # Calculate the intersectional classes and feature subsets for marginals
        super().__init__(data, predictor, id_column, protected_classes, target_col)

    def calculate_stability_individual(self, id):
        pass

    def run_group_experiment(self):
        pass

    def calculate_all_stability(self, marginal_features, delta=0.1, epsilon=0.1):
        # Calculate the intersectional classes and feature subsets for marginals
        self._calculate_marginal_subsets(marginal_features)

        n = len(self.data)

        # Calculate number of t iterations with delta, epsilon, n
        # TODO: Verify this iteration calculation
        t = int(np.ceil((np.log((2*n)/delta))/(epsilon**2)))
        print("Iterations: " + str(t))

        for marginal in self.all_marginals:
            for i in range(t):
                pass 

        # df.groupby(['FactorA', 'FactorB']).apply(lambda grp: grp.sample(n=2))