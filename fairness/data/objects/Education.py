from data.objects.Data import Data
	
class Education(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'education'
        self.class_attr = 'GradesUndergrad'
        self.positive_class_val = 1
        self.sensitive_attrs = ['Race', 'Sex']
        self.privileged_class_names = ["White", "Male"]
        self.categorical_features = [
        'StandardizedTestQuartile','TimeSpentOnHomeworkInSchool',
        'TimeSpentOnHomeworkOutOfSchool','TimeSpentOnExtracurriculars',
        'ParentsHighestLevelEducation','Socio-economicStatusQuartile',
        'Tutored','FathersWishes','MothersWishes']
        self.features_to_keep = [ 'Sex','Race','CompositeGrades',
        'StandardizedTestQuartile','TimeSpentOnHomeworkInSchool',
        'TimeSpentOnHomeworkOutOfSchool','TimeSpentOnExtracurriculars',
        'ParentsHighestLevelEducation','Socio-economicStatusQuartile',
        'Tutored','FathersWishes','MothersWishes','GradesUndergrad'] #StudentId
        self.missing_val_indicators = ['?']

