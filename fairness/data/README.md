
## Data sources and citations

### Education Data Set
United States Department of Education. National Center for Education Statistics. National Education Longitudinal Study, 1988. [distributor], 2006-01-18. https://doi.org/10.3886/ICPSR09389.v1
https://nces.ed.gov/surveys/nels88/

## Adding a data set

To add a data set, you need to:
1. Choose a single word lower case *name* to identify your data set.
2. Put the raw data set in the raw/ directory at *name*.csv.  Add any data info at *name*.txt.
3. Create a class *Name*.py that extends Data.py and implements all the required methods and fill in the fields.  Add it to objects/
4. Add your dataset object to the list at objects/list.py


## Generating the preprocessed versions of the data

All preprocessed versions of the data should be committed to the preprocessed directory.
To regenerate them, run:
> python3 preprocess.py
