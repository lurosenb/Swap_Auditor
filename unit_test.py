# TODO: Add some unit tests to make sure helper methods for the auditors are all working properly

# x = tiny_test.loc[tiny_test["StudentId"].isin([728513])]
# x_copy = tiny_test.copy()

# x_repeated = x.loc[x.index.repeat(len(x_copy))]

# columns_to_reassign = tiny_test.columns.difference(['Tutored','ParentsCheckHomework','FathersWishes'])

# print(x_copy.shape)
# print(x_repeated.shape)
# x_copy.loc[:,columns_to_reassign] = x_repeated.loc[:,columns_to_reassign].values
# x_copy