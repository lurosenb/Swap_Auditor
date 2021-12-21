
# Algorithms included (alphabetically)

## Feldman et al.

### Code source:
https://github.com/algofairness/BlackBoxAuditing/

also available via:
> pip3 install BlackBoxAuditing

### Papers to cite:
Numerical data:

Michael Feldman, Sorelle A. Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. Certifying and Removing Disparate Impact. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2015. http://arxiv.org/abs/1412.3756

Categorical data:

Philip Adler, Casey Falk, Sorelle A. Friedler, Gabriel Rybeck, Carlos Scheidegger, Brandon Smith, and Suresh Venkatasubramanian. Auditing Black-box Models for Indirect Influence. In Proceedings of the IEEE International Conference on Data Mining (ICDM), 2016.  https://arxiv.org/abs/1602.07043

# Adding a new algorithm

1. Make a new directory named after the first author of the relevant paper.
2. In the new directory create a file named *FirstAuthor*Algorithm.py that extends Algorithm.py and implements its run method.  Read through the other methods and implement any necessary for your algorithm.
3. Add any additional needed code in that directory or a subdirectory.
4. Add the algorithm to list.py.  Be sure to also add the ParamGridSearch version(s) of your algorithm if your algorithm has a parameter that can be used for tuning.
5. Add code source, citation, and any additional site information to this README.
6. Add a LICENSE.txt to the new directory if your code is licensed under a *different* license from this repository.
