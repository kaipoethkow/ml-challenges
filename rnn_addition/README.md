## Lo-fi Solution for the Adding Problem using RNNs

This is a solution to the adding problem for sequences where two
elements out of a sequence of random float values must be learned
to be correctly summed up by a model.

The adding problem is discussed in detail in "A Simple Way to
Initialize Recurrent Networks of Rectified Linear Units" by
Le et al. This implementation was developed on an outdated laptop
computer so we demonstrate the approach with fewer training
samples, shorter sequences and fewer training epochs compared
to the paper. It should be adaptable to larger data sets and more
computing power straightforwardly.

### Installation
With your venv activated, navigate to the extracted solution directory
and then run
```
pip install .
```

### Running

The models can be created and the results inspected using a notebook
`jupyter notebook sequence_addition.ipynb`.

A rendered version of the notebook is also included as a PDF file
`sequence_addition.pdf`.
