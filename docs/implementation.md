# Implementation notes

## Neural models

`Conv1` implements the convolutional network, `Model2` the GRU, `Model3` the MLP,
and `Model4` the residual network. The CNN stores its feature count on the model
instance. It corresponds to the first `Conv1` definition in the research
notebook; a later notebook cell reused that name for an MLP.

## Fusion interface

The CLI accepts aligned prediction columns with explicit row identifiers and
finite, nonnegative weights. Score combination supports optional per-column
min-max normalization; constant columns receive 0.5. Rank combination averages
tied ranks. Weights are supplied by the caller and should be selected using
independent validation data.

The reference module preserves the research cognitive-diversity expressions
with fixed model-column names. The CLI provides prediction combination;
dataset preparation, model training, and calibration belong to the calling
experiment pipeline.
