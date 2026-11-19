# Source lineage

Selected neural definitions were extracted from the original project without
executing its top-level dataset reads, plotting, or training cells.

The first Conv1 definition is retained as the CNN; a later notebook cell reused that name for an MLP. Its feature count is stored on the model instance instead of a notebook global.

The portable fusion interface adds input validation, arbitrary model-column
names, explicit row identifiers, finite nonnegative weights, stable handling
of ties, and explicit behavior for constant columns. These are release-level
engineering changes, not evidence of matching archival benchmark outputs.
The reference module retains the source's cognitive-diversity expressions.
Full protocol reproduction requires separately establishing authoritative data
preparation, model configuration, calibration, and evaluation splits.
