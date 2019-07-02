# datacura

Dataset iterators that generate batches adversarially.

The intuition is that training is the process of squeezing generalization out of the dataset.  Each point of training accuracy you gain is now used up for the sake of improving your validation accuracy.  You want to leave as much slack in your training accuracy as possible, while still learning, by composing difficult batches.

Datacura stands for dataset curation.
