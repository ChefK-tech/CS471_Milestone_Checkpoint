# CS471_Milestone_Checkpoint

This model was written to try and identify if an aircraft's type (C172, B738, or CRJ2) can be predicted based on how it flies.
These types of aircraft were decided based on a preliminary analysis of the limited dataset.

The dataset for this model was primarily derived from the Colorado Spring's airport, pulling one day of data from adsb.lol and then splitting that dataset into both training and testing (80/20)

The main goal of this model is to see if a QDA model can be correctly trained on the data using a small list of five features and then correctly identify whether the type of aircraft for flights that it has never seen.

The goal is to train the QDA model on a supervised dataset and then test it on a supervised dataset as well to see if the model can correctly identify the aircraft type.

It is trained using per-point flight features and its accuracy is evaluated using a confusion matrix. 
