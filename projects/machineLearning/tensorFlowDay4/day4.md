# text_id_neuralNet.py

Again, we're using tensorFlow and keras but this time it's to load the imdb movie review database. This will be used to make a model that will be trained to judge between a negative and positive review.

The model assigns each word in the review to an integer
(i.e. "the movie was the best ever" = 1 2 3 1 5 6 )

The integers are then turned into tensors for the model... I'm not quite sure how exactly the model uses these vectors from the dictionary to make its predictions though
