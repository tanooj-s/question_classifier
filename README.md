Binary classifier that can be used to distinguish between sentences that are questions vs sentences that are not questions. This can be considered a 'bag of tags' model, in that as opposed to a count vector defined by the frequencies of unique words in the entire corpus, it simply maps each word in a document to its corresponding part-of-speech tag and creates a count vector of the unique part-of-speech tags instead. This reduces the dimensionality of the matrix down from (~10000,N) to (39,N) (where N is the number of separate documents. 

Dependencies - keras, SpaCy, scikit-learn, numpy, pandas. 

In order to use this model, first run the clean_data.py script (this converts the test data to the bag of tags matrix), then run the model.py script. 

To use this on your own data, replace the test-inputs.txt file with your data, with each new document on a new line. If you want to replace the training data as well, run the clean_data script and modify line 13 to read in new training data and line 56 to write to 'train_clean.csv'. 
