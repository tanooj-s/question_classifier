import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer



# read in test data, get spaCy part of speech tags
# create a bag of tags which we can use for input to the model 
print("Loading spaCy English language model...")
nlp = spacy.load('en_core_web_sm')

df = pd.read_csv('data/test-inputs.txt',sep='\n')
df.columns = ['Sentence']

def get_tags(s):
	doc = nlp(s)
	tags = ''
	for token in doc:
		if token.pos_ not in ['PUNCT','X']:
			tags = tags + token.tag_ + ' '
	return tags

def wh_end_start(s):
	s = s.split()
	wh_dict = {'what','why','when','where','how','which','who','whom'}
	if s[0] in wh_dict:
		return 1
	elif s[1] in wh_dict:
		return 1
	elif s[-1] in wh_dict:
		return 1
	elif s[-2] in wh_dict:
		return 1
	return 0

print("Getting part of speech tags...")
df['tags'] = df['Sentence'].apply(get_tags)

print("Converting parts of speech to bag-of-tags matrix...")
#get counts of parts-of-speech
vectorizer = CountVectorizer()
tags_corpus = df['tags']
tag_count_vectors = vectorizer.fit_transform(tags_corpus)
# convert from sparse matrix format to ndarray
tag_count_vectors = tag_count_vectors.toarray()

# build out a new dataframe holding the clean test data using bag of tags in tag_count_vectors
test_df = pd.DataFrame(tag_count_vectors, columns=vectorizer.get_feature_names())
test_df['Sentence'] = df['Sentence']
test_df['wh_end_start'] = test_df['Sentence'].apply(wh_end_start)
test_df['num_words'] = test_df['Sentence'].apply(lambda s: len(s.split()))

#test_df['Other feature'] = df['other feature'] add other additional features to the first dataframe

test_df.to_csv('data/test_clean.csv')
print("Test data ready for model.")