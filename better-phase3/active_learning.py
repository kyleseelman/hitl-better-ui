import re
import csv
import numpy as np
import pandas as pd
import os

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# spacy for lemmatization
import spacy

from nltk.corpus import stopwords

def active(req_id):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'says', 'say'])

    li = []
    #req_id = 'IR-T1-r1'

    dir = '../demo-data/{}'.format(req_id)
    MAX_DOCS = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]) - 1

    # NEED TO FIX, EACH SENTENCE IN EACH DOCUMENT IS TREATED AS DIFFERENT DOCUMENT
    for i in range(MAX_DOCS):
        df = pd.read_csv('../demo-data/{}/doc_{}.tsv'.format(req_id, i), sep='\t', index_col=None, header=0,quoting=csv.QUOTE_NONE)

        # This should combine all the sentences from one document, into a single document
        np_data = np.asarray(df['DocText'])
        temp = np_data.flatten()[0]
        for j in range(1, np_data.flatten().shape[0]):
            string = temp + np_data.flatten()[j]
            temp = string
        data = [{'QueryID': 2, 'TaskLabel':3, 'DocID': 10, 'DocText': string}]  
        fd = pd.DataFrame(data)
        #print(fd)

        li.append(fd)

    frame = pd.concat(li, axis=0, ignore_index=True)
    data = frame['DocText']
    #data = pd.read_csv('new-ui/data/IR-T1-r1/doc_{}.tsv'.format(0), sep='\t')
    #data = data['DocText']

    # Convert to list
    #ata = data.content.values.tolist()

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

        # split the sentence into words
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'PROPN'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=3, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

    doc_lda = lda_model[corpus]

    # Gives the most dominant topic for each sentence
    def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)


    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    most = df_topic_sents_keywords['Dominant_Topic'].mode()
    doc_numbers = df_dominant_topic['Document_No'].where(df_dominant_topic['Dominant_Topic'] == most[0])
    doc_numbers = doc_numbers[~np.isnan(doc_numbers)]
    
    
    print(np.asarray(doc_numbers)[0])
    id = np.asarray(doc_numbers)[0]

    
    
    #print(texts)
    #print(corpus)
    print(len(data))

    # Initialize an instance of tf-idf Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Generate the tf-idf vectors for the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    #print(cosine_sim)
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[int(id)]))
    print(sim_scores)
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    print(sim_scores)
    # Get the scores for 5 most similar movies
    sim_scores = sim_scores[1:49]
    print(sim_scores)
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    movie_indices.insert(0, int(id))
    #print(movie_indices)
    
    return movie_indices


