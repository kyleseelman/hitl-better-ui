import re
import csv
import numpy as np
import pandas as pd
import os

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

from nltk.corpus import stopwords


def lda_cluster(req_id):

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'say', 'say'])

    li = []
    
    dir = '../demo-data/{}'.format(req_id)
    MAX_DOCS = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]) - 1
    #print(MAX_DOCS)
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

    #  Topic distribution across documents
    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    #print(topic_counts)
    #print('most common topic')
    most = df_topic_sents_keywords['Dominant_Topic'].mode()
    #print(most[0])
    #print(df_dominant_topic['Dominant_Topic'])
    # find all doc_id's for most common topic
    #print(df_dominant_topic['Document_No'].where(df_dominant_topic['Dominant_Topic'] == most[0]))
    doc_numbers = df_dominant_topic['Document_No'].where(df_dominant_topic['Dominant_Topic'] == most[0])
    doc_numbers = doc_numbers[~np.isnan(doc_numbers)]
    topic_key = df_dominant_topic['Keywords'].where(df_dominant_topic['Dominant_Topic'] == most[0])
    
    # get the unique topics removing the most dominant
    unique_topics = np.asarray(df_dominant_topic['Keywords'])
    unique_topics = np.unique(unique_topics[unique_topics != topic_key])
    #print(unique_topics)
    # get the topic numbers for the different topics
    doc_nums_minority = []
    for topic in range(0,unique_topics.shape[0]):
        nums = df_dominant_topic['Document_No'].where(df_dominant_topic['Keywords'] == unique_topics[topic])
        nums = nums[~np.isnan(nums)]
        doc_nums_minority.append(nums)
    #print(doc_nums_minority)

    #topic_key = topic_key[~np.isnan(topic_key)]
    #print(np.asarray(topic_key))
    #print(np.asarray(topic_key)[0])

    # TRYING TO GET ACTIVE LEARNING TO WORK, NEED TO REMOVE NAN FROM TEXT
    #text = (df_dominant_topic['Text'].where(df_dominant_topic['Dominant_Topic'] == most[0])).to_numpy()
    #print(text[2])
    # Initialize an instance of tf-idf Vectorizer
    #tfidf_vectorizer = TfidfVectorizer()

    # Generate the tf-idf vectors for the corpus
    #tfidf_matrix = tfidf_vectorizer.fit_transform(text)
    #cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    # Sort the movies based on the similarity scores
    #sim_scores = sorted(cosine_sim, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    #sim_scores = sim_scores[1:5]
    #print(sim_scores)
    # Get the movie indices
    #movie_indices = [i[0] for i in sim_scores]

    #print("Main topic numbers")
    #print(np.asarray(doc_numbers))
    #print("Minority topics numbers")
    temp1 = np.asarray(doc_nums_minority[0])
    temp2 = np.asarray(doc_nums_minority[1])
    #print(temp1, temp2)
    # combines all the doc numbers, just will have to change topics when i+1 < i
    #print(np.hstack((doc_numbers, temp1,temp2)))
    nums = np.hstack((doc_numbers, temp1,temp2))
    lengths = [len(doc_numbers), len(temp1), len(temp2)]
    #print([topic_key[0]])
    #print([[topic_key[0]] , list(unique_topics)])
    temp = [[topic_key[0]] , list(unique_topics)]
    # all the topics in list format
    topics = [item for each in temp for item in each]
    print(topics)
    topics = [x for x in topics if x==x]
    topics = [x.replace('say, ', '') for x in topics]
    print(topics)
   
    print([i.split(',')[0] for i in topics])
    #print(len(topics))
    #print(nums.shape)
    # sends the document numbers for the docs in the topic and the first topic 
    return np.asarray(nums), [i.split(',')[0] for i in topics], lengths#np.asarray(topic_key)[0].split(',', 1)[0], unique_topics, doc_nums_minority


def update_topic(req_id, topic):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    li = []

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
        data = [{'QueryID': 2, 'TaskLabel':3, 'DocID': i, 'DocText': string}]  
        fd = pd.DataFrame(data)
        #print(fd)

        li.append(fd)

    frame = pd.concat(li, axis=0, ignore_index=True)
    #print(frame)
    #data = frame['DocText']

    # Remove distracting single quotes
    #data = [re.sub("\'", "", sent) for sent in data]
    tops = []
    for j in range(0, len(frame)):
        if topic in frame['DocText'][j].lower():
            tops.append(j)
    
    print(tops)
    return tops

#lda_cluster("IR-T1-r1")