from embedded_topic_model.utils import embedding
from embedded_topic_model.utils import data
from embedded_topic_model.utils import preprocessing
from embedded_topic_model.models.etm import ETM

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

import json
import re
import pandas as pd
import csv
import numpy as np
import os


def etm_cluster(req_id):
    def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'would', 'also'])

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
        #ls.append(texts_es[i])
    frame = pd.concat(li, axis=0, ignore_index=True)
    data = frame['DocText']
    #data = pd.read_csv('new-ui/data/IR-T1-r1/doc_{}.tsv'.format(0), sep='\t')
    #data = data['DocText']

    # Convert to list
    #ata = data.content.values.tolist()

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    #print(data)
    #corpus = spanish_stopwords(ls)
    corpus = remove_stopwords(data)
    #corpus = corpus + corpus_2
    #print(corpus)
    docs = []
    for j in range(0, len(corpus)):
        docs.append(' '.join(corpus[j]))
    #print(docs)
    #docs = li
    # Preprocessing the dataset
    vocabulary, train_dataset, test_dataset, = preprocessing.create_etm_datasets(
        docs,
        min_df=0.01,
        max_df=0.75,
        train_size=0.85,
    )

    # Training word2vec embeddings
    embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(docs)#, embedding_file_path = './en_embedding', save_c_format_w2vec=True)

    etm_instance = ETM(
        vocabulary,
        embeddings=embeddings_mapping, # You can pass here the path to a word2vec file or
                                    # a KeyedVectors instance
        num_topics=3,
        epochs=50,
        debug_mode=True,
        train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
                                # topic embeddings. By default, is False. If 'embeddings' argument
                                # is being passed, this argument must not be True
        eval_perplexity=False,

    )

    etm_instance.fit(train_dataset, test_dataset)

    topics = etm_instance.get_topics(10)
    topic_coherence = etm_instance.get_topic_coherence()
    topic_diversity = etm_instance.get_topic_diversity()
    document_topic = etm_instance.get_document_topic_dist()
    topic_word_matrix = etm_instance.get_topic_word_matrix()

    print("Document topic distribution")
    print(document_topic)
    print(document_topic.shape)
    #print("Topic-Word Matrix")
    #print(topic_word_matrix)
    print(topic_diversity)
    print(topic_coherence)
    print(topics)

    topic_per_doc = np.argmax(document_topic, axis=1)
    print(topic_per_doc)
    idx = []
    lengths = []
    #idx = np.array([])
    for k in range(0,3):
        temp = np.where(topic_per_doc == k) 
        idx.append(temp[0]) 
        #idx = np.concatenate((idx, temp))
        lengths.append(temp[0].shape[0])
    print(idx)
    print(lengths)
   # for j in range(0,len(idx)):
    #    tem
    
    return idx, [i[0] for i in topics], lengths 
