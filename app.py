import json
import os
import re
from tempfile import tempdir
import sys
import importlib  


from lda import lda_cluster, update_topic
from active_learning import active
from jinja2 import Environment

import flask
from flask import Flask, render_template, request
#from testing_ui.active_learning import active
app = Flask(__name__)
jinja_env = Environment(extensions=['jinja2.ext.loopcontrols'])
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


import spacy
spacy_model_name = 'en_core_web_sm'
if not spacy.util.is_package(spacy_model_name):
    spacy.cli.download(spacy_model_name)
nlp = spacy.load(spacy_model_name)


MAX_DOCS = 50  # number of docs for annotation
# TASK_FILENAME = 'data/ir-hitl-performer-tasks.json'
TASK_FILENAME = 'ir-p2-hitl-performer-tasks.json'
ENTITY_FILENAME_TEMPLATE = 'data/{}/entity.txt'
TERM_FILENAME_TEMPLATE = 'data/{}/terms.tsv'
LABEL_FILENAME_TEMPLATE = 'data/{}/doc_{}.tsv'
#RESULTS_JSON_FILENAME = 'data/outputs_to_IE_fullHITL.json'
RESULTS_JSON_FILENAME = 'data/full-hitl-retrievals-outputToIE.json'
DATA_DIR = 'data'
SNIPPET_LEN = 5  # number of sentences in snippet

#UNLABELED_ENT_TEMPLATE = """
#<button type="submit" name="ent" class="btn btn-outline-dark btn-plain" value="{text}">{text}</button>
#"""
#LABELED_ENT_TEMPLATE = """
#<button type="submit" name="ent" class="btn btn-outline-dark btn-plain active" value="{text}">{text}</button>
#"""

UNLABELED_ENT_TEMPLATE = """
<button name="ent" type="button" class="btn btn-outline-dark btn-plain" value="{text}">{text}</button>
"""
LABELED_ENT_TEMPLATE = """
<button name="ent" type="button" class="btn btn-outline-dark btn-plain active" value="{text}">{text}</button>
"""


def process_data():
    with open(RESULTS_JSON_FILENAME, 'r') as f:
        results = json.load(f)
    for task in results:
        for req in task['taskRequests']:
            req_id = req['reqQueryID']
            req_dir = os.path.join(DATA_DIR, req_id)
            if not os.path.exists(req_dir):
                os.mkdir(req_dir)
                for doc_id, doc in enumerate(req['relevanceDocuments'][:MAX_DOCS]):
                    label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, doc_id)
                    with open(label_filename, 'w') as f:
                        print('\t'.join(['QueryID', 'TaskLabel', 'RequestLabel', 'DocID', 'DocText']), file=f)
                        sent_id = 0
                        doc_text = doc['docText']
                        doc_text = doc_text.replace(r'\n', ' ')
                        doc_text = doc_text.replace(r'\t', ' ')
                        doc_text = doc_text.replace(r'\r', ' ')
                        doc_text = re.sub(r'\\x..', '', doc_text)
                        for line in doc_text.split('\n'):
                            for sent in nlp(line.rstrip()).sents:
                                print('\t'.join([req_id, '', '', '{}_{}'.format(doc_id, sent_id), sent.text]), file=f)
                                sent_id += 1
                term_filename = TERM_FILENAME_TEMPLATE.format(req_id)
                with open(term_filename, 'w') as f:
                    terms = set()
                    print('\t'.join(['Term', 'Relevance']), file=f)
                    for term in req['reqQueryText'].strip().split():
                        if term not in terms:
                            print('\t'.join([term, 'checked']), file=f)
                            terms.add(term)

process_data()


def escape_html(text):
    """Replace <, >, &, " with their HTML encoded representation. Intended to
    prevent HTML errors in rendered displaCy markup.
    text (unicode): The original text.
    RETURNS (unicode): Equivalent text to be safely used within HTML.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text


def render_ents(text, labeled_ents):
    """Render entities in text."""
    markup = ""
    offset = 0
    for ent in nlp(text).ents:
        start = ent.start_char
        end = ent.end_char
        entity = escape_html(text[start:end])
        fragments = text[offset:start].split("\n")
        for i, fragment in enumerate(fragments):
            markup += escape_html(fragment)
            if len(fragments) > 1 and i != len(fragments) - 1:
                markup += "</br>"
        if entity.lower() in labeled_ents:
            markup += LABELED_ENT_TEMPLATE.format(text=entity)
        else:
            markup += UNLABELED_ENT_TEMPLATE.format(text=entity)
        offset = end
    fragments = text[offset:].split("\n")
    for i, fragment in enumerate(fragments):
        markup += escape_html(fragment)
        if len(fragments) > 1 and i != len(fragments) - 1:
            markup += "</br>"
    return markup


@app.route('/')
def main():
    print("testing")
    print("test", file=sys.stdout)
    with open(TASK_FILENAME, 'r') as f:
        tasks = json.load(f)
    reqs = [(req['req-num'], task['task-title'], req['req-text'], req['req-num'][:len(req['req-num'])-2]) for task in tasks for req in task['requests']]
    # if sending tasks, don't need ID, need ID for the requests
    #task_title = [(task['requests'][0]['req-num'], task['task-title']) for task in tasks]
    task_title = [(task['task-title'], task['requests'][0]['req-num'][:len(task['requests'][0]['req-num'])-2]) for task in tasks]
    #requests = [(req['req-num'], req['req-text']) for task in tasks for req in task['requests']]
    #print(task_title)
    #print(requests)
    #print(reqs)
    return render_template('main.html', reqs=reqs, title=task_title)

@app.route('/requests', methods=['GET', 'POST'])
def requests():
    print('in request')
    value = request.args.get('value')
    
    with open(TASK_FILENAME, 'r') as f:
        tasks = json.load(f)
    reqs = [(req['req-num'], req['req-text'], req['req-num'][:len(req['req-num'])-2]) for task in tasks for req in task['requests']]
    reqs = [item for item in reqs if item[2]==value]
    print(reqs)
    return flask.jsonify(req_id=[item[0] for item in reqs], req_text=[item[1] for item in reqs], value=[item[2] for item in reqs])

#@app.route('/request', methods=['GET', 'POST'])
#def request():
    print('in request')
    with open(TASK_FILENAME, 'r') as f:
        tasks = json.load(f)
    # get the req_text for each req_num here, 
    req_text = []
    for task in tasks:
        temp = []
        for req in task['requests']:
            temp.append(req['req-text'])
        req_text.append(temp)
    # list of list of the requests for each task, but causes failrure
    print(req_text)

    return flask.jsonify(req_text=req_text)

@app.route('/info', methods=['GET', 'POST'])
def info():
    print("In info")
    req_id = request.args.get('req_id')
    with open(TASK_FILENAME, 'r') as f:
        tasks = json.load(f)
    for task in tasks:
        if req_id.startswith(task['task-num']):
            req_text = ''
            for req in task['requests']:
                if req['req-num'] == req_id:
                    req_text = req['req-text']
            snippets = []
            for doc_id in range(MAX_DOCS):
                snippet = ''
                label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, doc_id)
                with open(label_filename, 'r') as f:
                    next(f)
                    n_sent = 0
                    for line in f:
                        if n_sent < SNIPPET_LEN:
                            snippet = snippet + ' ' + line.strip().split('\t')[-1]
                            n_sent += 1
                snippets.append((doc_id, snippet.strip() + ' ...'))
            return flask.jsonify(req_id=req_id, task_title=task['task-title'],
                                  task_stmt=task['task-stmt'], task_narr=task['task-narr'],
                                   task_in_scope=task['task-in-scope'], task_not_in_scope=task['task-not-in-scope'],
                                  req_text=req_text)
            #return render_template('info.html', req_id=req_id, task_title=task['task-title'],
            #                      task_stmt=task['task-stmt'], task_narr=task['task-narr'],
            #                       task_in_scope=task['task-in-scope'], task_not_in_scope=task['task-not-in-scope'],
            #                      req_text=req_text, snippets=snippets)

@app.route('/topic', methods=['GET', 'POST'])
def topic():
    print("In topic")
    req_id = request.args.get('req_id')
    al_flag = request.args.get('al_flag')
    #if request.is_ajax() and request.method == 'GET':
    print(req_id)
    # FROM HERE CALL THE TOPIC MODELING
    print(al_flag)
    #flag = True
    print("With Topic Modeling")
    doc_numbers, topics, lengths = lda_cluster(req_id)
    with open(TASK_FILENAME, 'r') as f:
        tasks = json.load(f)
    for task in tasks:
        if req_id.startswith(task['task-num']):
            req_text = ''
            for req in task['requests']:
                if req['req-num'] == req_id:
                    req_text = req['req-text']
            snippets = []
            topic = []
            count = 0
            idx = 0
            for doc_id in doc_numbers:
                doc_id = int(doc_id)
                snippet = ''
                label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, doc_id)
                with open(label_filename, 'r') as f:
                    next(f)
                    n_sent = 0
                    for line in f:
                        if n_sent < SNIPPET_LEN:
                            snippet = snippet + ' ' + line.strip().split('\t')[-1]
                            n_sent += 1
                snippets.append((doc_id, snippet.strip() + ' ...'))
                count += 1
                if count == lengths[idx]:
                    #print(idx)
                    topic.append((topics[idx], snippets))
                    idx += 1
                    snippets = []
                    count = 0
                
    
    #print(topic)
    print("returning snippets")
    #print([(topics, snippets)])

    return render_template('cluster.html', topic=topic, snippets=snippets, topics=topics)
    #return flask.jsonify(snippets=snippets, topics=topics)

@app.route('/docu', methods=['GET', 'POST'])
def docu():
    print("In Docu")
    req_id = request.args.get('req_id')
    flag = request.args.get('flag')
    al_flag = request.args.get('al_flag')
    print("HEREEEEE2")
    #if request.is_ajax() and request.method == 'GET':
    print(req_id)
    # FROM HERE CALL THE TOPIC MODELING
    print("HEREEEEE")
    print(al_flag)
    #flag = True
    if flag == '1':
        print("With Topic Modeling")
        doc_numbers, topics = lda_cluster(req_id)
        with open(TASK_FILENAME, 'r') as f:
            tasks = json.load(f)
        for task in tasks:
            if req_id.startswith(task['task-num']):
                req_text = ''
                for req in task['requests']:
                    if req['req-num'] == req_id:
                        req_text = req['req-text']
                snippets = []
                for doc_id in doc_numbers:
                    doc_id = int(doc_id)
                    snippet = ''
                    label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, doc_id)
                    with open(label_filename, 'r') as f:
                        next(f)
                        n_sent = 0
                        for line in f:
                            if n_sent < SNIPPET_LEN:
                                snippet = snippet + ' ' + line.strip().split('\t')[-1]
                                n_sent += 1
                    snippets.append((doc_id, snippet.strip() + ' ...'))
        print("returning snippets")
        return flask.jsonify(snippets=snippets)

    else:
        print("Without Topic Modeling" + flag)
    #req_id = 'IR-T1-r1'
        with open(TASK_FILENAME, 'r') as f:
            tasks = json.load(f)
        for task in tasks:
            if req_id.startswith(task['task-num']):
                req_text = ''
                for req in task['requests']:
                    if req['req-num'] == req_id:
                        req_text = req['req-text']
                snippets = []
                if al_flag == '1': doc_numbers = active(req_id)
                else: doc_numbers = list(range(MAX_DOCS))
                for doc_id in doc_numbers:
                    snippet = ''
                    label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, doc_id)
                    with open(label_filename, 'r') as f:
                        next(f)
                        n_sent = 0
                        for line in f:
                            if n_sent < SNIPPET_LEN:
                                snippet = snippet + ' ' + line.strip().split('\t')[-1]
                                n_sent += 1
                    snippets.append((doc_id, snippet.strip() + ' ...'))
        print("return snips")
        print(len(snippets))
        return flask.jsonify(snippets=snippets)

@app.route('/', methods=['GET', 'POST'])
def query():
    req_id = "IR-T1-r1"
    with open(TASK_FILENAME, 'r') as f:
        tasks = json.load(f)
    for task in tasks:
        if req_id.startswith(task['task-num']):
            req_text = ''
            for req in task['requests']:
                if req['req-num'] == req_id:
                    req_text = req['req-text']
            snippets = []
            for doc_id in range(MAX_DOCS):
                snippet = ''
                label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, doc_id)
                with open(label_filename, 'r') as f:
                    next(f)
                    n_sent = 0
                    for line in f:
                        if n_sent < SNIPPET_LEN:
                            snippet = snippet + ' ' + line.strip().split('\t')[-1]
                            n_sent += 1
                snippets.append((doc_id, snippet.strip() + ' ...'))
            return render_template('query.html', req_id=req_id, task_title=task['task-title'],
                                  task_stmt=task['task-stmt'], task_narr=task['task-narr'],
                                   task_in_scope=task['task-in-scope'], task_not_in_scope=task['task-not-in-scope'],
                                   req_text=req_text, snippets=snippets)


@app.route('/ui', methods=['GET', 'POST'])
def ui():
    req_id = request.args.get('req_id')
    doc_id = request.args.get('doc_id')
    print("At the ui")

    label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, doc_id)
    with open(label_filename, 'r') as f:
        header = next(f).strip()
        rows = [line.strip().split('\t') for line in f]

    ent_filename = ENTITY_FILENAME_TEMPLATE.format(req_id)
    if os.path.exists(ent_filename):
        with open(ent_filename, 'r') as f:
            labeled_ents = set(line.strip().lower() for line in f)
    else:
        labeled_ents = set()

    if request.method == 'POST':
        # save new entities
        if request.form.get('ent'):
            clicked_ent = request.form.get('ent').lower()
            if clicked_ent in labeled_ents:
                labeled_ents.remove(clicked_ent)
            else:
                labeled_ents.add(clicked_ent)
            with open(ent_filename, 'w') as f:
                for ent in labeled_ents:
                    print(ent, file=f)

        # save sentence labels
        with open(label_filename, 'w') as f:
            print(header, file=f)
            for req_id, task_label, req_label, sent_id, text in rows:
                if request.form.get('req_' + sent_id):
                    req_label = 'checked'
                    task_label = 'checked'
                else:
                    req_label = ''
                    if request.form.get('task_' + sent_id):
                        task_label = 'checked'
                    else:
                        task_label = ''
                print('\t'.join([req_id, task_label, req_label, sent_id, text]), file=f)

        # reload sentence labels to new version
        with open(label_filename, 'r') as f:
            next(f)
            rows = [line.strip().split('\t') for line in f]

    annotations = [(sent_id, task_label, req_label, render_ents(text, labeled_ents))
                   for _, task_label, req_label, sent_id, text in rows]
    return render_template('ui.html', annotations=annotations, req_id=req_id, doc_id=doc_id, max_docs=MAX_DOCS)


@app.route('/terms', methods=['GET', 'POST'])
def terms():
    print("In terms")
    #req_id = 'IR-T1-r1'
    req_id = request.args.get('req_id')
    print(req_id)
    term_filename = TERM_FILENAME_TEMPLATE.format(req_id)
    with open(term_filename, 'r') as f:
        next(f)
        rows = [line.strip('\n').split('\t') for line in f]

    if request.method == 'POST':
        # save sentence labels
        with open(term_filename, 'w') as f:
            print('\t'.join(['Term', 'Relevance']), file=f)
            for term, _ in rows:
                if request.form.get(term):
                    label = 'checked'
                else:
                    label = ''
                print('\t'.join([term, label]), file=f)
        # reload sentence labels to new version
        with open(term_filename, 'r') as f:
            next(f)
            rows = [line.strip('\n').split('\t') for line in f]

    return render_template('term.html', rows=rows)

@app.route('/update', methods=['GET', 'POST'])
def update():
    print("updateee")
    req_id = request.args.get('req_id')
    topic = request.args.get('topic')
    num = request.args.get('num')
    print(topic)
    print(num)
    tops = update_topic(req_id, topic)
    doc_numbers, topics, lengths = lda_cluster(req_id)

    print(tops)
    print(doc_numbers)
    temp = [int(x) for x in doc_numbers if x not in tops]
    print(temp)
    print(tops + temp)
    doc_numbers = tops+temp
    print(topics)
    topics[0] = topic
    print(topics)
    with open(TASK_FILENAME, 'r') as f:
        tasks = json.load(f)
    for task in tasks:
        if req_id.startswith(task['task-num']):
            req_text = ''
            for req in task['requests']:
                if req['req-num'] == req_id:
                    req_text = req['req-text']
            snippets = []
            topic = []
            count = 0
            idx = 0
            for doc_id in doc_numbers:
                doc_id = int(doc_id)
                snippet = ''
                label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, doc_id)
                with open(label_filename, 'r') as f:
                    next(f)
                    n_sent = 0
                    for line in f:
                        if n_sent < SNIPPET_LEN:
                            snippet = snippet + ' ' + line.strip().split('\t')[-1]
                            n_sent += 1
                snippets.append((doc_id, snippet.strip() + ' ...'))
                count += 1
                if count == lengths[idx]:
                    #print(idx)
                    topic.append((topics[idx], snippets))
                    idx += 1
                    snippets = []
                    count = 0

    print("returning snippets")
    #print([(topics, snippets)])

    return render_template('cluster.html', topic=topic, snippets=snippets, topics=topics)


    return 

if __name__ == '__main__':
    app.run()
