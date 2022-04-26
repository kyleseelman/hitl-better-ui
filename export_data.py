"""Export annotations as a JSON file."""
import json
from app import RESULTS_JSON_FILENAME, LABEL_FILENAME_TEMPLATE, ENTITY_FILENAME_TEMPLATE, MAX_DOCS

OUTPUT_FILENAME = 'data/hitl_annotation.json'


def find_docid(results, req_id, rank):
    for task in results:
        for req in task['taskRequests']:
            if req['reqQueryID'] == req_id:
                for doc in req['relevanceDocuments']:
                    if doc['docRank'] == rank:
                        return doc['docID']
    raise Exception('Could not find document for {} rank {}'.format(req_id, rank))


def main():
    with open(RESULTS_JSON_FILENAME, 'r') as f:
        results = json.load(f)
    data = []
    for task in range(1, 11):
        task_id = 'IR-T{}'.format(task)
        for req in range(1, 3):
            req_id = '{}-r{}'.format(task_id, req)
            for rank in range(MAX_DOCS):
                label_filename = LABEL_FILENAME_TEMPLATE.format(req_id, rank)
                with open(label_filename, 'r') as f:
                    next(f)  # skip header
                    for line in f:
                        fields = line.strip().split('\t')
                        if fields[1] == 'checked' or fields[2] == 'checked':
                            example = {'text': fields[4], 'type': 'sentence'}
                            if fields[1] == 'checked':
                                example['task'] = task_id
                            if fields[2] == 'checked':
                                example['request'] = req_id
                            example['doc-id'] = find_docid(results, req_id, rank + 1)
                            data.append(example)
            entity_filename = ENTITY_FILENAME_TEMPLATE.format(req_id)
            with open(entity_filename, 'r') as f:
                for line in f:
                    entity = line.strip()
                    if entity:  # skip empty line
                        example = {
                            'text': entity,
                            'type': 'entity',
                            'task': task_id,
                            'request': req_id
                        }
                        data.append(example)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()
