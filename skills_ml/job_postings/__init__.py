import json

from cached_property import cached_property

from skills_ml.job_postings.corpora import SimpleCorpusCreator


class JobPosting(object):
    def __init__(self, job_posting_json, corpus_creator=None):
        self.job_posting_json = job_posting_json
        self.properties = json.loads(self.job_posting_json.decode('utf-8'))
        if corpus_creator:
            self.corpus_creator = corpus_creator
        else:
            self.corpus_creator = SimpleCorpusCreator()

    @cached_property
    def text(self):
        return self.corpus_creator._join(self.properties)

    @cached_property
    def id(self):
        return self.properties['id']

    def __getattr__(self, attr):
        return self.properties.get(attr, None)
