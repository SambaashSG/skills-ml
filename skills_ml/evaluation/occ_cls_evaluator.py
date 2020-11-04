from sklearn import metrics
from cached_property import cached_property

import logging
import numpy as np

class ClassificationEvaluator(object):
    def __init__(self, result_generator):
        self.result_generator = result_generator
        if not hasattr(self.result_generator,'target_variable'):
            raise AttributeError("the result_generator should have target_variable property")
        else:
            self.target_variable = self.result_generator.target_variable
            self.labels = self.target_variable.choices
        self.result = np.array(list(self.result_generator))

    @cached_property
    def y_pred(self):
        return self.target_variable.encoder.inverse_transform(self.result[:, 0])

    @cached_property
    def y_true(self):
        return self.target_variable.encoder.inverse_transform(self.result[:, 1])

    @cached_property
    def accuracy(self):
        return metrics.accuracy_score(self.y_true, self.y_pred)

    @cached_property
    def precision(self):
        return metrics.precision_score(self.y_true, self.y_pred, labels=self.labels, average=None)

    @cached_property
    def recall(self):
        return metrics.recall_score(self.y_true, self.y_pred, labels=self.labels, average=None)

    @cached_property
    def f1(self):
        return metrics.f1_score(self.y_true, self.y_pred, labels=self.labels, average=None)

    @cached_property
    def confusion_matrix(self):
        return metrics.confusion_matrix(self.y_true, self.y_pred)

    @cached_property
    def macro_precision(self):
        return metrics.precision_score(self.y_true, self.y_pred, average='macro')

    @cached_property
    def micro_precision(self):
        return metrics.precision_score(self.y_true, self.y_pred, average='micro')

    @cached_property
    def macro_recall(self):
        return metrics.recall_score(self.y_true, self.y_pred, average='macro')

    @cached_property
    def micro_recall(self):
        return metrics.recall_score(self.y_true, self.y_pred, average='micro')

    @cached_property
    def macro_f1(self):
        return metrics.f1_score(self.y_true, self.y_pred, average='macro')

    @cached_property
    def micro_f1(self):
        return metrics.f1_score(self.y_true, self.y_pred, average='micro')


class OnetOccupationClassificationEvaluator(ClassificationEvaluator):
    def __init__(self, result_generator):
        super().__init__(result_generator)
        if not hasattr(self.result_generator, 'target_variable'):
            raise AttributeError("the result_generator should have target_variable property")
        else:
            self.target_variable = self.result_generator.target_variable
            self.labels = self.target_variable.choices

    @cached_property
    def _result_for_major_group(self):
        y_pred = [p[:2] for p in self.y_pred]
        y_true = [t[:2] for t in self.y_true]
        return y_true, y_pred

    @cached_property
    def accuracy_major_group(self):
        if self.target_variable.name == 'full_soc':
            y_true, y_pred = self._result_for_major_group
            return metrics.accuracy_score(y_true, y_pred)

        elif self.target_variable.name == 'major_group':
            return self.accuracy

    @cached_property
    def recall_per_major_group(self):
        if self.target_variable.name == 'major_group':
            return self.recall
        elif self.target_variable.name == 'full_soc':
            y_true, y_pred = self._result_for_major_group
            return metrics.recall_score(y_true, y_pred, average=None)

    @cached_property
    def precision_per_major_group(self):
        if self.target_variable.name == 'major_group':
            return self.precision
        elif self.target_variable.name == 'full_soc':
            y_true, y_pred = self._result_for_major_group
            return metrics.precision_score(y_true, y_pred, average=None)

    @cached_property
    def f1_per_major_group(self):
        if self.target_variable.name == 'major_group':
            return self.f1
        elif self.target_variable.name ==  'full_soc':
            y_true, y_pred = self._result_for_major_group
            return metrics.f1_score(y_true, y_pred, average=None)
