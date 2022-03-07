from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
import pandas as pd
import numpy as np


class SpanProcessor(DataProcessor):
    def __init__(self, dtype='cad'):
        super(SpanProcessor, self).__init__()
        self.dtype = dtype
        if dtype == 'cad':
            self.m1 = 'sentence1'
            self.m2 = 'sentence2'
            self.gold_label = 'gold_label'
        elif dtype == 'imdb':
            self.t = 'Text'
            self.gold_label = 'Sentiment'
        elif dtype == 'boolq':
            self.p = 'passage'
            self.q = 'question'
            self.gold_label = 'gold_label'
        self.labels = ['NP', 'P']
        
    
    def get_labels(self):
        return self.labels
    
    def get_examples(self, filepath):
        data = pd.read_csv(filepath, keep_default_na=False)
        examples = self._create_examples(data)
        return examples
    
    def _create_examples(self, labeled_examples):
        
        examples = []
        max_len = 0
        sum_len = 0
        for (idx, le) in labeled_examples.iterrows():
            guid = idx
            if self.dtype == 'cad':
                text_a, text_b = le[self.m1].replace('*',''), le[self.m2].replace('*','')
                label_1 = " ".join(['P' if '*' in word else 'NP' for word in le[self.m1].split(' ')])
                label_2 = " ".join(['P' if '*' in word else 'NP' for word in le[self.m2].split(' ')])
                label = label_1 + '\t' + label_2
            elif self.dtype == 'imdb':
                text_a, text_b = le[self.t].replace('*',''), None
                label = " ".join(['P' if '*' in word else 'NP' for word in le[self.t].split(' ')])
            elif self.dtype == 'boolq':
                text_a, text_b = le[self.q].replace('*',''), le[self.p]
                label_1 = " ".join(['P' if '*' in word else 'NP' for word in le[self.q].split(' ')])
                label_2 = " ".join(['<pad>' for word in le[self.p].split(' ')])
                label = label_1 + '\t' + label_2
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class QaProcessor(DataProcessor):
    def __init__(self):
        super(QaProcessor, self).__init__()
        self.p = 'passage'
        self.q = 'question'
        self.gold_label = 'hard_label'
        self.labels = [True, False]
    
    def get_examples(self, filepath):
        data = pd.read_csv(filepath, keep_default_na=False)
        examples = self._create_examples(data)
        return examples
    
    def get_labels(self):
        return self.labels
    
    def _create_examples(self, labeled_examples):
        
        examples = []
        for (idx, le) in labeled_examples.iterrows():
            guid = idx
            label = le[self.gold_label]
            text_a, text_b = le[self.q], le[self.p]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        
        return examples

class SenProcessor(DataProcessor):
    def __init__(self):
        super(SenProcessor, self).__init__()
        self.text = 'Text'
        self.gold_label = 'Sentiment'
        self.labels = ['positive', 'negative']
        
    def get_examples(self, filepath):
        data = pd.read_csv(filepath, sep='\t', keep_default_na=False)
        examples = self._create_examples(data)
        return examples
        
    def get_labels(self):
        return self.labels
    
    def _create_examples(self, labeled_examples):
        
        examples = []
        for (idx, le) in labeled_examples.iterrows():
            guid = idx
            label = le[self.gold_label].lower()
            text_a, text_b = le[self.text], None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            
        return examples

class ExpProcessor(DataProcessor):
    def __init__(self, two_class=False):
        super(ExpProcessor, self).__init__()
        self.s1 = 'sentence1'
        self.s2 = 'sentence2'
        self.index_col = "pairID"
        self.two_class = two_class
        self.labels = ["entailment", "contradiction", "neutral"]
        self.gold_label = "gold_label"
    def get_train_examples(self, filepath, data_format="instance", to_drop=[]):
        data = pd.read_csv(filepath, index_col=self.index_col, keep_default_na=False)
        examples = self._create_examples(data, 'train', data_format=data_format, to_drop=to_drop) 
        return examples

    def get_dev_examples(self, filepath, data_format="instance", to_drop=[]):
        data = pd.read_csv(filepath, index_col=self.index_col, keep_default_na=False)        
        examples = self._create_examples(data, 'dev', data_format=data_format, to_drop=to_drop) 
        return examples

    def get_labels(self):
        return self.labels

    data_formats = ["instance", "independent", "append", "instance_independent", "instance_append",
                        "all_explanation", "Explanation_1", "hyp_only"]
    #aggregate uses the same format as independent
    def _create_examples(self, labeled_examples, set_type, data_format="instance", to_drop=[]):
        """Creates examples for the training and dev sets."""
        if data_format not in self.data_formats:
            raise ValueError("Data format {} not supported".format(data_format))

        if 'explanation' in to_drop: to_drop = self.labels

        keep_labels = [True if l not in to_drop else False for l in self.labels]
        exp_names = ["{}_explanation".format(l) for l in self.labels]

        examples = []
        for (idx, le) in labeled_examples.iterrows():
            guid = idx
            label = le[self.gold_label]
            
            #For HANS
            if self.two_class and label == "non-entailment":
                label = "contradiction"

            if data_format in ["independent", "instance_independent"]:
                exp_text = [le[exp_name] if keep  else ""
                                for l, keep, exp_name in zip(self.labels, keep_labels, exp_names)]
            elif data_format in ["append", "instance_append"]:
                exp_text = " ".join(["{}: {}".format(l, le[exp_name]) if keep else ""
                                for l, keep, exp_name in zip(self.labels, keep_labels, exp_names)])

            if data_format == "instance":
                text_a, text_b = le[self.s1], le[self.s2]
            elif data_format == "hyp_only":
                text_a, text_b = le[self.s2], None
            elif data_format in ["Explanation_1", "all_explanation"]:
                text_a, text_b = le[data_format], None
            elif data_format in ["independent", "append"]:
                text_a, text_b = exp_text, None
            elif data_format in ["instance_independent", "instance_append"]:
                instance_text = "Premise: {} Hypothesis: {}".format(
                                    le[self.s1], le[self.s2]) if "instance" not in to_drop else "Premise: Hypothesis:"
                text_a, text_b = instance_text, exp_text

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def simple_accuracy(preds, labels, ignore_index=-100):
    return (preds[labels!=ignore_index] == labels[labels!=ignore_index]).mean()
    
def exp_compute_metrics(preds, labels, two_class=False):
    assert len(preds) == len(labels)
    #For HANS treat all contradiction, neutral as label 1 (non entailment)
    if two_class:
        preds[preds==2] = 1
        return {"acc_ent": simple_accuracy(preds[labels==0], labels[labels==0]), "acc_not_ent": simple_accuracy(preds[labels!=0], labels[labels!=0])}
    else:
        return {"acc": simple_accuracy(preds, labels)}
    
def compute_entropy(preds,labels,ignore_index=-100):
    valid_idx = (labels != ignore_index).astype(float)
    preds = preds - preds.max(axis=-1, keepdims=True)
    exp_preds = np.exp(preds)
    probs = exp_preds/exp_preds.sum(axis=-1, keepdims=True)
    log_probs = np.log(probs)
    entropy = np.sum(-probs * log_probs, axis=-1)
    normalized_entropy = np.sum(entropy * valid_idx, axis=-1) / np.sum(valid_idx, axis=-1)
    return np.mean(normalized_entropy), normalized_entropy