####### Mobiuscore Data Processing ######
from moana.dataset.repo import DatasetRepo
from mobiuscore.doc.structures.sentence import Sentence
import spacy
import json
from torchtext.data import TabularDataset, Field

MIMIC_DATASET = 'n2c2/test/tokenized_spacy'
MIMIC_GRAMMAR = {('ADE', 'DRUG'): 'ADE-DRUG',
                 ('DOSAGE', 'DRUG'): 'DOSAGE-DRUG',
                 ('DURATION', 'DRUG'): 'DURATION-DRUG',
                 ('FORM', 'DRUG'): 'FORM-DRUG',
                 ('FREQUENCY', 'DRUG'): 'FREQUENCY-DRUG',
                 ('REASON', 'DRUG'): 'REASON-DRUG',
                 ('ROUTE', 'DRUG'): 'ROUTE-DRUG',
                 ('STRENGTH', 'DRUG'): 'STRENGTH-DRUG'}

NLP = spacy.load('en_core_web_sm')

def spacy_get_vector(word):
    return NLP(word).vector.tolist()

def find_pos(sentence, head, tail):
    # find index of entity
    def find(sentence, entity):
        p = sentence.find(' ' + entity + ' ')
        if p == -1:
            if sentence[:len(entity) + 1] == entity + ' ':
                p = 0
            elif sentence[-len(entity) - 1:] == ' ' + entity:
                p = len(sentence) - len(entity)
            else:
                p = 0
        else:
            p += 1
        return p

    sentence = ' '.join(sentence.split())
    p1 = find(sentence, head)
    p2 = find(sentence, tail)
    words = sentence.split()
    num_words = len(words)
    cur_pos = 0
    pos1 = -1
    pos2 = -1
    # cur_pos is char index, pos1 pos2 are token index
    for i, word in enumerate(words):
        if cur_pos == p1:
            pos1 = i
        if cur_pos == p2:
            pos2 = i
        cur_pos += len(word) + 1
    return pos1, pos2

def get_mobius_dataset(dataset_name, grammar, get_vector=spacy_get_vector, verbose=True):
    dr = DatasetRepo()
    mobius_dataset = dr.get(dataset_name)
    dataset = []
    for i, doc in enumerate(mobius_dataset):
        if i > 10:
            break
        if verbose:
            print('Processing Doc:{}'.format(i))
        num_spans = len(doc.annotations.span_annotations)
        for span1_index in range(num_spans-1):
            for span2_index in range(span1_index, num_spans):
                span1 = doc.annotations.span_annotations[span1_index]
                span2 = doc.annotations.span_annotations[span2_index]
                if (span1.label.value, span2.label.value) in grammar.keys() or (span2.label.value, span1.label.value) in grammar.keys():
                    example = process_span_pair(span1, span2, doc)
                    dataset.append(example)
    return dataset



def process_span_pair(span1, span2, doc):
    text, words = get_text_and_tokens(span1, span2, doc)
    pos1, pos2 = find_pos(text, span1.text, span2.text)
    label = None
    for relation in span1.relations_to:
        if span2 == relation.annotation_from:
            if label is not None and label != relation.label.value:
                raise ValueError('Multiple relations present \
                        between\nspan1:\n{}\nspan2:\n{}'
                        .format(span1, span2))
            label = relation.label.value
    for relation in span1.relations_from:
        if span2 == relation.annotation_to and label != relation.label.value:
            if label is not None:
                raise ValueError('Multiple relations present \
                        between\nspan1:\n{}\nspan2:\n{}'
                        .format(span1, span2))
            label = relation.label.value
    if label is None:
        label = 'NA'
    example = {'text': text, 'pos1':pos1,
               'pos2': pos2, 'relation': label}
    return example

def get_text_and_tokens(span1, span2, doc):
    start_char_idx = min(span1.start, span2.start)
    end_char_idx = min(span1.end, span2.end)
    first_sent = None
    sents = []
    # Get sents in either span or inbetween
    for sent in doc.structures.sentences:
        if sent.end >= start_char_idx and sent.start < end_char_idx:
            sents.append(sent)
    sents.sort(key=lambda s: s.start)
    tokens = [tok.text for sent in sents for tok in sent.tokens]
    text = ' '.join(tokens)
    return text, tokens

def load_dataset(path):
    text_field = Field(batch_first=True, include_lengths=True)
    char_field = Field(tokenize=lambda x: list(x), batch_first=True, include_lengths=True)
    pos1_field = Field(sequential=False, batch_first=True)
    pos2_field = Field(sequential=False, batch_first=True)
    label_field = Field(sequential=False, batch_first=True)
    train_data = TabularDataset(path=path,
                                          format='json',
                                          fields={'text':[('text', text_field), ('chars', char_field)],
                                                  'pos1':('pos1', pos1_field),
                                                  'pos2':('pos2', pos2_field),
                                                  'relation':('relation', label_field)})
    text_field.build_vocab(train_data)
    char_field.build_vocab(train_data)
    return train_data

def write_dataset(dataset, path):
    f = open(path, 'w')
    for ex in dataset:
        f.write(json.dumps(ex) + '\n')
    f.close()


if __name__ == '__main__':
    #nre_dataset = get_mobius_dataset(MIMIC_DATASET, MIMIC_GRAMMAR)
    #write_dataset(nre_dataset, 'mimic/torchtest.json')
    train_data = load_dataset('mimic/torchtest.json')



