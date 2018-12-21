####### Mobiuscore Data Processing ######
from mobiuscore.dataset.serialize.json import DatasetJsonlSerializer
from mobiuscore.doc.structures.sentence import Sentence
from mobiuscore.doc.annotations.relation import Relation
import spacy
import json
import torch
from torchtext.data import Dataset, Field, NestedField, Example, BucketIterator
from torchtext.vocab import Vocab
from collections import Counter
import pickle
import numpy as np
import os
import models

SHORT = True
#MIMIC_DATASET = 'n2c2/train/tokenized_spacy'
#MIMIC_DATASET = '/efs/sid/mobius_data/mimic'
MIMIC_DATASET = '/Users/sidsachdeva/roam/data/mimic'
OUTPUT_PATH = 'output'
MIMIC_GRAMMAR = {('ADE', 'DRUG'): 'ADE-DRUG',
                 ('DOSAGE', 'DRUG'): 'DOSAGE-DRUG',
                 ('DURATION', 'DRUG'): 'DURATION-DRUG',
                 ('FORM', 'DRUG'): 'FORM-DRUG',
                 ('FREQUENCY', 'DRUG'): 'FREQUENCY-DRUG',
                 ('REASON', 'DRUG'): 'REASON-DRUG',
                 ('ROUTE', 'DRUG'): 'ROUTE-DRUG',
                 ('STRENGTH', 'DRUG'): 'STRENGTH-DRUG'}
MAX_SENTS = 5


def find_pos(words, span1, span2):
    num_tokens = 0
    pos1 = -1
    pos2 = -1
    for i, tok in enumerate(words):
        if tok.end >= span1.start and tok.start <= span1.end:
            pos1 = num_tokens
        if tok.end >= span2.start and tok.start <= span2.end:
            pos2 = num_tokens
        num_tokens += 1

    if pos1 is -1 or pos2 is -1:
        print('CANT FIND')
        print(pos1)
        print(pos2)
        print(span1)
        print(span2)
        print([(x.text, x.start, x.end) for x in words])
        assert False
    return pos1, pos2

def get_mobius_dataset(dataset_path, grammar=MIMIC_GRAMMAR, verbose=True):
    datasets = {}
    vocab = {f: Counter() for f in ['text', 'chars', 'pos1', 'pos2', 'pos1_rel', 'pos2_rel', 'relation', 'rel_type']}
    for split in ['train', 'test']:
        s = DatasetJsonlSerializer()
        mobius_dataset = s.load(dataset_path+'/{}.jsonl.gz'.format(split))
        dataset = []
        for i, doc in enumerate(mobius_dataset):
            if SHORT and i > 0:
                break
            if verbose:
                print('Processing Doc:{}'.format(i))
            for span1, span2, rel_type in span_pair_generator(doc):
                example = process_span_pair(span1, span2, doc, rel_type)
                if example is not None:
                    update_vocab(vocab, example)
                    dataset.append(example)
        datasets[split] = dataset
    return datasets, vocab


def span_pair_generator(doc, grammar=MIMIC_GRAMMAR):
    num_spans = len(doc.annotations.span_annotations)
    for span1_index in range(num_spans-1):
        for span2_index in range(span1_index, num_spans):
            span1 = doc.annotations.span_annotations[span1_index]
            span2 = doc.annotations.span_annotations[span2_index]
            rel_type = None
            if (span1.label.value, span2.label.value) in grammar.keys():
                rel_type = grammar[(span1.label.value, span2.label.value)]
            elif (span2.label.value, span1.label.value) in grammar.keys():
                rel_type = grammar[(span2.label.value, span1.label.value)]
            if rel_type is not None:
                yield (span1, span2, rel_type)

def update_vocab(vocab, example):
    vocab['text'].update(example['text'].split())
    vocab['chars'].update(list(example['text']))
    vocab['pos1_rel'].update(example['pos1_rel'])
    vocab['pos2_rel'].update(example['pos2_rel'])
    for key in example:
        if key not in ['text', 'pos1_rel', 'pos2_rel']:
            vocab[key].update([example[key]])

def clean(word):
    return word.replace('\s', '')

def process_span_pair(span1, span2, doc, rel_type):
    sentences = get_sentences(span1, span2, doc)
    if sentences is not None:
        words = [word for sent in sentences for word in sent.tokens]
        pos1, pos2 = find_pos(words, span1, span2)
        word_strs = [clean(x.text) for x in words]
        text = ' '.join([s for s in word_strs if s])
        words = text.split(' ')

        assert pos1 != -1
        assert pos2 != -1
        num_words = len(words)
        pos1_rel = relative_positions(pos1, num_words)
        pos2_rel = relative_positions(pos2, num_words)

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
        assert len(text.split(' ')) == len(pos2_rel), (len(text.split(' ')), num_words)
        example = {'text': text, 'pos1':pos1, 'pos2':pos2, 'pos1_rel':pos1_rel,
                   'pos2_rel': pos2_rel, 'relation': (label!= 'NA'),
                   'rel_type': rel_type}
        return example
    else:
        return None

def get_sentences(span1, span2, doc):
    start_char_idx = min(span1.start, span2.start)
    end_char_idx = max(span1.end, span2.end)
    sents = []
    # Get sents in either span or inbetween
    for sent in doc.structures.sentences:
        if sent.end >= start_char_idx and sent.start < end_char_idx:
            sents.append(sent)
    if len(sents) < MAX_SENTS:
        sents.sort(key=lambda s: s.start)
        return sents
    else:
        return None


def example_generator(path, fields):
    f = open(path + '/examples', 'r')
    num_ex = int(f.readline())
    cur_ex = 0
    while cur_ex < num_ex:
        line = f.readline()
        yield Example.fromJSON(line, fields)
        cur_ex += 1

def relative_positions(pos, num_words):
    pos_rel = list(range(num_words))
    pos_rel = [(j - pos) for j in pos_rel]
    return pos_rel

def proprocess(x):
    b = x!='NA'
    return str(bool(b))

def load_dataset(path, binary=True, vocab_path=None):
    print('Loading data from path {}...'.format(path))
    vocab_count = pickle.load(open(path + '/vocab', 'rb'))
    print('Constructing Fields...')
    fields_dict = make_fields(vocab_count)
    fields = convert_fields(fields_dict)
    print('Loading Examples...')
    train_examples = example_generator(path+'/train', fields_dict)
    train_data = Dataset(train_examples, fields)
    test_examples = example_generator(path+'/test', fields_dict)
    test_data = Dataset(test_examples, fields)
    return train_data, test_data

def convert_fields(inp_fields):
    fields = {}
    for v in inp_fields.values():
        if isinstance(v, list):
            for t in v:
                fields[t[0]] = t[1]
        else:
            fields[v[0]] = v[1]
    return fields

def make_fields(vocab_count, binary=True):
    text_field = Field(batch_first=True, include_lengths=True, tokenize = lambda x: x.split(' '))
    text_field.vocab = Vocab(vocab_count['text'])
    char_nesting_field = Field(batch_first=True, tokenize = list)
    char_field = NestedField(char_nesting_field, tokenize = lambda x: x.split(' '))
    char_nesting_field.vocab = Vocab(vocab_count['chars'])
    char_field.vocab = Vocab(vocab_count['chars'])
    pos1_field = Field(batch_first=True, sequential=False, use_vocab=False)
    pos2_field = Field(batch_first=True, sequential=False, use_vocab=False)
    pos1_rel_field = Field(sequential=True, batch_first=True)
    pos1_rel_field.vocab = Vocab(vocab_count['pos1_rel'])
    pos2_rel_field = Field(sequential=True, batch_first=True)
    pos2_rel_field.vocab = Vocab(vocab_count['pos2_rel'])
    if binary:
        label_field = Field(sequential=False,
                            batch_first=True)
    else:
        label_field = Field(sequential=False, batch_first=True)
    label_field.vocab = Vocab(vocab_count['relation'], specials=[])
    reltype_field = Field(batch_first=True, sequential=False)
    reltype_field.vocab = Vocab(vocab_count['rel_type'])
    fields_dict = {'text':[('text', text_field), ('chars', char_field)],
            'pos1':('pos1', pos1_field),
            'pos2':('pos2', pos2_field),
            'pos1_rel':('pos1_rel', pos1_rel_field),
            'pos2_rel':('pos2_rel', pos2_rel_field),
            'relation':('relation', label_field),
            'rel_type':('rel_type', reltype_field)}
    return fields_dict


def write_dataset(dataset, path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    f = open(path + '/examples', 'w')
    f.write(str(len(dataset)) + '\n')
    for ex in dataset:
        f.write(json.dumps(ex) + '\n')
    f.close()


if __name__ == '__main__':
    nre_dataset,  nre_vocab = get_mobius_dataset(MIMIC_DATASET, MIMIC_GRAMMAR)
    write_dataset(nre_dataset['train'], OUTPUT_PATH+'/train')
    write_dataset(nre_dataset['test'], OUTPUT_PATH+'/test')
    pickle.dump(nre_vocab, open(OUTPUT_PATH + '/vocab', 'wb'))
    train_data, test_data = load_dataset(OUTPUT_PATH)
