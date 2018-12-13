####### Mobiuscore Data Processing ######
from mobiuscore.dataset.serialize.json import DatasetJsonlSerializer
from mobiuscore.doc.structures.sentence import Sentence
import spacy
import json
import torch
from torchtext.data import Dataset, Field, NestedField, Example, BucketIterator
from torchtext.vocab import Vocab
from collections import Counter
import pickle
import numpy as np
import os

SHORT = True
#MIMIC_DATASET = 'n2c2/train/tokenized_spacy'
MIMIC_DATASET = '/Users/sidsachdeva/roam/data/mimic'
OUTPUT_PATH = MIMIC_DATASET
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

'''
def find_pos(sentence, head, tail):
    # print("SENTENCE")
    # print(sentence)
    # print("HEAD")
    # print(head)
    # print('TAIL')
    # print(tail)
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
'''


def get_mobius_dataset(dataset_path, grammar, verbose=True):
    datasets = {}
    vocab = {f: Counter() for f in ['text', 'chars', 'pos1', 'pos2', 'pos1_rel', 'pos2_rel', 'relation']}
    for split in ['train', 'test']:
        s = DatasetJsonlSerializer()
        mobius_dataset = s.load(dataset_path+'/{}.jsonl.gz'.format(split))
        dataset = []
        n_t = 0
        n_f = 0
        for i, doc in enumerate(mobius_dataset):
            if SHORT:
                if i > 0:
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
                        if example is not None:
                            if example['relation'] or np.random.uniform() > 0.97:

                                update_vocab(vocab, example)
                                dataset.append(example)

        datasets[split] = dataset
    return datasets, vocab

def update_vocab(vocab, example):
    vocab['text'].update(example['text'].split())
    vocab['chars'].update(list(example['text']))
    vocab['pos1'].update(example['pos1_rel'])
    vocab['pos2'].update(example['pos2_rel'])
    for key in example:
        if key not in ['text', 'pos1_rel', 'pos2_rel']:
            vocab[key].update([example[key]])

def clean(word):
    return word.replace('\s', '')

def process_span_pair(span1, span2, doc):
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
                   'pos2_rel': pos2_rel, 'relation': (label!= 'NA')}
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
    print('PREPROCESS')
    print(b)
    return str(bool(b))

def load_dataset(path, binary=True, vocab_path=None):
    vocab_count = pickle.load(open(path + '/vocab', 'rb'))
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
    fields_dict = {'text':[('text', text_field), ('chars', char_field)],
            'pos1':('pos1', pos1_field),
            'pos2':('pos2', pos2_field),
            'pos1_rel':('pos1_rel', pos1_rel_field),
            'pos2_rel':('pos2_rel', pos2_field),
            'relation':('relation', label_field)}
    fields = {'text': text_field,
            'chars': char_field,
            'pos1': pos1_field,
            'pos2': pos2_field,
            'pos1_rel':pos1_rel_field,
            'pos2_rel':pos2_rel_field,
            'relation':label_field}
    print('Loading data from path {}...'.format(path))
    train_examples = example_generator(path+'/train', fields_dict)
    train_data = Dataset(train_examples, fields)
    test_examples = example_generator(path+'/test', fields_dict)
    test_data = Dataset(test_examples, fields)
    return train_data, test_data


def write_dataset(dataset, vocab, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    f = open(path + '/examples', 'w')
    f.write(str(len(dataset)) + '\n')
    for ex in dataset:
        f.write(json.dumps(ex) + '\n')
    f.close()


if __name__ == '__main__':
    nre_dataset,  nre_vocab = get_mobius_dataset(MIMIC_DATASET, MIMIC_GRAMMAR)
    write_dataset(nre_dataset['train'], nre_vocab, OUTPUT_PATH+'/train')
    write_dataset(nre_dataset['test'], nre_vocab, OUTPUT_PATH+'/test')
    pickle.dump(nre_vocab, open(OUTPUT_PATH + '/vocab', 'wb'))
    train_data, test_data = load_dataset(OUTPUT_PATH)
