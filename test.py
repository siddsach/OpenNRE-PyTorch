from process_mobius import span_pair_generator, process_span_pair, make_fields, \
        MIMIC_DATASET, OUTPUT_PATH
from config.Config import get_mask
from torchtext.data import Example
from mobiuscore.doc.annotations.relation import Relation
from mobiuscore.dataset.serialize.json import DatasetJsonlSerializer
import models
import torch
import json
import os
import pickle

def annotate(model, doc, inp_fields):
    examples = []
    for span1, span2, rel_type in span_pair_generator(doc):
        example = process_span_pair(span1, span2, doc, rel_type)
        torch_ex = Example.fromdict(example, inp_fields)
        ex_tensors = {}
        for name in inp_fields:
            tgt_fields = inp_fields[name]
            if not isinstance(tgt_fields, list):
                tgt_fields = [tgt_fields]

            data = [getattr(torch_ex, name)]
            for name, field in tgt_fields:
                ex_tensors[name] = field.process(data)
        word, length = ex_tensors['text']
        value = model(word=word,
                    chars=ex_tensors['chars'],
                    pos1=ex_tensors['pos1'],
                    pos2=ex_tensors['pos2'],
                )
        relation = Relation(source='nre',
                            label=Label(value=value),
                            annotation_from=span1,
                            annotation_to=span2,
                            doc=doc)
        doc.add(relation)

def load_model_checkpoint(path='checkpoint/PCNN_FF/models',
                          hyperparams=None,
                          model_type=models.PCNN_FF):
    params = json.load(open(os.path.join(path, 'params.json')))
    model = model_type(params)
    state_dict = torch.load(os.path.join(path, 'Epoch-0'))
    model.load_state_dict(state_dict)
    print('MODEL')
    print(model)
    return model

def evaluate(dataset_path, vocab_path, model=None):
    if model is None:
        model = load_model_checkpoint()
    s = DatasetJsonlSerializer()
    test_dataset = s.load(dataset_path+'/{}.jsonl.gz'.format('test'))
    vocab = pickle.load(open(vocab_path, 'rb'))
    inp_fields = make_fields(vocab)
    for i, doc in enumerate(test_dataset):
        annotate(model, doc, inp_fields)
    #s.dump(test_dataset, '')

if __name__ == '__main__':
    evaluate(MIMIC_DATASET, OUTPUT_PATH+'/vocab', None)
