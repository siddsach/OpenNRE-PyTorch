from process_mobius import span_pair_generator, process_span_pair, make_fields, \
        MIMIC_DATASET, OUTPUT_PATH
from config.Config import get_mask
from torchtext.data import Example
from mobiuscore.doc.annotations.relation import Relation
from mobiuscore.doc.annotations.label import Label
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
                print(name)
                if type(ex_tensors[name]) is not tuple:
                    print(ex_tensors[name].shape)
                else:
                    print(ex_tensors[name][0].shape)
        word, length = ex_tensors['text']
        mask = get_mask(word, ex_tensors['pos1'], ex_tensors['pos2'], length)
        logits = model(word=word,
                    chars=ex_tensors['chars'],
                    pos1=ex_tensors['pos1_rel'],
                    pos2=ex_tensors['pos2_rel'],
                    mask=mask
                )
        _, output = torch.max(logits, dim = 1)
        output = [int(x.item()) for x in output]
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
