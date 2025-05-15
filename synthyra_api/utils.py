import pickle
from collections import defaultdict
from .annotation_mapping import (
    name_ec,
    ec_name,
    name_go,
    go_name,
    name_ip,
    ip_name,
    name_gene,
    gene_name
)


def describe_prompt(prompt, id2label, annotation_vocab_dict):
    descriptions = defaultdict(list)
    track_ids = {}
    for token in prompt:
        label = id2label[token]
        aspect = label.split('_')[-1].lower()
        if aspect in ['mf', 'bp', 'cc']:
            name_dict = annotation_vocab_dict['go'][1]
        elif aspect in ['ec', 'ip', '3d']:
            name_dict = annotation_vocab_dict[aspect][1]
        elif aspect == 'keywords':
            name_dict = None
        id = str(label.split('_')[0])
        try:
            if name_dict is not None:
                id = name_dict[id] + ' (' + id + ')'
                descriptions[aspect].append(id)
                track_ids[id] = token
            else:
                descriptions[aspect].append(id)
                track_ids[id] = token
        except:
            descriptions[aspect].append(id)
            track_ids[id] = token
    return descriptions, track_ids


with open('synthyra_api/id2label.pkl', 'rb') as f:
    id2label = pickle.load(f)


with open('synthyra_api/label2id.pkl', 'rb') as f:
    label2id = pickle.load(f)


annotation_vocab_dict = {
    'ec': (name_ec, ec_name),
    'go': (name_go, go_name),
    'ip': (name_ip, ip_name),
    '3d': (name_gene, gene_name)
}


aspect_dict = {
    'ec': 'Enzyme Comission Number',
    'bp': 'GO Biological Process',
    'cc': 'GO Cellular Component',
    'mf': 'GO Molecular Function',
    'ip': 'InterPro',
    'threed': 'Gene3D',
    'keywords': 'Uniprot Keywords',
    'cofactor': 'cofactor'
}
