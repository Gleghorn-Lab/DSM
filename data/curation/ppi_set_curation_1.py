from datasets import load_dataset
from collections import defaultdict
from functools import partial

"""
Make string dataset from model organsims clustered by 90% sequence similarity and filtered by 900 score
"""

model_orgs = load_dataset('Synthyra/Stringv12ModelOrgSeqs')
model_pairs = load_dataset('Synthyra/Stringv12ModelOrgPairs', split='train')
dicts = load_dataset('lhallee/ModelOrgsStringDB')

id_org_dict = dict(zip(model_orgs['90']['id'], model_orgs['90']['org']))
id_seq_dict = dict(zip(model_orgs['90']['id'], model_orgs['90']['seq']))

org_num_dict = {
    '0': 'mus musculus',
    '1': 'pseudomonas aeruginosa',
    '2': 'arabidopsis thaliana',
    '3': 'saccharomyces cerevisiae',
    '4': 'escherichia coli',
    '5': 'caenorhabditis elegans',
    '6': 'drosophila melanogaster',
    '7': 'danio rerio',
    '8': 'human',
}


# Create a defaultdict to store sequences for each organism
org_seq_dict = defaultdict(set)

# Populate the dictionary
for name, split in dicts.items():
    if 'seqs' in name:
        num = name.split('_')[0]
        org = org_num_dict[num]
        org_seq_dict[org].update(split['value'])

# Create an ordered dictionary based on the specified order
ordered_org_seq_dict = {}
org_order = ['human', 'mus musculus', 'escherichia coli', 'saccharomyces cerevisiae', 
             'caenorhabditis elegans', 'danio rerio', 'pseudomonas aeruginosa', 
             'arabidopsis thaliana', 'drosophila melanogaster']

for org in org_order:
    if org in org_seq_dict:
        ordered_org_seq_dict[org] = org_seq_dict[org]

# Replace the original dictionary with the ordered one
org_seq_dict = ordered_org_seq_dict

# Print the number of sequences for each organism
for key, value in org_seq_dict.items():
    print(key, len(value))


def get_org_pairs(ex, id_set):
    score = ex['scores']
    if score > 900:
        id_a, id_b = ex['pairs'].split('|')
        if id_a in id_set and id_b in id_set:
            return True
        else:
            return False
    else:
        return False

def map_org_pairs(ex, id_org_dict, id_seq_dict):
    id_a, id_b = ex['pairs'].split('|')
    ex['OrgA'] = id_org_dict[id_a]
    ex['OrgB'] = id_org_dict[id_b]
    ex['SeqA'] = id_seq_dict[id_a]
    ex['SeqB'] = id_seq_dict[id_b]
    return ex


id_set = set(model_orgs['90']['id'])
print(len(id_set))


get_org_pairs_fn = partial(get_org_pairs, id_set=id_set)

print(len(model_pairs))
model_pairs = model_pairs.filter(get_org_pairs_fn, num_proc=72)
print(len(model_pairs))

map_org_pairs_fn = partial(map_org_pairs, id_org_dict=id_org_dict, id_seq_dict=id_seq_dict)
model_pairs = model_pairs.map(map_org_pairs_fn, num_proc=72)

# rename scores to score
# rename pairs to IDs
model_pairs = model_pairs.rename_column('scores', 'score')
model_pairs = model_pairs.rename_column('pairs', 'IDs')
model_pairs.push_to_hub('lhallee/Stringv12ModelOrgPairs90', private=True)
