from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict

data = load_dataset('lhallee/Stringv12ModelOrgPairs90', split='train')

negatome = load_dataset('Synthyra/Negatome')

negatome_seqs = set()

for name, split in negatome.items():
    negatome_seqs.update(split['SeqA'])
    negatome_seqs.update(split['SeqB'])

print(len(negatome_seqs))


def filter_out_negatome(ex):
    if ex['SeqA'] in negatome_seqs or ex['SeqB'] in negatome_seqs:
        return False
    return True

print(len(data))
data = data.filter(filter_out_negatome)
print(len(data))

data = data.train_test_split(test_size=0.5, seed=42)

train_split = data['train']
test_split = data['test']

train_seqs = set()
for example in train_split:
    train_seqs.add(example['SeqA'])
    train_seqs.add(example['SeqB'])

def no_overlap_with_train(example):
    return example['SeqA'] not in train_seqs and example['SeqB'] not in train_seqs

def perfect_overlap_with_train(example):
    return example['SeqA'] in train_seqs and example['SeqB'] in train_seqs

# Keep track of discarded examples to add back to train
discarded_from_test_for_train = []
discarded_from_test_for_valid = []

def collect_discarded(example):
    if not no_overlap_with_train(example):
        if perfect_overlap_with_train(example):
            discarded_from_test_for_train.append(example)
            return False
        else:
            discarded_from_test_for_valid.append(example)
            return False
    return True

test_split = test_split.filter(collect_discarded)

discarded_dataset = Dataset.from_list(discarded_from_test_for_train)
train_split = concatenate_datasets([train_split, discarded_dataset])
valid_split = Dataset.from_list(discarded_from_test_for_valid)

print(train_split)
print(valid_split)
print(test_split)


final_data = DatasetDict({
    'train': train_split,
    'valid': valid_split,
    'test': test_split
})
final_data.push_to_hub('lhallee/string_model_org_90_90_split', private=True)
