from itertools import islice
from datasets import load_dataset, Dataset
from bitdelta.data import get_dataloader


def build_dataloaders(tokenizer, max_length: int):
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation'
    )

    en_limited = Dataset.from_list(list(islice(traindata, 1000)))
    eval_en_limited = Dataset.from_list(list(islice(valdata, 1000)))

    dataloader = get_dataloader(en_limited, tokenizer, batch_size=1, num_workers=0, max_length=max_length)
    val_dataloader = get_dataloader(eval_en_limited, tokenizer, batch_size=1, num_workers=0, max_length=max_length)
    return dataloader, val_dataloader
