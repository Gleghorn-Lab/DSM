from torch.utils.data import DataLoader
from transformers import Trainer
from data.dataset_classes import IterableDatasetFromHF


def get_iterable_trainer(
        model,
        hf_dataset,
        data_collator,
        training_args,
        batch_size,
        col_name="sequence",
        num_workers=4,
        prefetch_factor=10,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        eval_dataset=None,
    ):
    train_dataset = IterableDatasetFromHF(hf_dataset, col_name=col_name)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=data_collator,
    )

    class IterableTrainer(Trainer):
        def get_train_dataloader(self):
            # Return the custom DataLoader instead of relying on a map-style dataset.
            return train_dataloader

    return IterableTrainer(
        model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        eval_dataset=eval_dataset,
        optimizers=optimizers,
    )
