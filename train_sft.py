import os
import tqdm
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings("ignore")

from source.config import parse_args_llm
from source.utils.help_funcs import seed_everything
from source.utils.help_funcs import collate_fn
from source.utils.help_funcs import _save_checkpoint, _reload_model
from source.datasets import *
from source.models import *
from source.utils.evaluation import *

import wandb

tasks = [
    "Caco-2",
    "Lipophilicity",
    "Solubility",
    "PPBR",
    "VDss",
    "Half life",
    "Clearance microsome",
    "Clearance hepatocyte",
    "LD50",
]

def main(args):
    seed = args.seed
    seed_everything(seed=seed)
    wandb.init(project=f"{args.project}",
        name=f"{args.run_name}",
        config=args,
    )

    # Step 1: Build dataset
    train_datasets = []
    for task in tasks:
        train_datasets.append(load_dataset[args.dataset](path = args.path, task = task, split = args.split, use_rule = args.use_rule, rule_index = args.rule_index))
    train_loader = DataLoader(ConcatDataset(train_datasets), shuffle=True, batch_size=args.batch_size, pin_memory=True, collate_fn=collate_fn)

    # Step 2: Build model and optimizer
    args.llm_path = get_llm_path[args.llm_name]
    model = load_model[args.model_name](args=args)
    if args.checkpoint_path is not None:
        model = _reload_model(model, args.checkpoint_path)

    trainable_params, all_param = model.print_trainable_params()
    print("-"*len(f"No. Trainable Params: {trainable_params} ({100 * trainable_params / all_param:.4f} %)"))
    print(f"No. Trainable Params: {trainable_params} ({100 * trainable_params / all_param:.4f} %)")
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.weight_decay}],
        betas=(0.9, 0.999)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
        lambda step: (1e-6 / args.lr) + (1 - step / (args.num_epochs * len(train_loader))) * (1 - 1e-6 / args.lr)
    )

    # Step 3: Training
    progress_bar = tqdm.tqdm(range(args.num_epochs * len(train_loader)))
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.

        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()

            wandb.log({
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/loss': loss.item()
            })
            progress_bar.update(1)

        print(f"Epoch {epoch}|{args.num_epochs}: Train Loss: {epoch_loss / len(train_loader):.4f}")
        if epoch == args.num_epochs:
            _save_checkpoint(model, epoch, args, is_best=False)

if __name__ == "__main__":
    args = parse_args_llm().parse_args()
    main(args)