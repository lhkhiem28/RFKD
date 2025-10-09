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

    for task in tasks:
        # Step 1: Build dataset
        test_dataset = load_dataset[args.dataset](path = args.path, task = task, split = "test", use_rule = args.use_rule, rule_index = args.rule_index)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=collate_fn)

        # Step 2: Build model
        args.llm_path = get_llm_path[args.llm_name]
        model = load_model[args.model_name](args=args)
        if args.checkpoint_path is not None:
            model = _reload_model(model, args.checkpoint_path)

        # Step 3: Evaluating
        model.eval()
        eval_outputs = []
        for _, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                eval_outputs.append(output)

        # Step 4: Post-processing & report
        scores = eval_funcs[args.dataset](eval_outputs)
        print("MAE: {:.3f} | 1-Spearman: {:.3f} (Validity: {:.2f}%)".format(
            *scores
        ))

if __name__ == "__main__":
    args = parse_args_llm().parse_args()
    main(args)