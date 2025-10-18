import json
import random
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

class GenerationDataset(Dataset):
    def __init__(self, path, task, split="train", use_rule=False, rule_index=None):
        super().__init__()

        self.questions = load_dataset(f"lhkhiem28/{path.split('/')[-1]}", task, split=f"{split}")
        self.use_rule = use_rule
        self.rule_index = rule_index
        if self.use_rule:
            with open(f'{path}/TDC/DrugADMET_rules/{task}.json', 'r', encoding="utf8") as f:
                self.rules = json.load(f)

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        item = self.questions[index]

        if self.use_rule:
            if self.rule_index is None:
                rule = random.choice(self.rules)
            else:
                rule = self.rules[self.rule_index]
            return {"id": index,
                "prompt": item['messages_fgs'][0]['content'] + f'\nFor your reference, you will be provided with a predictive rule in the form of a tree structure, containing numerous if-then conditions. Hierarchy within the tree is indicated using the number of tab characters (\t). \nThe predictive rule is:\n{rule}',
                "label": item['messages_fgs'][1]['content'],
            }
        else:
            return {"id": index,
                "prompt": item['messages'][0]['content'],
                "label": item['messages'][1]['content'],
            }