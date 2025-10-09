import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

def get_scores_generation(eval_outputs):
    df = pd.concat([pd.DataFrame(output) for output in eval_outputs])

    labels, preds = [], []
    for label, pred in zip(df['label'].values.tolist(), df['pred'].values.tolist()):
        try:
            label, pred = float(label), float(pred)
            if abs(pred) < 1e4:
                labels.append(label), preds.append(pred)
        except:
            pass

    return [mean_absolute_error(labels, preds), 1-spearmanr(labels, preds)[0], 100*len(preds)/len(df)]

eval_funcs = {
    'generation': get_scores_generation,
}