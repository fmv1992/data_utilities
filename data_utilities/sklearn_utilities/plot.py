import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from data_utilities.sklearn_utilities import get_estimator_name
from data_utilities.sklearn_utilities.metrics import (
    _get_probability_tresholds,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
    true_positive_rate)

# pylama: ignore=D103

_TIGHT_LAYOUT_RECT = [0, 0.03, 1, 0.95]


def plot_precision_and_recall_curve(iterable_of_models, x, y, outputfile):
    fig, ax = plt.subplots()

    for model in iterable_of_models:
        probas_1 = model.predict_proba(x)[:, 1]
        precision, recall, _ = precision_recall_curve(y, probas_1)
        ax.plot(recall,
                precision,
                label=get_estimator_name(model))

    ax.legend()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])

    fig.suptitle('Precision and recall curve')

    fig.tight_layout(rect=_TIGHT_LAYOUT_RECT)
    fig.savefig(outputfile, dpi=500)


def plot_confusion_matrix_rates(estimator,
                                x,
                                y,
                                outputfile,
                                metrics=('fnr', 'fpr', 'tnr', 'tpr')):

    estimator_name = get_estimator_name(estimator)

    probas_1 = estimator.predict_proba(x)[:, 1]

    unique_scores = _get_probability_tresholds(probas_1)

    acronym_to_func = dict()
    acronym_to_func['tpr'] = true_positive_rate
    acronym_to_func['fpr'] = false_positive_rate
    acronym_to_func['tnr'] = true_negative_rate
    acronym_to_func['fnr'] = false_negative_rate

    acronym_to_metric = dict()
    for metric in metrics:
        acronym_to_metric[metric] = acronym_to_func[metric](y, probas_1)

    acronym_to_legend = dict()
    acronym_to_legend['tpr'] = 'True positive rate'
    acronym_to_legend['fpr'] = 'False positive rate'
    acronym_to_legend['tnr'] = 'True negative rate'
    acronym_to_legend['fnr'] = 'False negative rate'

    fig, ax = plt.subplots()

    for metric in metrics:
        ax.plot(unique_scores,
                acronym_to_metric[metric],
                label=acronym_to_legend[metric])

    ax.legend()
    ax.set_xlabel('Probability Threshold (Cutpoint)')
    ax.set_ylabel('Metric value')
    ax.set_xlim([0.0, 1.0])

    fig.suptitle('Confusion Matrix Metrics for {0}'.format(estimator_name))

    fig.tight_layout(rect=_TIGHT_LAYOUT_RECT)
    fig.savefig(outputfile, dpi=500)
