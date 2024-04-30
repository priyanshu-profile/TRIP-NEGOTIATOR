from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def get_stage_prediction(curr_sentence):
    model_args = ClassificationArgs()

    # Create a ClassificationModel
    model = ClassificationModel(
        'roberta',
        '../neg-tour/outputs_stage_classifier/best_model/',
        num_labels=4,
        args=model_args, use_cuda=False
    ) 
    predictions, raw_outputs = model.predict(list(curr_sentence))
    return predictions[0]
