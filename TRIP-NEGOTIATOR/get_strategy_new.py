from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def get_prediction(curr_sentence,idx,role_id):
    model_args = ClassificationArgs()

    # Create a ClassificationModel
    model = ClassificationModel(
        'roberta',
        '/DATA/komal_2021cs16/Priyanshu/neg-tour/nsc-model/best_model/',
        num_labels=8,
        args=model_args, use_cuda=False
    ) 
    predictions, raw_outputs = model.predict([curr_sentence])
    

    predicted_probabilities = raw_outputs

    
    return predicted_probabilities
