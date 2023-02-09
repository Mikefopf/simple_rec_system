#!/usr/bin/env python3
"""
Class for all pipeline of model learning.
"""

from typing import List, Tuple, Dict
import logging

import pandas as pd
import numpy as np
import implicit

import utils

logging.basicConfig(filename='req_sys.log', filemode='w', level=logging.DEBUG)

class PipelineReqSystem:
    """
    Full pipeline simple recomendation system
    Allows you to run the following stages of the model in production:
    data preparation, obtaining baseline metrics, selection of hyperparameters,
    training the best checkpoint, comparison with baseline, saving, inference

    Attributes:
        n_preds:
            Number of items to predict for specific user.
        learn_data:
            All transaction those use for learning.
            Dataframe consists of two columns: row - user id,
            col - item id.
        train:
            Aggregated by user data for train.
        val:
            Aggregated by user data for validation.
        test:
            Aggregated by user data for test.
        score_baseline:
            Score of baseline algorithm that is prediction
            top items for every user.
        hyperparams_als:
            Best hyperparameters for als algorithm.
        hyperparams_cosine:
            Best hyperparameters for cosine nearest neighbours algorithm.
        model_als:
            Best checkpoint for als algorithm.
        model_cosine:
            Best checkpoint for cosine nearest neighbours algorithm.
        max_score:
            Score of best model.
    """

    def __init__(self, learn_data: pd.DataFrame, n_preds=30):
        self.n_preds = n_preds
        self.learn_data = learn_data
        self.train, self.val, self.test = self.prepare_data()
        self.score_baseline = self.get_score_baseline()
        self.hyperparams_als, self.hyperparams_cosine = self.get_hyperparams()
        self.model_als, self.model_cosine = self.get_checkpoints()
        self.max_score = self.compare_models()

    def prepare_data(self):
        """
        Data preparation.

        Train, validation, test split (0.6 : 0,2 : 0.2) and aggregate by user.

        Return:
            Aggregated train, val, test data.
        """
        logging.debug('Prepare data')
        if self.learn_data.shape[0] == 0:
            raise ValueError('Data is empty')
        elif self.learn_data.shape[0] < 1000:
            raise ValueError('Data is too small')        
        random_df = self.learn_data.sample(frac=1, random_state=42)
        train, validate, test = (random_df[0:int(0.6 * self.learn_data.shape[0])],
                                 random_df[int(0.6 * self.learn_data.shape[0]):
                                           int(0.8 * self.learn_data.shape[0])],
                                 random_df[int(0.8 * self.learn_data.shape[0]):])
        return (utils.aggregate_users(df.values) for df in [train, validate, test])

    def get_score_baseline(self) -> float:
        """
        Calculate baseline score.

        Baseline is algorithm that recommend top items for every user.

        Return:
            Score for baseline.
        """
        logging.debug('Calculate baseline score')
        tops = self.learn_data.groupby('col').count().sort_values('data', ascending=False)
        top_items = list(np.array(tops.index)[:30])
        return utils.get_score_top(self.test, top_items)

    def get_hyperparams(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate best hyperparameters.

        Simple grid search for als and cosine. Parameters for als:
        factors - number of hiden parameters, regularization -
        regularization coefficient, iterations - number of iterations in
        als. Parameters for cosine: K - number of neighbours to use.

        Return:
            Best params for als and cosine.
        """
        factors_grid = [32, 64]
        regularization_grid = [0, 0.2]
        iterations_grid = [8, 16]
        params_als = {'factors': factors_grid,
                      'regularization': regularization_grid,
                      'iterations': iterations_grid}
        params_cosine = {'K': [1, 3, 5, 10, 20, 50, 100, 200]}
        logging.debug('Grid search als')
        als_best_params = utils.grid_search_recommend(
            implicit.als.AlternatingLeastSquares, params_als, self.train, self.val)
        logging.debug('Grid search cosine')
        cosine_best_params = utils.grid_search_recommend(
            implicit.nearest_neighbours.CosineRecommender, params_cosine, self.train, self.val)
        return als_best_params, cosine_best_params

    def get_checkpoints(self) -> tuple:
        """
        Get checkpoints of als and cosine with best hyperparameters.

        Return:
            Best params for als and cosine.
        """
        logging.debug('Train best checkpoints')
        model_als = implicit.als.AlternatingLeastSquares(**self.hyperparams_als)
        model_cosine = implicit.nearest_neighbours.CosineRecommender(**self.hyperparams_cosine)
        x_sparse = utils.create_sparse_matrix(utils.USERS_NUM + 1, utils.ITEMS_NUM + 1, self.train)
        model_als.fit(x_sparse)
        model_cosine.fit(x_sparse)
        return model_als, model_cosine

    def compare_models(self) -> float:
        """
        Compare models with baseline.

        If baseline better than every model raise error.
        If there is a model better than baseline return
        best score.

        Return:
            Best score.
        """
        logging.debug('Compare models with baseline')
        score_als = utils.get_score_model(self.test, self.train, self.model_als, utils.ITEMS_NUM + 1)
        score_cosine = utils.get_score_model(self.test, self.train, self.model_cosine, utils.ITEMS_NUM + 1)
        if self.get_score_baseline() > max(score_als, score_cosine):
            raise RuntimeError('ML models are worse than baseline')
        return max(score_als, score_cosine)

    def inference(self, inference_data: List[int]) -> List[List[int]]:
        """
        Method to infer model.

        Create sparse matrix. Than for every user get recommendations.

        Return:
            List of recommendations.
        """
        res = []
        for key in inference_data:
            row_sparse = utils.make_coo_row(self.train.get(key, []), utils.ITEMS_NUM + 1).tocsr()
            recommended_items = self.model_cosine.recommend(
                int(key - 1), row_sparse, N=30, filter_already_liked_items=True,
                recalculate_user=False,
            )
            res.append(recommended_items)
        return res

def main():
    interactions_df = pd.read_csv('../interactions.csv')
    print(max(interactions_df['row']))
    #pipeline = PipelineReqSystem(interactions_df)
    #print(pipeline.score_baseline, pipeline.max_score)

if __name__ == '__main__':
    main()