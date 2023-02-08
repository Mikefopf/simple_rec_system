import pandas as pd
import numpy as np
import implicit

import utils
import metrics

class PipelineReqSystem:
    """
    Full pipeline simple recomendation system
    
    Allows you to run the following stages of the model in production: data preparation, obtaining baseline metrics, selection of hyperparameters, training the best checkpoint, comparison with baseline, saving, inference
    
    Attributes:
        
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
        train, validate, test = \
              np.split(self.learn_data.sample(frac=1, random_state=42), 
                       [int(.6*len(self.learn_data)), int(.8*len(self.learn_data))])
        return (utils.aggregate_users(df.values) for df in [train, validate, test])
    
    def get_score_baseline(self):
        tops = self.learn_data.groupby('col').count().sort_values('data', ascending=False)
        top_items = list(np.array(tops.index)[:30])
        return utils.get_score_top(self.test, top_items)
    
    def get_hyperparams(self):
        factors_grid = [16, 32, 64]
        regularization_grid = [0, 0.2]
        iterations_grid = [4, 8, 16]
        params_als = {'factors': factors_grid, 
                      'regularization': regularization_grid, 
                      'iterations': iterations_grid}
        params_cosine = {'K': [1, 3, 5, 10, 20, 50, 100, 200]}
        als_best_params = utils.grid_search_recommend(implicit.als.AlternatingLeastSquares, params_als, self.train, self.val)
        cosine_best_params = utils.grid_search_recommend(implicit.nearest_neighbours.CosineRecommender, params_cosine, self.train, self.val)
        return als_best_params, cosine_best_params
    
    def get_checkpoints(self):
        model_als = implicit.als.AlternatingLeastSquares(**self.hyperparams_als)
        model_cosine = implicit.nearest_neighbours.CosineRecommender(**self.hyperparams_cosine)
        X_sparse = utils.create_sparse_matrix(30910, 18494, self.train)
        model_als.fit(X_sparse)
        model_cosine.fit(X_sparse)
        return model_als, model_cosine
    
    def compare_models(self):
        score_als = utils.get_score_model(self.test, self.train, self.model_als, 18494)
        score_cosine = utils.get_score_model(self.test, self.train, self.model_cosine, 18494)
        if self.get_score_baseline() > max(score_als, score_cosine):
            raise RuntimeError('ML models are worse than baseline')
        return max(score_als, score_cosine)
    
    def inference(self, inference_data):
        res = []
        for key in inference_data:
            row_sparse = make_coo_row(self.train.get(key, []), n_items).tocsr()
            recommended_items = model.recommend(
                int(key - 1), row_sparse, N=30, filter_already_liked_items=True, recalculate_user=False
            )
            res.append(recommended_items)
        return res
        
interactions_df = pd.read_csv('../interactions.csv')
pipeline = PipelineReqSystem(interactions_df)
print(pipeline.score_baseline, pipeline.max_score)