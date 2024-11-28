from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

class KNearestNeighborRF(BaseEstimator, RegressorMixin):
    """
    A random forest regressor that uses the values of and distances to the k-nearest neighbors of each point
    as features in the random forest.

    Takes a (geo)dataframe with a geometry column and a column of values to predict as input to fit().
    Takes a (geo)dataframe with a geometry column as input to predict().

    Parameters
    ----------
    n_neighbors : int, optional
        The number of neighbors to use as features. The default is 10.
    **params : dict
        Parameters to pass to the underlying RandomForestRegressor.
    """

    def __init__(self, n_neighbors=7, **params):
        self.n_neighbors = n_neighbors
        self.estimator = RandomForestRegressor(n_estimators = 766, max_features = 1.0, max_depth = 20,
                           min_samples_split = 2, min_samples_leaf = 4, bootstrap = True, random_state = 1)

        self.estimator.set_params(**params)

    def set_params(self, **params):
        if "n_neighbors" in params:
            self.n_neighbors = params.pop("n_neighbors")
        self.estimator.set_params(**params)

        return self

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, **self.estimator.get_params(deep)}

    def fit(self, X, y):
        self.geom_col_ = X.geometry.name
        self.fit_y_ = np.array(y)
        # Get the coordinates of the points as a numpy array
        X_locs = np.array(list(zip(X.geometry.x, X.geometry.y)))
        # Drop the geometry column and convert to numpy array
        _X = np.array(X.drop(columns=self.geom_col_))

        # Get the k nearest neighbors of each point
        self.neighbors_ = NearestNeighbors(
            n_neighbors=self.n_neighbors+1, algorithm="ball_tree"
        ).fit(X_locs)
        distances, indices = self.neighbors_.kneighbors(X_locs)
        # Get the values of and distances to the k nearest neighbors
        neighbor_ys = self.fit_y_[indices[:, 1:]]
        neighbor_dists = distances[:, 1:]
        # Concatenate the original features with the neighbor features
        _X = np.hstack((_X, neighbor_ys, neighbor_dists))
        # Fit the random forest
        self.estimator.fit(_X, self.fit_y_)
        return self

    def transform(self, X):
        # Get the coordinates of the points as a numpy array
        X_locs = np.array(list(zip(X.geometry.x, X.geometry.y)))
        _X = X.drop(columns=self.geom_col_)
        # Get the k nearest neighbors in the fit dataset of each point in the predict dataset
        distances, indices = self.neighbors_.kneighbors(X_locs)
        # Get the values of and distances to the k nearest neighbors
        neighbor_ys = self.fit_y_[indices[:, 1:]]
        neighbor_dists = distances[:, 1:]
        # Create column names for the neighbor features
        neighbor_ys_colnames = [f'neighbor_{i+1}_val' for i in range(0, neighbor_ys.shape[1])]
        neighbor_dists_colnames = [f'neighbor_{i+1}_dist' for i in range(0, neighbor_dists.shape[1])]
        # Concatenate the original features with the neighbor features
        _X = np.hstack((_X, neighbor_ys, neighbor_dists))
        # Convert to dataframe
        _X = pd.DataFrame(_X, columns=list(X.columns.drop('geometry')) + neighbor_ys_colnames + neighbor_dists_colnames)
        return _X


    def predict(self, X):
        return self.estimator.predict(self.transform(X))

    def score(self, X, y):
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

from sklearn.metrics import r2_score
class BufferDistanceRF(BaseEstimator, RegressorMixin):
    """
    A random forest regressor that uses the distances to every point in the fit data as features in the random forest.
    A column is added to passed data for each point in the fit data, containing the distance to that point.

    Takes a (geo)dataframe with a geometry column and a column of values to predict as input to fit().
    Takes a (geo)dataframe with a geometry column as input to predict().

    Parameters
    ----------
    **params : dict
        Parameters to pass to the underlying RandomForestRegressor.
    """
    def __init__(self, bins = 20, **params):
        self.estimator = RandomForestRegressor(n_estimators = 766, max_features = 1.0, max_depth = 20,
                           min_samples_split = 2, min_samples_leaf = 4, bootstrap = True, random_state = 1)
        self.bins = 20

        self.estimator.set_params(**params)

    def set_params(self, **params):
        if "bins" in params:
            self.bins = params.pop("bins")
        self.estimator.set_params(**params)

        return self

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def fit(self, X, y):
        self.geom_col_ = X.geometry.name
        self.fit_y_ = np.array(y)
        # Get the coordinates of the points as a numpy array
        X_locs = np.array(list(zip(X.geometry.x, X.geometry.y)))
        # Drop the geometry column and convert to numpy array
        _X = np.array(X.drop(columns=self.geom_col_))

        # Get the distances to every point in the fit dataset
        self.neighbors_ = NearestNeighbors(
            n_neighbors=len(X_locs), algorithm="ball_tree"
        ).fit(X_locs)
        distances, indices = self.neighbors_.kneighbors(X_locs)

        # if bins is None, just put the distance in the column
        if self.bins == None:
            buffer_matrix = np.array([d_vec[i_vec.argsort()] for d_vec, i_vec in zip(distances, indices)])
        # if bins is full, put n if the point is the nth nearest neighbor, essentially using N bins given dataset of size N
        elif self.bins == 'full':
            buffer_matrix = np.array([i_vec.argsort() for i_vec in indices])
        # if bins is an integer, group the points into that many distance bins, and assign the bin number
        # would be interesting to try with pre-assigned buffer distances as well. (eg 1 km, 5 km, 20 km, etc)
        else:
            buffer_matrix = np.array([np.digitize(d_vec[i_vec.argsort()], np.linspace(0, d_vec.max(), self.bins)) for d_vec, i_vec in zip(distances, indices)])
        # Concatenate the original features with the buffer distances matrix
        _X = np.hstack((_X, buffer_matrix))

        # Fit the random forest
        self.estimator.fit(_X, self.fit_y_)

        return self

    def transform(self, X):
        # Get the coordinates of the points as a numpy array
        X_locs = np.array(list(zip(X.geometry.x, X.geometry.y)))
        _X = X.drop(columns=self.geom_col_)
        # Get the distances to every point in the fit dataset
        distances, indices = self.neighbors_.kneighbors(X_locs)
        if self.bins == None:
            buffer_matrix = np.array([d_vec[i_vec.argsort()] for d_vec, i_vec in zip(distances, indices)])
        elif self.bins == 'full':
            buffer_matrix = np.array([i_vec.argsort() for i_vec in indices])
        else:
            buffer_matrix = np.array([np.digitize(d_vec[i_vec.argsort()], np.linspace(0, d_vec.max(), self.bins)) for d_vec, i_vec in zip(distances, indices)])
        # Concatenate the original features with the buffer distances matrix
        _X = np.hstack((_X, buffer_matrix))

        # add column names
        buffer_colnames = [f'buffer_point_{i+1}' for i in range(0, buffer_matrix.shape[1])]
        _X = pd.DataFrame(_X, columns=list(X.columns.drop('geometry')) + buffer_colnames)
        return _X

    def predict(self, X):
        # Get the coordinates of the points as a numpy array
        X_locs = np.array(list(zip(X.geometry.x, X.geometry.y)))
        _X = np.array(X.drop(columns=self.geom_col_))
        # Get the distances to every point in the fit dataset
        distances, indices = self.neighbors_.kneighbors(X_locs)
        if self.bins == None:
            buffer_matrix = np.array([d_vec[i_vec.argsort()] for d_vec, i_vec in zip(distances, indices)])
        elif self.bins == 'full':
            buffer_matrix = np.array([i_vec.argsort() for i_vec in indices])
        else:
            buffer_matrix = np.array([np.digitize(d_vec[i_vec.argsort()], np.linspace(0, d_vec.max(), self.bins)) for d_vec, i_vec in zip(distances, indices)])
         # Concatenate the original features with the buffer distances matrix
        _X = np.hstack((_X, buffer_matrix))

        return self.estimator.predict(_X)

    def score(self, X, y):
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

from sklearn.metrics import r2_score
class GeographicallyWeightedRF(BaseEstimator, RegressorMixin):
    """
    Geographically weighted random forest model.
    Takes a (geo)dataframe with a geometry column and a column of values to predict as input to fit().
    Takes a (geo)dataframe with a geometry column as input to predict().
    Builds a global, spatially-agnostic model on all fit data,
    and a local model for each point in the fit data, based on the k-nearest neighbors of that point.
    The final prediction is a weighted average of the global and local models.

    Parameters
    ----------
    n_neighbors : int, default=100
        Number of neighbors to use for local model.
    local_weight : float, default=0.25
        Weight of local model in final prediction.
    **params : dict
        Parameters to pass to the global model.
    """

    def __init__(self, n_neighbors=100, local_weight=0.25, **params):
        self.n_neighbors = n_neighbors
        self.local_weight = local_weight
        self.global_model = RandomForestRegressor(n_estimators = 766, max_features = 1.0, max_depth = 20,
                           min_samples_split = 2, min_samples_leaf = 4, bootstrap = True, random_state = 1)
        self.local_models = []

        self.set_params(**params)

    def set_params(self, **params):
        # only sets parameters for the global model
        # maybe just set the same params to the local models tbh
        if "n_neighbors" in params:
            self.n_neighbors = params.pop("n_neighbors")
        if "local_weight" in params:
            self.local_weight = params.pop("local_weight")
        self.global_model.set_params(**params)
        return self

    def get_params(self, deep=True):
        return {
            "n_neighbors": self.n_neighbors,
            "local_weight": self.local_weight,
            **self.global_model.get_params(deep),
        }

    def fit(self, X, y):
        self.geom_col_ = X.geometry.name
        # locations of each sample in X
        X_locs = np.array(list(zip(X.geometry.x, X.geometry.y)))

        # fit global model
        self.global_model.fit(X.drop(columns=[self.geom_col_]), y)

        # fit nearest neighbors model
        self.neighbors_ = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm="ball_tree"
        ).fit(X_locs)

        # get indices of nearest neighbors to each sample
        distances, indices = self.neighbors_.kneighbors(X_locs)

        # fit a local model to each sample using its n nearest neighbors
        for i in range(X.shape[0]):
            local_X = X.iloc[indices[i][1:], :].drop(columns=[self.geom_col_])
            local_y = y.iloc[indices[i][1:]]
            self.local_models.append(RandomForestRegressor(n_estimators = 766, max_features = 1.0, max_depth = 20,
                           min_samples_split = 2, min_samples_leaf = 4, bootstrap = True, random_state = 1).fit(local_X, local_y))

        return self

    def predict(self, X):
        # get locations of each test point then lose the geometry column
        X_locs = np.array(list(zip(X.geometry.x, X.geometry.y)))
        X = X.drop(columns=self.geom_col_)
        # get predictions from  global model
        global_pred = self.global_model.predict(X)
        # get nearest local model to each test point
        nearest_idxs = self.neighbors_.kneighbors(X_locs, n_neighbors=1)[1][:, 0]
        # get predictions from local models
        local_pred = X.apply(
            lambda row: self.local_models[
                nearest_idxs[X.index.get_loc(row.name)]
            ].predict(row.values.reshape(1, -1))[0],
            axis=1,
        )
        # return weighted average of local and global predictions
        return np.array(
            self.local_weight * local_pred + (1 - self.local_weight) * global_pred
        )

    def score(self, X, y):
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import r2_score

class OrdinaryKrigingRF(BaseEstimator, RegressorMixin):
    """
    Ordinary kriging with random forest residuals.
    Takes a (geo)dataframe with a geometry column and a column of values to predict as input to fit().
    Takes a (geo)dataframe with a geometry column as input to predict().
    Builds a spatially-agnostic model on all fit data, and uses it to predict on that fit data.
    Takes the residuals from that prediction and uses them to fit an ordinary kriging interpolation.

    The final prediction is the sum of the global model prediction and the value of the kriged surface
        at the site of the test point.

    Parameters
    ----------
    variogram_model : str, default="linear"
        Variogram model to use for kriging.
    verbose : bool, default=False
        Whether to print variogram model parameters.
    enable_plotting : bool, default=False
        Whether to plot variogram model.
    **params : dict
        Parameters to pass to the random forest model.
    """

    def __init__(self, **kwargs):
        self.estimator = RandomForestRegressor(n_estimators = 766, max_features = 1.0, max_depth = 20,
                           min_samples_split = 2, min_samples_leaf = 4, bootstrap = True, random_state = 1)

        self.variogram_model = kwargs.get("variogram_model", "linear")
        self.sill = 31
        self.range = 100000
        self.nugget = 25
        # Slope is calculated based on given values of sill, range, and nugget from Clay's thesis
        self.slope = 0.00006 
        self.exponent = kwargs.get("exponent", None)
        self.scale = kwargs.get("scale", None)

        self.verbose = kwargs.get("verbose", False)
        self.enable_plotting = kwargs.get("enable_plotting", False)
        self.nlags = kwargs.get("nlags", 6)
        self.weight = kwargs.get("weight", False)

    def set_params(self, **params):
        if "variogram_model" in params:
            self.variogram_model = params.pop("variogram_model")

        if "sill" in params:
            self.sill = params.pop("sill")
        if "range" in params:
            self.range = params.pop("range")
        if "nugget" in params:
            self.nugget = params.pop("nugget")
        if "slope" in params:
            self.slope = params.pop("slope")
        if "exponent" in params:
            self.exponent = params.pop("exponent")
        if "scale" in params:
            self.scale = params.pop("scale")

        if "verbose" in params:
            self.verbose = params.pop("verbose")
        if "enable_plotting" in params:
            self.enable_plotting = params.pop("enable_plotting")
        if "nlags" in params:
            self.nlags = params.pop("nlags")
        if "weight" in params:
            self.weight = params.pop("weight")

        self.estimator.set_params(**params)
        return self

    def get_params(self, deep=True):
        return {
            "variogram_model": self.variogram_model,

            "sill": self.sill,
            "range": self.range,
            "nugget": self.nugget,
            "slope": self.slope,
            "scale": self.scale,
            "exponent": self.exponent,

            "verbose": self.verbose,
            "enable_plotting": self.enable_plotting,
            "nlags": self.nlags,
            "weight": self.weight,
            **self.estimator.get_params(deep),
        }


    def fit(self, X, y):
        self.geom_col_ = X.geometry.name
        _X = X.drop(columns = self.geom_col_)
        base_model = self.estimator.fit(_X, y)
        base_preds = self.estimator.predict(_X)
        #generate semivariogram for RF residuals
        self.OK_ = OrdinaryKriging(
            X.geometry.x,
            X.geometry.y,
            y.values - base_preds,
            verbose = self.verbose,
            enable_plotting = self.enable_plotting,
            nlags=self.nlags,
            weight=self.weight,
        )
        variogram_parameters = {
            "sill": self.sill,
            "range": self.range,
            "nugget": self.nugget,
            "slope": self.slope,
            "scale": self.scale,
            "exponent": self.exponent,
        }

        #get items in variogram_parameters that are not None
        variogram_parameters = {k: v for k, v in variogram_parameters.items() if v is not None}

        self.OK_.update_variogram_model(
            variogram_model = self.variogram_model,
            variogram_parameters = variogram_parameters,
            nlags = self.nlags,
            weight = self.weight,
        )

        return self

    def predict(self, X):
        _X = X.drop(columns=self.geom_col_)
        base_preds = self.estimator.predict(_X)
        self.z_, self.ss_ = self.OK_.execute("points", xpoints=X.geometry.x, ypoints=X.geometry.y)
        return self.z_.data + base_preds

    def kriged_residuals(self):
        return self.z_.data, self.ss_.data

    def score(self, X, y):
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2

class OrdinaryKrigingRF_raf(BaseEstimator, RegressorMixin):
    """
    Ordinary kriging with random forest residuals.
    Takes a (geo)dataframe with a geometry column and a column of values to predict as input to fit().
    Takes a (geo)dataframe with a geometry column as input to predict().
    Builds a spatially-agnostic model on all fit data, and uses it to predict on that fit data.
    Takes the residuals from that prediction and uses them to fit an ordinary kriging interpolation.

    The final prediction is the sum of the global model prediction and the value of the kriged surface
        at the site of the test point.

    Parameters
    ----------
    variogram_model : str, default="linear"
        Variogram model to use for kriging.
    verbose : bool, default=False
        Whether to print variogram model parameters.
    enable_plotting : bool, default=False
        Whether to plot variogram model.
    **params : dict
        Parameters to pass to the random forest model.
    """

    def __init__(self, **kwargs):
        self.estimator = RandomForestRegressor(n_estimators = 766, max_features = 1.0, max_depth = 20,
                           min_samples_split = 2, min_samples_leaf = 4, bootstrap = True, random_state = 1)

        self.variogram_model = kwargs.get("variogram_model", "linear")
        self.sill = kwargs.get("sill", None)
        self.range = kwargs.get("range", None)
        self.nugget = kwargs.get("nugget", None)
        self.slope = kwargs.get("slope", None)
        self.exponent = kwargs.get("exponent", None)
        self.scale = kwargs.get("scale", None)

        self.verbose = kwargs.get("verbose", False)
        self.enable_plotting = kwargs.get("enable_plotting", False)

    def set_params(self, **params):
        if "variogram_model" in params:
            self.variogram_model = params.pop("variogram_model")

        if "sill" in params:
            self.sill = params.pop("sill")
        if "range" in params:
            self.range = params.pop("range")
        if "nugget" in params:
            self.nugget = params.pop("nugget")
        if "slope" in params:
            self.slope = params.pop("slope")
        if "exponent" in params:
            self.exponent = params.pop("exponent")
        if "scale" in params:
            self.scale = params.pop("scale")

        if "verbose" in params:
            self.verbose = params.pop("verbose")
        if "enable_plotting" in params:
            self.enable_plotting = params.pop("enable_plotting")

        self.estimator.set_params(**params)
        return self

    def get_params(self, deep=True):
        return {
            "variogram_model": self.variogram_model,

            "sill": self.sill,
            "range": self.range,
            "nugget": self.nugget,
            "slope": self.slope,
            "scale": self.scale,
            "exponent": self.exponent,

            "verbose": self.verbose,
            "enable_plotting": self.enable_plotting,
            **self.estimator.get_params(deep),
        }


    def fit(self, X, y):
        self.geom_col_ = X.geometry.name
        _X = X.drop(columns = self.geom_col_)
        self.estimator.fit(_X, y)
        base_preds = self.estimator.predict(_X)
        resids = y.values - base_preds
        #generate semivariogram for RF residuals
        self.OK_ = OrdinaryKriging(
            X.geometry.x,
            X.geometry.y,
            resids,
            verbose = self.verbose,
            enable_plotting = self.enable_plotting
        )
        variogram_parameters = {
            "sill": self.sill,
            "range": self.range,
            "nugget": self.nugget,
            "slope": self.slope,
            "scale": self.scale,
            "exponent": self.exponent,
        }

        #get items in variogram_parameters that are not None
        variogram_parameters = {k: v for k, v in variogram_parameters.items() if v is not None}

        self.OK_.update_variogram_model(
            variogram_model = self.variogram_model,
            variogram_parameters = variogram_parameters
        )

        #add residuals as a column to _X
        _X["residuals"] = resids
        self.estimator.fit(_X, y)

        return self

    def predict(self, X):
        _X = X.drop(columns=self.geom_col_)
        self.z_, self.ss_ = self.OK_.execute("points", xpoints=X.geometry.x, ypoints=X.geometry.y)
        _X['residuals'] = self.z_.data
        preds = self.estimator.predict(_X)
        return preds

    def kriged_residuals(self):
        return self.z_.data, self.ss_.data

    def score(self, X, y):
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        return r2
