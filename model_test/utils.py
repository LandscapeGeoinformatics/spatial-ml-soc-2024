# Useful functions for the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fasttreeshap
import random
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def metrics_transformer(name, X_train, X_test, y_train, y_test, model):
  '''
  function to get the necessary inputs to the metrics function from a python sklearn model
  inputs: string name to label the model, training data without target column, testing data without target column, training data target column, testing data target column, the sklearn model itself
  outputs: list containing model_metrics inputs
  '''
  train_preds = model.predict(X_train)
  test_preds = model.predict(X_test)
  model_preds = model.oob_prediction_

  return [name, y_train, y_test, train_preds, test_preds, model_preds]

def model_metrics(name, train_y, test_y, train_preds, test_preds, model_preds, df = None):
  '''
  function to return metrics about a model in a dataframe that can be appended to other model results for display
  inputs: string name to label the model, training set target (SOC), testing set target (SOC), model predictions on training set, model predictions on testing set, model OOB predictions
  outputs: single-row dataframe containing the passed name of the model, the model's r^2 on testing data, oob score, pearson corr on testing data, pearson corr on training data, and r^2 on training data
  '''

  pearsons_test = pearsonr(test_y, test_preds)[0] #pearson correlation of test set results with reality
  pearsons_train = pearsonr(train_y, train_preds)[0] #pearson correlation of train set results with reality
  r2 = r2_score(test_y, test_preds) #r^2 of the model prediction the testing data
  forest_score = r2_score(train_y, train_preds) #r^2 of the model predicting the training data
  oob_score = r2_score(train_y, model_preds) #calculate out-of-bag score (r^2 of training values with out-of-bag predictions of those values)

  metrics = pd.DataFrame([[name, r2, oob_score, pearsons_test, pearsons_train, forest_score]], columns = ['Seed', 'R$^{2}$', 'OOB Score', "Pearson's R (test)", "Pearson's R (train)", 'Forest Score'])

  if df is not None:
    return(pd.concat([df, metrics]))
  else:
    return(metrics)


def rand_or_int(input = 'rand'):
  '''
  function to return a random integer between 0 and 1000, or the input if it is an integer
  inputs: integer or string 'rand'
  outputs: integer
  '''
  if input == 'rand':
    return(random.randint(0,1000))
  else:
    return(input)

def plot_shap_vals(shaps):
  '''
  function to plot shap values
  inputs: shap values
  outputs: none
  '''
  plt.figure()
  fasttreeshap.plots.beeswarm(shaps, max_display = 25, show = False, color_bar = False)
  # plt.ylabel("Variable", fontsize = 8)
  plt.xlabel("SHAP value (impact on model output)", fontsize = 8)
  plt.xticks(fontsize='small')
  plt.yticks(fontsize='small')
  m = cm.ScalarMappable(cmap = fasttreeshap.plots.colors.red_blue)
  m.set_array([0,1])
  cb = plt.colorbar(m, ax = plt.gca(), ticks = [0,1], shrink = 0.5, label = "Feature Value")
  cb.set_ticklabels(['Low', 'High'])
  cb.ax.tick_params(labelsize = 8)
  cb.outline.set_visible(False)
  for text in plt.gca().texts:
      text.set_fontsize(8)
  plt.gcf().set_size_inches(5,4)
  plt.gcf().set_dpi(100)
  plt.tight_layout()

from sklearn.cluster import KMeans
def spatial_quoting_split(data, label_name, n_quotas=4, train_size=0.7, test_size=None, random_state=None):
  """
  Spatial quoting train/test splitting algorithm, as described by Baglaeva et al. (2020).

  "Space&max&min quote: spatial quoting of raw data, which consists of three steps:
  1. The survey area was contoured by a convex polygon such that a geodesic line drawn between any two points (real sampling points) was inside this polygon. Contouring was done by connecting the boundary points. This procedure satisfies the interpolation condition, since any prediction is inside this polygon.
  2. The polygon was split into areas (spatial quotas) including the same number of observations (real sampling points). The boundary points of the polygon were included in the training subsample. In addition, the maximum and minimum values from each spatial quota were necessarily included in the training subsample. We propose that the number of spatial quotas should be at least four. This is consistent with selected geographic directions. In the presence of pronounced geographic features (e.g., a dramatic change in vegetation, the presence of pronounced relief), this zone should also be allocated to a separate spatial quota (quotas). The volume of the spatial quota is limited by the required amount for the test subset. With regard to general statistical considerations, it is desirable that this subset contain at least 30 points for the robust assessment of interpolation accuracy. If the amount of data allows, then the number of spatial quotas may be increased.
  3. Points for the training subsample were randomly selected from each spatial quota so that its proportion was 70%.
  4. The final training subsample consisted of (1) boundary points, (2) maximum and minimum values from each spatial quota, and (3) random additions from each spatial quota to 70%. The test sample included the remaining 30% of the points."

  Parameters
  ----------
  data : geopandas.GeoDataFrame
      The data to split into training and test samples.
  label_name : str
      The name of the column containing the target variable.
  n_quotas : int, optional
      The number of spatial quotas to split the data into, by default 'auto'. If 'auto', the number of spatial quotas will be determined by the number of points in the data such that all spatial quotas have at least 30 test points.
  train_size : float, optional
      The proportion of the data to include in the training sample, by default 0.7. If test_size is not None, this will be ignored.
  test_size : float, optional
      The proportion of the data to include in the test sample, by default 0.3.
  random_state : int, optional

  Returns
  -------
  training_sample : geopandas.GeoDataFrame
      The training sample.
  test_sample : geopandas.GeoDataFrame
      The test sample.

  References
  ----------
  Baglaeva, E.M., Sergeev, A.P., Shichkin, A.V. et al. The Effect of Splitting of Raw Data into Training and Test Subsets on the Accuracy of Predicting Spatial Distribution by a Multilayer Perceptron. Math Geosci 52, 111–121 (2020). https://doi.org/10.1007/s11004-019-09813-9
  """
  if test_size is not None:
    train_size = 1 - test_size

  #if n_quotas is auto, number of spatial quotas should be such that the test sample contains at least 30 points
  if n_quotas == 'auto':
    n_quotas = int(np.floor(test_size * gdf.shape[0] / 30))

  # Step 1: Contouring
  convex_hull = data.unary_union.convex_hull
  # points in the dataframe that are part of the hull boundary
  boundary_points = data[data.geometry.touches(convex_hull)]

  # Step 2: Splitting into areas (spatial quotas)
  # cluster the points into n_quotas spatial clusters
  kmeans = KMeans(n_clusters=n_quotas, random_state=random_state)
  kmeans.fit(np.array(list(zip(data.geometry.x, data.geometry.y))))

  # hmmmm the line above gives ValueError: setting an array element with a sequence.
  data['cluster'] = kmeans.labels_

  # Step 3: Add boundary points and max/min values to training sample
  # add boundary points to training sample
  training_sample = boundary_points.copy()
  # add max/min values to training sample
  training_sample = training_sample.append(data.sort_values(label_name).drop_duplicates('cluster', keep='first'))
  training_sample = training_sample.append(data.sort_values(label_name).drop_duplicates('cluster', keep='last'))
  for cluster in data.cluster.unique():
    cluster_data = data[data.cluster == cluster]
    training_sample = training_sample.append(cluster_data.loc[cluster_data[label_name].idxmax()])
    training_sample = training_sample.append(cluster_data.loc[cluster_data[label_name].idxmin()])

  # Step 4: Randomly select points from each cluster to make up 70% of the training sample
  for cluster in data.cluster.unique():
    cluster_data = data[data.cluster == cluster]
    n_pts = len(cluster_data)
    n_train = int(n_pts * train_size)
    n_pts_to_add = n_train - len(training_sample[training_sample.cluster == cluster])
    if n_pts_to_add > 0:
      training_sample = training_sample.append(cluster_data.sample(n_pts_to_add, random_state=random_state))
  # remove duplicates
  training_sample = training_sample.drop_duplicates()
  # add remaining points to test sample
  test_sample = data[~data.index.isin(training_sample.index)]

  training_sample = training_sample.drop(columns='cluster')
  test_sample = test_sample.drop(columns='cluster')
  return training_sample, test_sample
