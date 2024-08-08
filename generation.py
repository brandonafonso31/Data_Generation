import geopandas
import numpy as np
import pandas as pd
from shapely import MultiPolygon, unary_union, Point
from torch.utils.data import DataLoader, IterableDataset

import torch

import random

def get_minimal_sampling(df_test, df_model):

    dict_samples = {k:v for k, v in zip(df_test["area_name"], df_test["new_samples"])}
    list_of_df = []
    cpt_samples = 0

    for idx, row in df_model.iterrows():

        if row["area_name"] != '':

            if dict_samples[row["area_name"]] > 0:
                list_of_df.append(row)
                dict_samples[row["area_name"]] -= 1

        cpt_samples += 1

        if sum(dict_samples.values()) == 0:
            break
        
        
    return pd.DataFrame(list_of_df), cpt_samples

def create_df_new_samples(df_raw, var=0.02):

    df_test = pd.DataFrame(df_raw['area_name'].value_counts().reset_index())
    df_test = df_test.rename(columns={"index" : "area_name", "area_name":"count"})
    df_test.loc[:, "new_samples"] = df_test["count"].apply(gen_num_samples, args=(var,))
    df_test.loc[:, "perc."] = df_test["new_samples"] / df_test["count"] * 100 - 100
    df_test.loc[:, "delta"] = df_test["new_samples"] - df_test["count"]
    df_test.loc[df_test["delta"].idxmax(), "new_samples"] -= df_test["delta"].sum()
    df_test.loc[:, "perc."] = df_test["new_samples"] / df_test["count"] * 100 - 100
    df_test.loc[:, "delta"] = df_test["new_samples"] - df_test["count"]
    return df_test

def new_df_sample(df_test, df_model):

    list_of_df = []
    for idx, row in df_test.iterrows():
                
        if len(df_model[df_model["area_name"] == row["area_name"]]) >= row["new_samples"]:
            list_of_df.append(df_model[df_model["area_name"] == row["area_name"]].sample(int(row["new_samples"])))
            
        else:            
            list_of_df.append(df_model[df_model["area_name"] == row["area_name"]])

    return pd.concat(list_of_df, ignore_index=True)

def gen_num_samples(num_samples, var):

    sign = bool(random.getrandbits(1))
    
    if sign:
        x = random.uniform(1, 1+var)
    else:
        x = random.uniform(1-var, 1)

    return int(round(num_samples * x))

class RandomWalkDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.size =  len(dataset)
    def generator(self):
        while True:
            yield self.dataset[random.randint(0,self.size-1)]
    def __iter__(self):
        return self.generator()

def get_iterator(df, cols, batchsize):
    train_data = df[cols].to_numpy(dtype=np.float32)
    dataset = torch.from_numpy(train_data)
    dataloader = DataLoader(RandomWalkDataset(dataset), batch_size=batchsize, drop_last=False)
    return iter(dataloader)

def compute_x_by_y(df, col_x, col_y, reindex = True):

    table = []
    
    raw_unique = sorted(df[col_x].unique())
    col_unique = sorted(df[col_y].unique())

    for x in raw_unique:
    
        l_x = []
        df_x = df[df[col_x] == x]

        for y in col_unique:

            occ_y = len(df_x[df_x[col_y] == y])
            l_x.append(occ_y)
            
        table.append(l_x)
        
    df_table = pd.DataFrame(data = table, columns = col_unique)
    
    if reindex:
        
        df_table.index += 1
        
    return df_table

def get_stats(col, df_raw, df_fake):

    df_test = pd.DataFrame(df_raw[col].value_counts().sort_index())
    df_test["fake"] = df_fake[col].value_counts().sort_index()
    df_test["delta"] = (df_test[col] - df_test["fake"]).abs()
    df_test["perc."] = round(df_test["delta"] / df_test[col] * 100)
    return df_test

def get_centroid(area, zones):
    if area in zones["area"].values:
        return zones['centroid'][zones["area"] == area].values[0]
    else:
        return Point(0, 0)
    
def get_area(area, zones):
    if area in zones["area"].values:
        return zones['area'][zones["area"] == area].values[0]
    else:
        return 0
    
def get_point_in_area(point, zones):
    res = zones["area"][zones.contains(point)].values
    if res.size > 0:
        return res[0]
    else:
        return ""
    

def reverse_df_minmax(df, cols_to_unminmax, df_ref, mini = -0.95, maxi = 0.95):
    
    df_res = df_ref.copy()
    
    X_ref = df_res[cols_to_unminmax].values.astype(np.float32)
    data = df[cols_to_unminmax].astype(np.float32)
    rev_data = reverse_min_max(data, -0.95, 0.95, X_ref)

    for idx, col in enumerate(cols_to_unminmax):

        df_res[col] = rev_data[:, idx]

    return df_res

def feat_minmax(col, mini=-0.95, maxi=0.95):
    mi_x = col.min()
    ma_x = col.max()
    return col.apply(min_max, args=(mi_x, ma_x, -0.95, 0.95)).astype(np.float32)




def encode_periodic(value, period):
    angle = (2 * np.pi * value) / period
    return np.sin(angle), np.cos(angle)

def decode_periodic(sin_value, cos_value, period):
    angle = np.arctan2(sin_value, cos_value)
    if angle < 0:
        angle += 2 * np.pi  # Ensure angle is in [0, 2*pi) range
    value = (angle * period) / (2 * np.pi)
    return round(value)

def min_max(x, mi_x, ma_x, mi=0, ma=0):    
    X_std = (x - mi_x) / (ma_x - mi_x)
    return X_std * (ma - mi) + mi

def reverse_min_max(data, mi, ma, X_ref):    
    new_data = data.values - mi
    new_data /= (ma - mi)       
    new_data = new_data * (X_ref.max(axis=0) - X_ref.min(axis=0))
    new_data += X_ref.min(axis=0)
    return new_data

def transformation_fin(ListCasernesAnticipes : list, area_names : list):
  '''
  fonction servant à savoir si toutes les premières casernes dans le classement sont valables pour notre création de area
  '''
  for nom_caserne in ListCasernesAnticipes:
    if nom_caserne in area_names:
      return False
  return True

def remplace_si_necessaire(x : geopandas.GeoDataFrame, nom_premiere_caserne, i, ListCasernesAnticipes):
  '''
  Cette fonction regarde pour un iris si il son area actuellement attribué est conforme aux attentes.
  Sinon il l’échange avec le prochain dans la liste (un par un)
  '''
  if ((str(x[nom_premiere_caserne]) in ListCasernesAnticipes) or (not control_debut_chaine_bool(str(x[nom_premiere_caserne]), ['Z', 'X-']))):
    return x[nom_premiere_caserne[:-1] + str(i)]
  return x[nom_premiere_caserne]

def control_debut_chaine(string : str, list_elem : list):
  for elem in list_elem:
    if string.startswith(elem):
      return ''
  return string

def control_debut_chaine_bool(string : str, list_elem : list):
  boolean = True
  for elem in list_elem:
    if string.startswith(elem):
      boolean = False
  return boolean

def transformation_contre_anticipation(gdf_iris : geopandas.GeoDataFrame, nom_premiere_caserne : str , ListCasernesAnticipes = []):
  '''
  Cette fonction appelle la fonction remplace_si_necessaire jusqu’à ce que le premier area de chaque iris soit bien un area qui respecte nos critères
  '''
  i = 1
  while not transformation_fin(ListCasernesAnticipes, list(set(gdf_iris[nom_premiere_caserne]))):
    gdf_iris[nom_premiere_caserne] = gdf_iris.apply(lambda x : remplace_si_necessaire(x, nom_premiere_caserne, i, ListCasernesAnticipes), axis=1)
    i+=1
  return gdf_iris

def union_iris(gdf_iris : geopandas.GeoDataFrame, nom_premiere_caserne : str , ext = True, ListCasernesAnticipes = []):

  gdf_iris = gdf_iris.drop(np.where(gdf_iris[nom_premiere_caserne].apply(lambda x : control_debut_chaine(x, ['Z'])) == '')[0], axis = 0)#, 'X-'
  gdf_iris = gdf_iris.explode(index_parts=False)
  gdf_iris = transformation_contre_anticipation(gdf_iris, nom_premiere_caserne, ListCasernesAnticipes )

  area_name_net = list(set(gdf_iris[nom_premiere_caserne]))
  areas_geo = []
  for nom in area_name_net :
    areas_geo.append(MultiPolygon( list(gdf_iris[gdf_iris[nom_premiere_caserne] == nom].geometry) ))

  areas = geopandas.GeoDataFrame({'area' : area_name_net, 'geometry' : areas_geo} , crs=gdf_iris.crs)
  areas.geometry = areas.geometry.to_crs(2154)

  if ext:
    areas.geometry = areas.geometry.apply(lambda x : unary_union(x))
    return areas
  return areas

def get_inter_from_sector(area, df_sorties):
    return df_sorties[(df_sorties["Nom du Centre"] == area) & 
                      (df_sorties["Ordre Renfort"] == "P") & 
                      (df_sorties["Annulation de départ (O/N)"] == "non")]["Numéro d'intervention"].to_list()

def get_inc_from_inter(list_of_inter, df_inter, perc=False):
    if perc:
        return dict(df_inter[df_inter["Numéro d'intervention"].isin(list_of_inter)]["Sinistre initial - Nom"].value_counts(normalize=True).mul(100).round(2))
    else:
        return dict(df_inter[df_inter["Numéro d'intervention"].isin(list_of_inter)]["Sinistre initial - Nom"].value_counts())
    
# def get_stats(df):

#     meanx = df.iloc[:, 2:].mean(axis=1)
#     stdx = df.iloc[:, 2:].std(axis=1)
#     df2 = df.copy()

#     df2.loc[:,["mean"]] = meanx
#     df2.loc[:,["std"]] = stdx
#     df2.loc[:,["min2std"]] = meanx - 2 * stdx
#     df2.loc[:,["max2std"]] = meanx + 2 * stdx

#     return df2

def get_gt(row):
    r2check = row[2:-4]
    gt = r2check[r2check > row['max2std']].index.to_list() 
    return gt

def get_lt(row):
    r2check = row[2:-4]
    lt = r2check[r2check < row['min2std']].index.to_list() 
    # print(gt, lt)
    return lt

def get_duration_from_inter(list_of_inter, df_inter):
    
    duree = df_inter[df_inter["Numéro d'intervention"].isin(list_of_inter)]["duree"]
    
    mean = duree.mean()
    median = duree.median()
    std = duree.std()
    
    return (mean, median, std)


