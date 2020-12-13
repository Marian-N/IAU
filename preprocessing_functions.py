import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import re

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler


# returns true if we can merge duplicates -> there are no conflicts
def mergeable(dfo):
    dfo_duplicates = dfo[dfo.duplicated(['name'], keep=False)]
    unique_names = dfo_duplicates['name'].unique()
    for name in unique_names:
        duplicates = dfo_duplicates.loc[dfo_duplicates['name'] == name]
        for col in duplicates.columns:
            if col == 'ID':
                continue
            values = duplicates[col].unique()
            if len(values) > 2:
                return False
            elif len(values) == 1:
                continue
            else:
                if pd.isnull(values[0]) or pd.isnull(values[1]):
                    continue
    return True


# merges duplicates
def merge(name, dfo_duplicates):
    df = dfo_duplicates.loc[dfo_duplicates['name'] == name]
    return df.groupby(['name'], as_index=False).first()


def merge_other_personal(df):
    dfo_duplicates = dfo[dfo.duplicated(['name'], keep=False)]
    dfo_unique = dfo.drop_duplicates(subset=["name"], keep=False)
    merged = []

    for name in dfo_duplicates['name'].unique():
        merged.append(merge(name, dfo_duplicates))

    dfo_unique = dfo_unique.append(merged)

    x = pd.merge(dfp, dfo_unique, on='name')
    x = x.drop(columns=['address_y', 'ID_y'])
    x = x.rename(columns={"ID_x": "ID", "address_x": "address"})
    return x

def unpack_medical(df):
    x = df.copy()
    for i, row in x.iterrows():
        if not pd.isnull(x.at[i, 'medical_info']):
            x.at[i, 'medical_info'] = json.loads(x["medical_info"][i].replace("\'", "\""))
    # vytvorenie stlpcov z medical_info a ich spojenie so zvyskom dataframe
    df_med_info = x["medical_info"].apply(pd.Series)
    df_med_info = df_med_info.drop(0, 1)
    x = pd.concat([x, df_med_info], axis = 1).drop("medical_info", axis = 1)
    return x

def obj_to_float(df):
    x = df.copy()
    # kurtosis_oxygen
    x['kurtosis_oxygen'] = x['kurtosis_oxygen'].astype(np.float)
    # mean_oxygen
    x['mean_oxygen'] = x['mean_oxygen'].astype(np.float)
    # skewness_oxygen
    x['skewness_oxygen'] = x['skewness_oxygen'].astype(np.float)
    # std_oxygen
    x['std_oxygen'] = x['std_oxygen'].astype(np.float)
    return x

def remove_unimportant_columns(df):
    x = df.copy()
    x = x.drop(['ID', 'name', 'fnlwgt', 'date_of_birth'], axis=1)
    return x

def remove_space(df):
    x = df.copy()
    x = df.apply(lambda y: y.str.strip() if y.dtype == "object" else y)
    return x

def put_0_1_values(df):
    x = df.copy()
    
    # pohlavia: Male -> 1; Female -> 0
    x['sex'] = x['sex'].replace('Male', 1)
    x['sex'] = x['sex'].replace('Female', 0)
    
    # tehotnost: T -> 1; F -> 0
    x['pregnant'] = x['pregnant'].replace(regex='(?i)f.*', value=0)
    x['pregnant'] = x['pregnant'].replace(regex='(?i)t.*', value=1)
    
    # muzi oznaceni ako tehotny su prepisani na 0
    x.loc[(x['pregnant'] == 1) & (x['sex'] == 1), 'pregnant'] = 0
    
    # zmena income hodnot: <=50K -> 0; >50K -> 1
    x['income'] = x['income'].replace('<=50K', 0)
    x['income'] = x['income'].replace('>50K', 1)
    
    # zmena nazvov stlpcov na presnejsie
    x = x.rename(columns={"pregnant": "is_pregnant", "income": "income_>50K"})
    
    return x

def education_analysis(df):
    # prints unique values in education
    x = df.copy()
    unique_edu = pd.unique(x['education'])
    print("Pred zjednotenim:\n", unique_edu)
    
    # Zjednotenie reprezentacii
    x['education'] = x['education'].replace(regex='(?i)_', value='-')
    unique_edu = pd.unique(x['education'])
    print("\nPo zjednoteni:\n", unique_edu)
    
    # hodnoty v education-num a v education
    print("\nHodnoty v jednotlivych education:")
    for item in unique_edu:
        edu_num = x.query("education == @item")["education-num"].unique()
        print(item, edu_num)
        
#vrati unikatne hodnoty v stlpci education
def get_unique_edu(df):
    x = df.copy()
    x['education'] = x['education'].replace(regex='(?i)_', value='-')
    unique_edu = pd.unique(x['education'])
    return unique_edu

# rozne hodnoty education-num pre unikatny education zmeni na jedno (napr.: 5th-6th [  3. 300.] -> 3)
def get_edu_num(edu_num):
    for num in edu_num:
        if num is None:
            continue
        elif num < 100:
            return int(num)

def transform_education(df):
    x = df.copy()
    edu_to_num = {}
    #vytvorenie dictionary s moznymi hodnotami v jendotlivych education values
    for item in get_unique_edu(x):
        edu_num = x.query("education == @item")["education-num"].unique()
        edu_to_num[item] = get_edu_num(edu_num)
    
    # zmena moznych hodnot v education na rovnake
    x['education'] = x['education'].replace(regex='(?i)_', value='-')

    # namapuje nazvy education na cisla z dictionary
    x["education"] = x["education"].map(edu_to_num)
    
    # Dropne nepotrebny column education-num (bol nahradeny)
    x = x.drop(['education-num'], axis=1)
    return x

def find_state(address):
    i = re.search('\x5cn.+\D', address)
    return address[i.start():i.end()][-3:-1]
    #return address[-8:][:2]

def address_to_state(df):
    x = df.copy()
    x['address'] = x['address'].apply(find_state)
    x = x.rename(columns={"address": "state"})
    return x


def replace_with_nan(df):
    x = df.copy()
    x = x.replace(['??', '?'], np.nan)
    return x

def transform_workclass(df):
    x = df.copy()
    x['workclass'] = x['workclass'].str.lower()
    return x


def integration_combined(df):
    # merges other and personal df
    df = merge_other_personal(df)
    # upacks medical info into columns
    df = unpack_medical(df)
    # changes objects unpacked in medical info (odfygen) to float
    df = obj_to_float(df)

    # remove unimportant columns like name or fnlwgt
    df = remove_unimportant_columns(df)
    # removes spaces at the start of values
    df = remove_space(df)
    # replace values that have only 2 options vwith 1 or 2
    df = put_0_1_values(df)
    # gets rid of education and replaces it with number from education-num
    df = transform_education(df)
    # gets rid of address and only keep state
    df = address_to_state(df)
    # replaces ? with nan
    df = replace_with_nan(df)
    # workclass values into lower case -> gets rid of duplicates
    df = transform_workclass(df)
    return df

def label_encode_strings(df):
    """Nahradi stringy ciselnymi hodnotami pomocou sklearn LabelEncoderu."""
    x = df.copy()
    
    enc = LabelEncoder()
    cols_to_transform = ['race', 'state', 'marital-status', 'occupation', 'relationship', 'native-country', 'workclass']
    x[cols_to_transform] = x[cols_to_transform].apply(enc.fit_transform)
    
    print("Table of mapping numeric values to cathegorical data:\n")
    for i in cols_to_transform:
        print(f'--- {i} ---')
        values = df[i].unique()
        encoded = enc.fit(values).transform(values)
        encoding = enc.inverse_transform(encoded)
        for e in range(len(encoded)):
            print(f'{encoded[e]} : {encoding[e]}')
        print('\n')
    
    return x

def separate_by_dtype(df):
    """Rozdeli dataframe na stringy a numericke data."""
    df_num = pd.DataFrame()
    df_str = pd.DataFrame()

    for col in df:
        # Ak je to int alebo float tak sa jedna o numericke data
        if df[col].dtypes in ['float64', 'int64']:
            df_num[col] = df[col]
        else: # Inak string
            df_str[col] = df[col]
    
    return df_num, df_str

def replace_missing_strings(df):
        """Nahradi chybajuce stringy pomocou SimpleImputer zo sklearn.impute strategiou most_frequent."""
        x = df.copy()
        x = SimpleImputer(strategy="most_frequent").fit_transform(x)
        
        # Z novych hodnot sa vytvori dataframe
        x = pd.DataFrame(x)
        
        # Pomenujeme stlpce a riadky rovnako ako v povodnom dataframe
        x.columns = df.columns
        x.index = df.index
        
        return x

def replace_missing_numbers(df, strat='median'):
    """Nahradi chybajuce numericke data pomocou zvolenej strategie (median, mean alebo kNN)."""
    x = df.copy()
    
    # Pre zvolenu strategiu sa vytvori imputer
    if strat in ['mean', 'median']:
        imp = SimpleImputer(strategy=strat)
    else:
        imp = KNNImputer()
    
    # Doplnia sa chybajuce hodnoty
    x = imp.fit_transform(x)
    
    # Z novych hodnot sa vytvori dataframe
    x = pd.DataFrame(x)
    
    # Pomenujeme stlpce a riadky rovnako ako v povodnom dataframe
    x.columns = df.columns
    x.index = df.index
    
    x['class'] = x['class'].round()
    x['income_>50K'] = x['income_>50K'].round()
    
    return x

def replace_missing_values(df, strat='median'):
    df_num, df_str = separate_by_dtype(df)
    
    df_str = replace_missing_strings(df_str)
    df_num = replace_missing_numbers(df_num, strat)
    
    return pd.concat([df_str, df_num], axis=1, sort=False)

def outliers(df, method='percentil'):
    x = df.copy()
    # vyber stlpcov pre ktore  chceme outlierov najst
    outliners_for =['skewness_glucose','mean_glucose', 'std_glucose', 'kurtosis_glucose', 
                    'skewness_oxygen', 'mean_oxygen', 'std_oxygen', 'kurtosis_oxygen']
    
    for column in df.columns:
        if column in outliners_for:
            # vypocitame mean standard deviation pre stlpec
            mean = x[column].mean()
            std_dev = x[column].std()
            # zistime hranicne hodnoty
            border_right = mean + 3 * std_dev
            border_left = mean - 3 * std_dev

            # remove len ako test
            if (method == 'remove'):
                x.drop(x.loc[(x[column] > border_right)].index, inplace = True, axis=0)
                x.drop(x.loc[(x[column] < border_left)].index, inplace = True, axis=0)
            
            # odstaranenie outlinerov pomocou percentilov
            elif (method == 'percentil'):
                #vypocet percentilov
                p_95 = x[column].quantile(0.95)
                p_05 = x[column].quantile(0.05)
                # nahradenie hodnot za hranicami s percentilmi
                x.loc[(x[column] > border_right), column] = p_95
                x.loc[(x[column] < border_left), column] = p_05
    
    x = x.reset_index(drop=True)
    return x

def transform(df, method='power', plot='age'):
    x = df.copy()
    # vykreslenie grafu pred
    plt.figure()
    sns.histplot(x[plot], kde=True, color="Green")
    
    # Power Transform
    if method == 'power':
        trans = PowerTransformer(method='yeo-johnson')
    # Min Max Scale
    else:
        trans = MinMaxScaler()
    
    # aplikacia transformovania -> vrati array
    x = trans.fit_transform(x)
    # convert the array back to a dataframe -> columns mame zapisane ako ciselne hodnoty
    dataset = pd.DataFrame(x)
    # zmena ciselnych nazvov stlpcov v tabulke na slovne
    X_imputed_df = pd.DataFrame(x, columns = df.columns)
    # vykreslenie grafu po transformacii
    plt.figure()
    sns.histplot(X_imputed_df[plot], kde=True, color="Red")
    
    return X_imputed_df


def create_pipeline(dfo_got, dfp_got):
    return Pipeline([
    ('integration', FunctionTransformer(func=integration_combined)),
    ('replace_nan', FunctionTransformer(func=replace_missing_values, kw_args={'strat': 'median'})),
    ('to_numeric', FunctionTransformer(func=label_encode_strings)),
    ('replace_outliers', FunctionTransformer(func=outliers, kw_args={'method': 'percentil'})),
    ('transform', FunctionTransformer(func=transform, kw_args={'method': 'power', 'plot': 'std_glucose'}))
])
