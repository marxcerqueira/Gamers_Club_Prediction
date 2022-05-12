# %%
from random import random
import pandas as pd
import numpy  as np
import sqlalchemy

import matplotlib.pyplot as plt

from sklearn import metrics, model_selection
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
from sklearn import pipeline

from feature_engine import imputation
from feature_engine import encoding

import scikitplot as skplt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,\
confusion_matrix, matthews_corrcoef, make_scorer, roc_curve, precision_recall_curve

#helper functions
pd.set_option('display.max_columns', None)

# Model's performance function on training dataset
def performance(model, x_train, y_train):
    # define scoring metrics
    scoring = {'accuracy': 'accuracy',
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score),
               'roc_auc_score': make_scorer(roc_auc_score),
               'mcc': make_scorer(matthews_corrcoef)}

    # calculate scores with cross_validate
    scores = cross_validate(model, x_train, y_train, cv=10, scoring=scoring)
    
    # performance data frame
    performance = pd.DataFrame.from_dict(scores).drop(['fit_time', 'score_time'], axis=1)
    performance = pd.DataFrame(performance.mean()).T
    return performance

# %%
# SAMPLE
conn = sqlalchemy.create_engine("sqlite:///../../../data/gc.db")
df = pd.read_sql_table('tb_abt_sub', conn)

#base Out of Time (back-test)
df_oot = df[df['dtReff'].isin(['2022-01-15', '2022-01-16'])].copy()
df_train = df[~df['dtReff'].isin(['2022-01-15', '2022-01-16'])].copy()

features = df_train.columns.tolist()[2:-1]
target = 'flagSub'

#define df_train, df_test (80/20)
X_train, X_test, y_train, y_test = model_selection.train_test_split(df_train[features],
                                                                    df_train[target],
                                                                    random_state=42,
                                                                    test_size=0.2)

# %%
# EXPLORE

num_features = X_train.select_dtypes(np.number).columns.tolist()
cat_features = X_train.select_dtypes('object').columns.tolist()
# %%]
print('Missing numérico:')
is_na = X_train[num_features].isna().sum()
print(is_na[is_na>0]
)

missing_1 =  ['WinRateMirage',
              'WinRateNuke',   
              'WinRateInferno',
              'WinRateVertigo',
              'WinRateAncient',
              'WinRateDust2',  
              'WinRateTrain',  
              'WinRateOverpass',
              'vlIdadePlayer' ]
# %%
print('Missing categórico:')
is_na = X_train[cat_features].isna().sum()
print(is_na[is_na>0]
)

# %%

# MODIFY
#imputação dos dados
input_1 = imputation.ArbitraryNumberImputer(arbitrary_number = -1, variables =missing_1 )

# one hot enconding
onehot = encoding.OneHotEncoder(drop_last = True, variables=cat_features)

# %%

## MODEL
rf_clf = ensemble.RandomForestClassifier(n_estimators=200,
                                         min_samples_leaf=50,
                                         n_jobs = -1,
                                         random_state=42)

ada_clf = ensemble.AdaBoostClassifier(n_estimators=200,
                                      learning_rate=0.8,
                                      random_state=42)

et_clf = ensemble.ExtraTreesClassifier(n_estimators=200,
                                      max_depth=15,
                                      min_samples_leaf=50,
                                      n_jobs=-1,
                                      random_state=42)

dt_clf = tree.DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=50,
                                     random_state=42)

lr_clf = linear_model.LogisticRegressionCV(cv=4, n_jobs=-1)


## Definir um pipeline

params = {'n_estimators': [50,100,200,250,],
          'min_samples_leaf': [5,10,20,50,100]}

grid_search = model_selection.GridSearchCV(rf_clf, 
                                          params, 
                                          n_jobs=-1, 
                                          cv=4, 
                                          scoring='roc_auc',
                                          verbose=3,
                                          refit=True)

pipe_rf_gs = pipeline.Pipeline(steps=[('Input -1', input_1),
                                      ('One Hot', onehot),
                                      ('Modelo', grid_search)])

# %%
# train test report

def train_test_report(model, X_train, y_train, X_test, y_test, key_metric, is_prob = True):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    metric_result = key_metric(y_test, prob[:,1]) if is_prob else key_metric(y_test, pred)

    return metric_result

# %%
pipe_rf_gs.fit(X_train, y_train)

# %%
pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score')
# %%

## Assess
#training base
y_train_pred = pipe_rf_gs.predict(X_train)
y_train_prob = pipe_rf_gs.predict_proba(X_train)

acc_train = round(100*metrics.accuracy_score(y_train, y_train_pred),2)
roc_train = metrics.roc_auc_score(y_train,y_train_prob[:,1] )
precision_train = metrics.precision_score(y_train, y_train_pred)
recall_train = metrics.recall_score(y_train, y_train_pred)


print('Baseline:',round((1-y_train.mean())*100,2))
print('Acc:', acc_train)
print('roc_train', roc_train)
print('precision_train:', precision_train*100)
print('recall_train:', recall_train*100)

# %%

#testing base
y_test_pred = pipe_rf_gs.predict(X_test)
y_test_prob = pipe_rf_gs.predict_proba(X_test)

acc_test = round(100*metrics.accuracy_score(y_test, y_test_pred),2)
roc_test = metrics.roc_auc_score(y_test,y_test_prob[:,1] )
precision_test = metrics.precision_score(y_test, y_test_pred)
recall_test = metrics.recall_score(y_test, y_test_pred)

print('Baseline:',round((1-y_test.mean())*100,2))
print('acc_test', acc_test)
print('roc_test', roc_test)
print('precision_train:', precision_test*100)
print('recall_train:', recall_test*100)
# %%

skplt.metrics.plot_precision_recall_curve(y_train, y_train_prob)
plt.show()

# %%

skplt.metrics.plot_roc(y_train, y_train_prob)
plt.show()
# %%

skplt.metrics.plot_ks_statistic(y_train, y_train_prob)
plt.show()
# %%

skplt.metrics.plot_lift_curve(y_train, y_train_prob)
plt.show()
# %%

y_pro_ass = y_train_prob[:,1].copy()
y_pro_ass.sort()
y_pro_ass

# %%

df_analysis = pd.DataFrame({'target': y_train, 'prob': y_train_prob[:,1]})
df_analysis.sort_values('prob', inplace=True, ascending=False).head()

# %%
df_analysis.sort_values('prob', inplace=True, ascending=False)

# %%
df_analysis.head(10000)['target'].mean() / df_analysis['target'].mean()

# %%

X_oot, y_oot = df_oot([features]), df_oot[target]

y_prob_oot = pipe_rf_gs.predict_proba(X_oot)

roc_oot = metrics.roc_auc_score(y_oot, y_prob_oot[:,1])
print('roc oot', roc_oot)

# %%
#business report
#calculando conversão
conv_model = (df_oot.sort_values('prob', ascending=False)
                    .head(1000)
                    .mean()['prob'])


conv_baseline = (df_oot.sort_values('prob', ascending=False)
                    .mean()['prob'])


conv_model_qty = (df_oot.sort_values('prob', ascending=False)
                    .head(1000)
                    .sum()['prob'])

conv_baseline_qty = (df_oot.sort_values('prob', ascending=False)
                    .sum()['prob'])

print(f'Total de pessoas convertidas com modelo: {conv_model_qty} ({round(100*conv_model, 2)%})')
print(f'Total de pessoas convertidas com baseline: {conv_baseline_qty} ({round(100*conv_baseline, 2)%})')
# %%
