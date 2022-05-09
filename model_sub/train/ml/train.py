# %%
import pandas as pd
import numpy  as np
import sqlalchemy

import matplotlib.pyplot as plt

from sklearn import metrics, model_selection
from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics

from feature_engine import imputation
from feature_engine import encoding

import scikitplot as skplt

#helper functions
pd.set_option('display.max_columns', None)
# %%
# SAMPLE
conn = sqlalchemy.create_engine("sqlite:///../../../data/gc.db")
df = pd.read_sql_table('tb_abt_sub', conn)

#base Out of Time (back-test)
dt_oot = df[df['dtReff'].isin(['2022-01-15', '2022-01-16'])].copy()
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
model = ensemble.RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs = -1)

## Definir um pipeline
model_pipe = pipeline.Pipeline(steps=[('Input -1', input_1),
                                      ('One Hot', onehot),
                                      ('Modelo', model)])
# %%
model_pipe.fit(X_train, y_train)

# %%

## Assess
#training base
y_train_pred = model_pipe.predict(X_train)
y_train_prob = model_pipe.predict_proba(X_train)

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
y_test_pred = model_pipe.predict(X_test)
y_test_prob = model_pipe.predict_proba(X_test)

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

features_model = model_pipe[:-1].transform(X_train.head()).columns.tolist()

fs_importance = pd.DataFrame({'importance': model_pipe[-1].feature_importances_,
                                'feature': features_model}) 

fs_importance.sort_values('importance', ascending = False).head(20)
# %%
