#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


DATA_PERIOD_DAYS = 2  # колво дней по которым доступны данные для прогноза


def get_x_y(events, submissions):
  
    """" создадим признаки и метку
     
    Parameters
    ----------
    events: pandas.DataFrame
        действия студентов со степами
    submissions: pandas.DataFrame
        действия студентов по практике     
     """
    events_train = preprocess_timestamp_cols(events)
    events_train = truncate_data_by_nday(events_train, DATA_PERIOD_DAYS)

    submissions_train = preprocess_timestamp_cols(submissions)
    submissions_train = truncate_data_by_nday(submissions_train, DATA_PERIOD_DAYS)

    X = create_user_data(events_train, submissions_train)
    X = X.set_index('user_id')
    
#     safe_drop_cols_df(X, ['last_timestamp'])
    X = X = X.drop('last_timestamp', axis='columns')

    y = get_y(events, submissions)

    # после создания признаков и метки порядок следования user_id может не совпадать
    X = X.sort_index()
    y = y.sort_index()
    assert X.shape[0] == y.shape[0]
    
    return X, y


# In[9]:


def preprocess_timestamp_cols(data):
  
    """ 
    Parameters
    ----------
    data : pd.DataFrame
        данные с действиями пользователя
    """
    data['date'] = pd.to_datetime(data.timestamp, unit='s')
    data['day'] = data.date.dt.date
    return data


# In[10]:


def truncate_data_by_nday(data, n_day):
  
    """ Взять события из n_day первых дней по каждому пользователю 
        Parameters
        ----------
        data: pandas.DataFrame
            действия студентов со степами или практикой
        n_day : int
            размер тестовой выборки
    """
    
    users_min_time = data.groupby('user_id', as_index=False).agg({'timestamp': 'min'}).rename(
        {'timestamp': 'min_timestamp'}, axis=1)
    users_min_time['min_timestamp'] += 60 * 60 * 24 * n_day

    events_data_d = pd.merge(data, users_min_time, how='inner', on='user_id')
    cond = events_data_d['timestamp'] <= events_data_d['min_timestamp']
    events_data_d = events_data_d[cond]

    assert events_data_d.user_id.nunique() == data.user_id.nunique()
    
    return events_data_d.drop(['min_timestamp'], axis=1)


# In[11]:


def create_user_data(events, submissions):
  
    """ создать таблицу с данными по каждому пользователю
    Parameters
    ----------
    events : pd.DataFrame
        данные с действиями пользователя
    submissions : pd.DataFrame
        данные самбитов практики
    """
    users_data = events.groupby('user_id', as_index=False)         .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})

    # попытки сдачи практики пользователя
    users_scores = submissions.pivot_table(index='user_id',
                                           columns='submission_status',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0).reset_index()
    users_data = users_data.merge(users_scores, on='user_id', how='outer')
    users_data = users_data.fillna(0)

    # колво разных событий пользователя по урокам
    users_events_data = events.pivot_table(index='user_id',
                                           columns='action',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0).reset_index()
    users_data = users_data.merge(users_events_data, how='outer')

    # колво дней на курсе
    users_days = events.groupby('user_id').day.nunique().to_frame().reset_index()
    users_data = users_data.merge(users_days, how='outer')

    return users_data


# In[12]:


def create_interaction(events, submissions):
  
    """ объединить все данные по взаимодействию
    Parameters
    ----------
    events : pd.DataFrame
        данные с действиями пользователя
    submissions : pd.DataFrame
        данные самбитов практики
    """
    interact_train = pd.concat([events, submissions.rename(columns={'submission_status': 'action'})])
    interact_train.action = pd.Categorical(interact_train.action,
                                           ['discovered', 'viewed', 'started_attempt',
                                            'wrong', 'passed', 'correct'], ordered=True)
    interact_train = interact_train.sort_values(['user_id', 'timestamp', 'action'])
    
    return interact_train


# In[13]:


def get_y(events, submissions, course_threshold=40, target_action='passed'):
  
    """ создать метку  (целевая переменная для прогноза is_gone
    Parameters
    ----------
    events : pd.DataFrame
        данные с действиями пользователя
    submissions : pd.DataFrame
        данные самбитов практики
    course_threshold : int
        порог в колве заданий, когда курс считается пройденным
    target_action: string
        название действия по степу, по колву которых мы рассчитываем целевую переменную 
    """
    interactions = create_interaction(events, submissions)
    users_data = interactions[['user_id']].drop_duplicates()

    assert target_action in interactions.action.unique()
    passed_steps = (interactions.query("action == @target_action")
                    .groupby('user_id', as_index=False)['step_id'].count()
                    .rename(columns={'step_id': target_action}))
    users_data = users_data.merge(passed_steps, how='outer')

    # пройден ли курс
    users_data['is_gone'] = users_data[target_action] > course_threshold
    assert users_data.user_id.nunique() == events.user_id.nunique()
    users_data = (users_data.drop(target_action, axis=1)
                  .set_index('user_id'))
    
    return users_data['is_gone']


# In[14]:


def split_events_submissions(events, submissions, test_size=0.3):
  
    """ разделение выборки на трейн и тест по пользователям
        Parameters
        ----------
        events: pandas.DataFrame
            действия студентов со степами
        submissions: pandas.DataFrame
            действия студентов по практике     
        test_size : float
            размер тестовой выборки
     """

    # сделаем случайную выборку пользователей курса для теста
    users_ids = np.unique(np.concatenate((events.user_id.unique(), submissions.user_id.unique())))
    np.random.shuffle(users_ids)
    test_sz = int(len(users_ids) * test_size)
    train_sz = len(users_ids) - test_sz
    train_users = users_ids[:train_sz]
    test_users = users_ids[-test_sz:]
    # Проверка что пользователи не пересекаются
    assert len(np.intersect1d(train_users, test_users)) == 0

    # теперь делим данные
    event_train = events[events.user_id.isin(train_users)]
    event_test = events[events.user_id.isin(test_users)]
    submissions_train = submissions[submissions.user_id.isin(train_users)]
    submissions_test = submissions[submissions.user_id.isin(test_users)]

    return event_train, event_test, submissions_train, submissions_test


# In[15]:


def create_dataset(events_data, submissions_data):

  DEBUG = True
  
  if DEBUG:
    print('events_data: ', events_data.shape)
    print('submissions_data: ', submissions_data.shape)  

  user_learning_time_treshold = 2 * 24 * 60 * 60
  
  events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
  events_data['day'] = events_data.date.dt.date

  submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
  submissions_data['day'] = submissions_data.date.dt.date
  
  users_course_result = events_data[events_data.action == 'passed'][['step_id', 'user_id']]                              .drop_duplicates()                             .groupby(['user_id'])                             .agg({'step_id' : 'count'})                             .rename({'step_id' : 'steps_passed'}, axis='columns')                             .reset_index()
  
  if DEBUG: print('users_course_result: ', users_course_result.shape)
  
#   users_course_result = users_course_result.sort_values('steps_passed', ascending=False)
  users_course_result['passed_course'] = users_course_result.steps_passed > 40
  users_course_result['passed_course'] = users_course_result['passed_course'].map(int)
  users_course_result = users_course_result.drop('steps_passed', axis='columns')
  
  user_start_time = events_data.groupby('user_id', as_index=False)         .agg({'timestamp': 'min'})         .rename({'timestamp': 'start_course_timestamp'}, axis=1)    

  if DEBUG: print('user_start_time: ', user_start_time.shape)
    
  user_last_visit_time = events_data.groupby('user_id', as_index=False)         .agg({'timestamp': 'max'})         .rename({'timestamp': 'last_visit_time'}, axis=1)    
  
  if DEBUG: print('user_last_visit_time: ', user_last_visit_time.shape)
  
  events_data = events_data.merge(user_start_time, on='user_id', how='outer')
  events_data = events_data[events_data.timestamp <= events_data.start_course_timestamp + user_learning_time_treshold]
  events_data.drop('start_course_timestamp', axis=1)
    
  if DEBUG: print('events_data after triming: ', user_start_time.shape)
 
  unique_days_in_events_data = events_data.groupby('user_id').day.nunique().max()
  
  if DEBUG: print('unique_days_in_events_data: ', unique_days_in_events_data)
  
  submissions_data = submissions_data.merge(user_start_time, on='user_id', how='outer')
  submissions_data = submissions_data[submissions_data.timestamp <= submissions_data.start_course_timestamp + user_learning_time_treshold]
  submissions_data.drop('start_course_timestamp', axis=1)  
      
  if DEBUG: print('submissions_data after triming: ', user_start_time.shape)
 
  unique_days_in_submissions_data = submissions_data.groupby('user_id').day.nunique().max()
  
  if DEBUG: print('unique_days_in_submissions_data: ', unique_days_in_submissions_data)
  
  users_scores = submissions_data.pivot_table(index='user_id',
                          columns='submission_status',
                          values='step_id',
                          aggfunc='count',
                          fill_value=0).reset_index()

  if DEBUG: print('users_scores: ', users_scores.shape)

  steps_tried = submissions_data.groupby('user_id').step_id         .nunique().to_frame().reset_index().rename(columns={
            'step_id': 'steps_tried'})
  
  if DEBUG: print('steps_tried: ', steps_tried.shape)

  users_actions = events_data.pivot_table(index='user_id',
                            columns='action',
                            values='step_id',
                            aggfunc='count',
                            fill_value=0).reset_index()  
  
  if DEBUG: print('users_actions: ', users_actions.shape)

  users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index()
  
  if DEBUG: print('users_days: ', users_days.shape)
    
  X = users_days
    
#   X = X.merge(user_start_time, on='user_id', how='outer')
  
#   X = X.merge(user_last_visit_time, on='user_id', how='outer')  

  X = X.merge(users_actions, on='user_id', how='outer')  

  X = X.merge(users_scores, on='user_id', how='outer')  

#   X = X.merge(steps_tried, on='user_id', how='outer')

  X = X.merge(users_course_result, on='user_id', how='outer')  

  #X['correct_ratio'] = np.where(X.correct + X.wrong > 0, X.correct / (X.correct + X.wrong), 0)

  #X['passed_by_day'] = X.passed / X.day

  #X['discovered_by_day'] = X.discovered / X.day

#   X['passed_course'] = np.where(X.passed > 40, 1, 0)

  X = X.fillna(0)

  X = X.set_index('user_id')

  y = X.passed_course
  X = X.drop('passed_course', axis='columns')

  if DEBUG:
    print('X: ', X.shape)
    print('y: ', y.shape)

  return X, y


# In[16]:


events_data_train = pd.read_csv('https://stepik.org/media/attachments/course/4852/event_data_train.zip', compression='zip')
# events_data_train = pd.read_csv('event_data_train.zip', compression='zip')
events_data_train.shape

print('events_data_train shape', events_data_train.shape)

submissions_data_train = pd.read_csv('https://stepik.org/media/attachments/course/4852/submissions_data_train.zip', compression='zip')
# submissions_data_train = pd.read_csv('submissions_data_train.zip', compression='zip')
submissions_data_train.shape

print('submissions_data_train shape', submissions_data_train.shape)


# In[17]:


control_events_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/events_data_test.csv')
control_events_data.shape

print('control_events_data shape', control_events_data.shape)

control_submissions_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/submission_data_test.csv')
control_submissions_data.shape

print('control_submissions_data shape', control_submissions_data.shape)


# In[18]:


X, y = create_dataset(events_data_train, submissions_data_train)


# In[19]:


X.head()


# In[20]:


y.value_counts()


# In[21]:


eug_X, eug_y = get_x_y(events_data_train, submissions_data_train)


# In[22]:


eug_X.head()


# In[23]:


eug_y.value_counts()


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.61)


# In[25]:


event_train, event_test, submissions_train, submissions_test = split_events_submissions(events_data_train, submissions_data_train)

X_train, y_train = create_dataset(event_train, submissions_train)
X_test, y_test   = create_dataset(event_test, submissions_test)


# In[26]:


X_train.shape


# In[27]:


X_train.head()


# In[28]:


y_train.value_counts()


# In[29]:


control_X, control_y = create_dataset(control_events_data, control_submissions_data)


# In[30]:


print(pd.cut(control_X.passed[control_X.passed > 20], 10).value_counts())
# print(control_X.passed.value_counts())

control_X[(control_X.passed > 20) & (control_X.passed < 40)].passed.hist(bins  = 20)


# In[31]:


control_y.value_counts()


# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics

from sklearn.linear_model import LogisticRegression

#import xgboost as xgb


# In[34]:


estimator = RandomForestClassifier()

#params = {'n_estimators':range(1, 101), 'max_depth' : range(1, 20), 'min_samples_leaf' :[10], 'min_samples_split' : range(20, 100, 10), 'class_weight':['balanced','balanced_subsample']}
# params = {'n_estimators':range(1, 101), 'min_samples_leaf' :[10], 'min_samples_split' : range(20, 50, 10), 'class_weight':['balanced','balanced_subsample']}

params = {'n_estimators':[100], 'min_samples_leaf' :[10], 'min_samples_split' : [10], 'class_weight':['balanced','balanced_subsample']}

# estimator = xgb.XGBClassifier()
# params = {'n_estimators' : range(1, 51), 'learning_rate': np.arange(0.1, 0.6, 0.1), 'max_depth' : range(1,11) , 'min_child_weight': range(2, 11)}
# params = {'n_estimators' :[45] , 'learning_rate': [0.3], 'max_depth' : range(1,11) , 'min_child_weight': range(2, 11)}

searcher = GridSearchCV(estimator = estimator, param_grid=params, cv=3, scoring='roc_auc')

get_ipython().magic(u'time searcher.fit(X_train, y_train)')

best_estimator =  searcher.best_estimator_

scorer = searcher.scorer_
print(scorer)

best_params = searcher.best_params_
print(best_params)

# best_estimator = RandomForestClassifier(n_estimators=100, n_jobs=2, 
#                             min_samples_leaf=10, min_samples_split=10, 
#                             class_weight='balanced')

# best_estimator.fit(X_train, y_train)


# In[35]:


fimp = pd.DataFrame([best_estimator.feature_importances_], columns=X_train.columns).T
fimp.columns = ['weight']
fimp.sort_values(by='weight', ascending=False)


# In[36]:


def model_assessment(estimator, X, y, set_label):
  
  print(set_label, ' score:', estimator.score(X, y))
  print(set_label, ' roc_auc:', metrics.roc_auc_score(y, estimator.predict(X)))

  cv_scores = cross_val_score(estimator, X, y, scoring='roc_auc', cv=3, n_jobs=-1)
  mean_cv_scores = np.mean(cv_scores)
  print (set_label, ' cross_val mean score:', mean_cv_scores)
  print('\n')
  
  predict_proba = estimator.predict_proba(X)
  print(pd.cut(predict_proba[:,1], 10).value_counts())
  print('\n')

  pd.Series(predict_proba[:,1]).hist()

  fpr, tpr, thresholds = metrics.roc_curve(y, predict_proba[:,1])
  roc_auc = metrics.auc(fpr, tpr)
  print(roc_auc)
  print('\n')

  plt.figure()
  plt.plot(fpr, tpr, label = 'Area = %0.2f' % (roc_auc))
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0, 1])
  plt.ylim([0, 1.05])

  predict_y = estimator.predict(X)
  print(metrics.confusion_matrix(y, predict_y))
  print('\n')  


# In[37]:


prdict_proba_train = best_estimator.predict_proba(X_train)
print(prdict_proba_train)

prdict_proba_train[np.where(y_train == 1), 1] = 1
print(prdict_proba_train)

fpr, tpr, thresholds = metrics.roc_curve(y_train, prdict_proba_train[:,1])
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)
print('\n')

metrics.roc_auc_score(y_train, best_estimator.predict(X_train))


# In[38]:


model_assessment(best_estimator, X_train, y_train, 'X_train')


# In[40]:


X_train.shape


# In[41]:


model_assessment(best_estimator, X_test, y_test, 'X_test')


# In[42]:


X_test.shape


# In[43]:


model_assessment(best_estimator, control_X, control_y, 'control_X')


# In[44]:


control_X.shape


# In[46]:


result_X = control_X.copy()
result_y = control_y.copy()

result_predict_proba = best_estimator.predict_proba(result_X)
# result_predict_proba[np.where(result_y == 1), 1] = 1


# In[47]:


result_X['is_gone'] = pd.Series(result_predict_proba[:,1], index=result_X.index)
             
#result_X.loc[:,is_gone'] = pd.Series(result_predict_proba[:,1], index=result_X.index)

# result_X['is_gone'] = np.where(result_X['passed'] >= 40, 1, result_X['passed'] / 40)
# result_X['is_gone'] = np.where(result_X['passed'] < 13, 0, result_X['is_gone'])


# In[48]:


result_X[result_X.is_gone > 0]


# In[49]:


pd.cut(result_X.is_gone, 10).value_counts()


# In[50]:


result_X.is_gone.hist()


# In[51]:


user_success_probability = result_X[['is_gone']]


# In[52]:


user_success_probability.shape


# In[53]:


user_success_probability.head()


# In[55]:


user_success_probability.to_csv('C:\Users\Asus\Desktop\STEPIC\\user_success_probability.csv')


# In[ ]:




