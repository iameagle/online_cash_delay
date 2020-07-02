#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 忽略警告
import warnings
warnings.filterwarnings('ignore')
# 单个单元格多个输出
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

import lightgbm as lgb
import xgboost as xgb
import catboost as cab
from  sklearn.metrics import *


# In[2]:


#节约内存 
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


# 读取数据
original_data = pd.read_csv('data_with_all_feature.csv')
original_data.head()


# In[4]:


# 缺失率统计
null_count_df = original_data.isnull().sum().to_frame()
null_count_df['var'] = null_count_df.index
null_count_df.reset_index(drop=True,inplace=True)
null_count_df.rename(columns={0:'null_counts'},inplace=True)
null_count_df['null_rates'] = null_count_df['null_counts'] / len(original_data)
null_count_df.head().append(null_count_df.tail())


# In[5]:


# 过滤掉覆盖率小于0.01的变量
# len(null_count_df[null_count_df['null_rates']>=0.99])

filter_var = null_count_df[null_count_df['null_rates']<=0.99]['var'].tolist()
# filter_var

filter_data = original_data[filter_var]
filter_data.drop(['mobile_md5','id_number_md5','name_md5','name','mobile'],axis=1,inplace=True)
filter_data.head()


# In[6]:


# 删除倾斜特征
# 目的：判断是否存在某个特征的某个值占比超过99%,一般会删除该特征
def value_counts_sta(df):
    columns = df.columns.tolist()
    lean_cols = []
    for col in columns:
        df_col = df[col].value_counts().to_frame()
        df_col[col+'_true'] = df_col.index
        df_col = df_col.reset_index(drop=True)
        df_col.rename(columns={col:'counts'},inplace=True)
        top1_value = df_col['counts'].head(1).tolist()[0]
        if top1_value/(df_col['counts'].sum()) >= 0.99:
            lean_cols.append(col)
    return lean_cols

filter_data.drop(value_counts_sta(filter_data),axis=1,inplace=True)
filter_data.shape


# In[7]:


filter_data.info()
data_bin = filter_data.copy()


# In[8]:


# iv值过滤 大于0.03
from tqdm import tqdm
dic_bin = pd.DataFrame()
# 数值变量
num_cols = filter_data.select_dtypes(exclude=['category','object']).columns.tolist()
num_cols.remove('y_label')

for var in tqdm(num_cols):
    s_col = [var] + ['y_label']
    temp_df = data_bin[s_col]
    temp_df[var] = pd.qcut(temp_df[var],5,duplicates='drop')
    temp_df[var] = temp_df[var].replace(np.NaN,'missing')
    bin_group  = temp_df.groupby(var)['y_label'].agg([('bad_ratio','mean'),('count','count'),('bad_num','sum')]).reset_index()
    bin_group['good_num'] = bin_group['count']-bin_group['bad_num']
    target0_num=bin_group['good_num'].sum()
    target1_num=bin_group['bad_num'].sum()
    bin_group['good_perc']=(bin_group['good_num']+1)/(target0_num+1)
    bin_group['bad_perc']=(bin_group['bad_num']+1)/(target1_num+1)
    bin_group['woe'] = np.log((bin_group['bad_perc'])/(bin_group['good_perc']))
    bin_group['iv'] = (bin_group['bad_perc']-bin_group['good_perc'])*bin_group['woe']
    bin_group['total_iv']=bin_group['iv'].sum()
    bin_group['var_name']=var
    bin_group.columns=['bin','bad_ratio','count','bad_num','good_num','good_perc','bad_perc','woe','iv','total_iv','var_name']
#         data_woe[var]=data_bin[var].map(dict(zip(bin_group.bin,bin_group.woe)))
    dic_bin=pd.concat([dic_bin,bin_group])

dic_bin.to_csv('var_iv.csv',index=0,encoding='gbk')


# In[9]:


# 选择iv大于0.02 小于0.5的变量以及类别变量 
iv_var = set(dic_bin[(dic_bin['total_iv']>=0.02) & (dic_bin['total_iv']<=0.5)]['var_name'])
select_var = filter_data.select_dtypes(include=['category','object']).columns.tolist() + list(iv_var) + ['y_label']
# 删除关联key信息
drop_list = [
 'gaode_手机号/设备号',
 'haoduoshu_name',
 'haoduoshu_姓名',
 'liandong_编号',
 'liandong_加密手机号',
 'liandong_加密ID',
 'liandong_加密姓名',
 'qianweidu_id',
 'tenxun_phonenum_md5',
 'tenxun_id_card_md5',
 'tenxun_name',
 'tenxun_general_tag']

filter_data = filter_data[select_var].drop(drop_list,axis=1)
filter_data.to_csv('filter_data.csv',index=0,encoding='gbk')
filter_data.head()


# # 数据分析EDA

# In[10]:


# 缺失率统计
null_count_df = filter_data.isnull().sum().to_frame()
null_count_df['var'] = null_count_df.index
null_count_df.reset_index(drop=True,inplace=True)
null_count_df.rename(columns={0:'null_counts'},inplace=True)
null_count_df['null_rates'] = null_count_df['null_counts'] / len(original_data)
null_count_df.to_csv('null_counts.csv',index=False,encoding='gbk')


# In[11]:


# 类别特征
cat_features = filter_data.select_dtypes(include=['object']).columns.tolist()
for col in cat_features:
    print('类别特征的值个数:',(col,filter_data[col].nunique()))
    
# 数值特征
num_features = filter_data.select_dtypes(exclude=['object']).columns.tolist()
len(num_features)


# In[12]:


# 类别特征数据观察
# haoduoshu_手机可信等级 分布情况 做one hot编码
f, [ax1,ax2] = plt.subplots(2, 1, figsize=(15, 15))
sns.countplot(x = 'haoduoshu_手机可信等级' ,data = filter_data,ax = ax1)
sns.countplot(x = 'haoduoshu_手机可信等级', hue = 'y_label',hue_order = [0, 1],data = filter_data , ax = ax2)

# 其他为日期，单独一个日期date意义不大，建议删除，

# liandong_t_drcard_trans_tim_first liandong_t_drcard_trans_tim_curr日期做差 后者-前者
# liandong_t_crdt_trans_tim_first  liandong_t_crdt_trans_tim_curr日期做差

# 身份证可以提取年龄，地区等信息


# In[13]:


# 数值特征数据观察 已经计算了iv值，筛选出的具有预测能力的变量不需要做更多关注。
# 观察变量与目标值的相关性

# filter_data[num_features].corr()


# # 特征工程

# In[14]:


# id_number 提取年龄，地区
filter_data['age'] = filter_data['id_number'].map(lambda x: 2020- int(str(x)[6:10]))
filter_data['cus_area'] = filter_data['id_number'].map(lambda x: str(x)[:6])

filter_data.drop(['id_number'],axis=1,inplace=True)
filter_data.shape


# In[15]:


# 日期特征，计算时间差
filter_data['liandong_t_drcard_trans_tim_used_days'] = (pd.to_datetime(filter_data['liandong_t_drcard_trans_tim_curr'],format='%Y-%m-%d', errors='coerce') - 
                            pd.to_datetime(filter_data['liandong_t_drcard_trans_tim_first'], format='%Y-%m-%d', errors='coerce')).dt.days
filter_data['liandong_t_crdt_trans_tim_used_days'] = (pd.to_datetime(filter_data['liandong_t_crdt_trans_tim_curr'], format='%Y-%m-%d', errors='coerce') - 
                            pd.to_datetime(filter_data['liandong_t_crdt_trans_tim_first'], format='%Y-%m-%d', errors='coerce')).dt.days

filter_data[['liandong_t_drcard_trans_tim_used_days','liandong_t_crdt_trans_tim_used_days']].head()

# filter_data.drop(['date','liandong_t_drcard_trans_tim_first',
#  'liandong_t_drcard_trans_tim_curr',
#  'liandong_t_crdt_trans_tim_first',
#  'liandong_t_crdt_trans_tim_curr'],axis=1,inplace=True)
filter_data.shape


# In[16]:


from sklearn.model_selection import StratifiedKFold,KFold
from itertools import product
class MeanEncoder:
    def __init__(self, categorical_features, n_splits=10, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode
 
        :param n_splits: the number of splits used in mean encoding
 
        :param target_type: str, 'regression' or 'classification'
 
        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """
 
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}
 
        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None
 
        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))
 
    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()
 
        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()
 
        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)
 
        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values
 
        return nf_train, nf_test, prior, col_avg_y
 
    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)
 
        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new
 
    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
 
        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
 
        return X_new


# In[17]:


# 高基类别特征做平均数编码
MeanEnocodeFeature = ['cus_area','date','liandong_t_drcard_trans_tim_first',
 'liandong_t_drcard_trans_tim_curr',
 'liandong_t_crdt_trans_tim_first',
 'liandong_t_crdt_trans_tim_curr'] #声明需要平均数编码的特征
ME = MeanEncoder(MeanEnocodeFeature,target_type='classification') #声明平均数编码的类
X_data = filter_data.drop(['y_label'],axis=1)
Y_data = filter_data['y_label']

X_data = ME.fit_transform(X_data,Y_data) #对训练数据集的X和y进行拟合
#x_train_fav = ME.fit_transform(x_train,y_train_fav)#对训练数据集的X和y进行拟合
# filter_data['cus_area'].nunique() 3481
# X_data.drop(['cus_area'],axis=1,inplace=True)
X_data.shape


# In[18]:


# 中文表头转换
cnn_2_en_dic = {'haoduoshu_该手机号对应自然人在天猫上注册的ID个数':'haoduoshu_phone_tianmao_counts',
                'tenxun_疑似信贷恶意行为':'tenxun_yisieyixindai',
               'gaode_打分结果':'gaode_score',
               'tenxun_疑似资料仿冒行为':'tenxun_fake_info',
               'tenxun_疑似金融黑产相关':'tenxun_finance_black',
               'haoduoshu_手机可信等级':'haoduoshu_shoujikexindengji'}

X_data.rename(columns=cnn_2_en_dic,inplace=True)


# In[19]:


# 年龄分箱
age_bins = [20,30,40,50,60]
X_data['age'] = pd.cut(X_data['age'],age_bins)


# In[20]:


# 将原始编码特征drop
drop_col = ['date',
 'liandong_t_drcard_trans_tim_first',
 'liandong_t_drcard_trans_tim_curr',
 'liandong_t_crdt_trans_tim_first',
 'liandong_t_crdt_trans_tim_curr',
 'cus_area']

X_data.drop(drop_col,axis=1,inplace=True)


# In[22]:


# 数据one hot 编码
X_data = pd.get_dummies(X_data,columns=['haoduoshu_shoujikexindengji','age'])
X_data.info()


# In[23]:


# 数据缺失值填充，填充中位数 因为数据有偏移的情况
for col in X_data.columns:
    X_data[col].fillna(X_data[col].median(),inplace=True)


# In[28]:


# 年龄one hot后的变量重命名
re_dict = {'age_(20, 30]':'age_20_30',
 'age_(30, 40]':'age_30_40',
 'age_(40, 50]':'age_40_50',
 'age_(50, 60]':'age_50_60'}

X_data.rename(columns=re_dict,inplace=True)


# In[29]:


# 根据特征重要性排序挑选特征，使用null importance feature消除噪声
# 获取树模型的特征重要性
def get_feature_importances(data, shuffle, target, seed=None):
    
    # 特征
    train_features = [f for f in data if f not in [target]]
   
    y = data[target].copy()
    
    # 在造null importance时打乱
    if shuffle:
        # 为了打乱而不影响原始数据，采用了.copy().sample(frac=1.0)这种有点奇怪的做法
        y = data[target].copy().sample(frac=1.0)
    
    # 使用lgb的随机森林模式，据说会比sklearn的随机森林快点
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 30,
        'max_depth': 5,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 4
    }
    
    #训练
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    
    return imp_df

actual_imp = get_feature_importances(data=pd.concat([X_data,Y_data],axis=1),shuffle=False,target='y_label')
actual_imp.sort_values("importance_gain",ascending=False)


# In[31]:


# 计算null importance
null_imp_df = pd.DataFrame()
nb_runs = 10
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # 获取当前轮feature impotance
    imp_df = get_feature_importances(data=pd.concat([X_data,Y_data],axis=1), shuffle=True, target='y_label')
    imp_df['run'] = i + 1 
    # 加到合集上去
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # 擦除旧信息
    for l in range(len(dsp)):
        print('b', end='', flush=True)
    # 显示当前轮信息
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)


# In[32]:


# 计算原始特征重要性和null importance method的特征重要性的得分
# score = log((1+actual_importance)/(1+null_importance_75))
feature_scores = []
for _f in actual_imp['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp.loc[actual_imp['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp.loc[actual_imp['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    final_score = 0.4*split_score+0.6*gain_score
    feature_scores.append((_f, split_score, gain_score,final_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score','final_score'])
null_importance_score = 'null_importance_score.xlsx'
scores_df.to_excel(null_importance_score,index=False)


# In[34]:


# 过滤
filter_var = scores_df[scores_df['final_score']>0.1].feature.tolist()

X_data = X_data[filter_var]
X_data.shape


# In[35]:


# 数据做归一化
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X_data.values)
X_data = pd.DataFrame(min_max_scaler.transform(X_data.values),columns=X_data.columns)


# In[37]:


# 拆分数据集
from sklearn.model_selection import train_test_split
y = Y_data # target
X = X_data
# 80% 是 是线下数据
# 20% 是 是线上的数据
X_offset, X_onset, y_offset, y_onset = train_test_split(X, y ,test_size=0.2,random_state=2) 
X_offset.shape,X_onset.shape


# # LR CV

# In[38]:


from sklearn.linear_model import LogisticRegression,ElasticNet

clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)

# 线上预测的结果集
onset_predictions = np.zeros(len(X_onset))
# 线下验证的结果集
offset_predictions = np.zeros(len(X_offset))
#  线下cv auc的均值
mean_score = 0
sk = StratifiedKFold(n_splits=5,shuffle=True)
for fold,(train_idx,val_idx) in enumerate(sk.split(X_offset.values,y_offset.values)):
    print('fold {}:'.format(fold+1))
    trn_data = X_offset.iloc[train_idx]
    val_data = X_offset.iloc[val_idx]
    
    clf.fit(trn_data,y_offset.iloc[train_idx])
    #  线下每折的结果集    
    offset_predictions[val_idx] = clf.predict_proba(val_data.values)[:,1]
    mean_score += roc_auc_score(y_offset.iloc[val_idx],offset_predictions[val_idx])/sk.n_splits
    onset_predictions += clf.predict_proba(X_onset.values)[:,1]/sk.n_splits
    
print('lr 线下cv的平均auc:{:<8.5f}'.format(mean_score))
print('lr 线下结果集的auc:{:<8.5f}'.format(roc_auc_score(y_offset,offset_predictions)))
    
# 线上得分
print('lr 线上结果集的auc:{:<8.5f}'.format(roc_auc_score(y_onset,onset_predictions)))
fpr,tpr,thresholds=roc_curve(y_onset,onset_predictions)
print('lr 线上结果集的ks:{:<8.5f}'.format(max(tpr-fpr)))


# # LGB CV

# In[39]:


from sklearn.model_selection import StratifiedKFold
params = {
'boosting_type': 'gbdt',
'objective': 'binary',
'metric':  'auc',
'num_leaves': 30,
'max_depth': -1,
'min_data_in_leaf': 450,
'learning_rate': 0.01,
'feature_fraction': 0.9,
'bagging_fraction': 0.95,
'bagging_freq': 5,
'lambda_l1': 1,
'lambda_l2': 0.001,# 越小l2正则程度越高
'min_gain_to_split': 0.2,
#'device': 'gpu',
'is_unbalance': True
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=99999)
oof = np.zeros(len(X_offset))
predictions = np.zeros(len(X_onset))
mean_score=0
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_offset.values, y_offset.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(X_offset.iloc[trn_idx], label=y_offset.iloc[trn_idx])
    val_data = lgb.Dataset(X_offset.iloc[val_idx], label=y_offset.iloc[val_idx])
    clf = lgb.train(params, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(X_offset.iloc[val_idx], num_iteration=clf.best_iteration)
    mean_score += roc_auc_score(y_offset.iloc[val_idx], oof[val_idx]) / folds.n_splits
    predictions += clf.predict(X_onset, num_iteration=clf.best_iteration) / folds.n_splits

# 线下cv
print("CV score: {:<8.5f}".format(roc_auc_score(y_offset, oof)))
print("mean score: {:<8.5f}".format(mean_score))

# 线上得分
print("lgb online score: {:<8.5f}".format(roc_auc_score(y_onset, predictions)))
fpr,tpr,thresholds=roc_curve(y_onset,predictions)
print('lgb 线上结果集 ks:{:<8.5f}'.format(max(tpr-fpr)))


# # XGB CV

# In[40]:


params={
    'booster':'gbtree',
    'objective ':'binary:logistic',
    'gamma':0.1,                     # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth':6,                   # 构建树的深度 [1:]
    'subsample':0.8,                 # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_by_tree':0.8,          # 构建树时的采样比率 (0:1]
    'eta': 0.01,                     # 如同学习率
    'seed':555,                      # 随机种子
    'silent':1,
    'eval_metric':'auc',
    'n_job':-1,
#     'tree_method':'gpu_hist'
}

# 线上用于预测的矩阵 
test_data = xgb.DMatrix(X_onset,label=y_onset)
# 线上预测的结果集
onset_predictions = np.zeros(len(X_onset))
# 线下验证的结果集
offset_predictions = np.zeros(len(X_offset))
#  线下cv auc的均值
mean_score = 0
sk = StratifiedKFold(n_splits=5,shuffle=True)
for fold,(train_idx,val_idx) in enumerate(sk.split(X_offset.values,y_offset.values)):
    print('fold {}:'.format(fold))
    trn_data = xgb.DMatrix(X_offset.iloc[train_idx],label=y_offset.iloc[train_idx])
    val_data = xgb.DMatrix(X_offset.iloc[val_idx],label=y_offset.iloc[val_idx])
    
    clf = xgb.train(params=params,dtrain=trn_data,num_boost_round=10000,evals=[(trn_data,'train'),(val_data,'val')],
                    early_stopping_rounds=200,verbose_eval=100)
    #  线下每折的结果集    
    offset_predictions[val_idx] = clf.predict(val_data,ntree_limit=clf.best_iteration)
    mean_score += roc_auc_score(y_offset.iloc[val_idx],offset_predictions[val_idx])/sk.n_splits
    onset_predictions += clf.predict(test_data,ntree_limit=clf.best_iteration)/sk.n_splits
    
print('xgb 线下cv的平均auc:{:<8.5f}'.format(mean_score))
print('xgb 线下结果集的auc:{:<8.5f}'.format(roc_auc_score(y_offset,offset_predictions)))
    
# 线上得分
print('xgb 线上结果集的auc:{:<8.5f}'.format(roc_auc_score(y_onset,onset_predictions)))
fpr,tpr,thresholds=roc_curve(y_onset,onset_predictions)
print('xgb 线上结果集的ks:{:<8.5f}'.format(max(tpr-fpr)))


# # CAB CV

# In[41]:


clf = cab.CatBoostClassifier(iterations=100000, 
learning_rate=0.01, 
depth=5, loss_function='Logloss',early_stopping_rounds = 100,eval_metric='AUC')

# 线下预测结果集
offset_predictions = np.zeros(len(X_offset))
# 线上预测结果集
onset_predictions = np.zeros(len(X_onset))
# 线下cv auc平均得分
mean_score = 0
sk = StratifiedKFold(n_splits=5,shuffle=True)

for fold,(train_idx,val_idx) in enumerate(sk.split(X_offset.values,y_offset.values)):
    print('fold {}'.format(fold+1))
    train_data = X_offset.iloc[train_idx]
    val_data = X_offset.iloc[val_idx]
    
    clf.fit(train_data, y_offset.iloc[train_idx], eval_set=(val_data,y_offset.iloc[val_idx]),verbose= 50)
    offset_predictions[val_idx] = clf.predict_proba(val_data)[:,1]
    mean_score += roc_auc_score(y_offset.iloc[val_idx],offset_predictions[val_idx])/sk.n_splits
    onset_predictions += clf.predict_proba(X_onset)[:,1]/sk.n_splits

print('cab 线下cv的平均auc:{:<8.5f}'.format(mean_score))
print('cab 线下结果集的auc:{:<8.5f}'.format(roc_auc_score(y_offset,offset_predictions)))
    
# 线上得分
print('cab 线上结果集的auc:{:<8.5f}'.format(roc_auc_score(y_onset,onset_predictions)))
fpr,tpr,thresholds=roc_curve(y_onset,onset_predictions)
print('cab 线上结果集的ks:{:<8.5f}'.format(max(tpr-fpr)))

