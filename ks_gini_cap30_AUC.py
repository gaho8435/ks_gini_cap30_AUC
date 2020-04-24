#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from keras.utils import to_categorical
from sklearn import ensemble, preprocessing, metrics, model_selection
from sklearn.metrics import roc_auc_score


class ks_gini_cap30_AUC():
    def __init__(self, classes, model_predict_proba, y, do_cate=False):
        tStart = time.time()
        self.classes = classes #類別數,包含0
        self.model_predict_proba = model_predict_proba #預測機率
        self.y = y #真實y
        
        ############################計算ROC&AUC############################
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if do_cate == True:
            train_y = to_categorical(self.y)
        else:
            train_y = self.y
        for i in range(self.classes):
            fpr[i],tpr[i],_ = metrics.roc_curve(train_y[:,i],self.model_predict_proba[:,i])
            roc_auc[i] = metrics.auc(fpr[i],tpr[i])
        
        #micro-average ROC
        fpr['micro'],tpr['micro'],_ = metrics.roc_curve(train_y.ravel(),self.model_predict_proba.ravel())
        roc_auc['micro'] = metrics.auc(fpr['micro'],tpr['micro'])
        
        #macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.classes):
            mean_tpr += np.interp(all_fpr,fpr[i],tpr[i])
        mean_tpr /= self.classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = metrics.auc(fpr['macro'],tpr['macro'])
        
        ############################計算ROC&AUC############################
        self.fpr = fpr #false positive rate
        self.tpr = tpr #true positive rate
        self.roc_auc = roc_auc # AUC
        tEnd = time.time()
        print('Complete! Cost ' + str(round(tEnd - tStart,2)) + 's')
        
        
    def calculate_auc(self,num = 1):
        if num <= self.classes: #Y處理
            auc = self.roc_auc[num]
        else:
            print('num is wrong!')
        return auc
    
    def calculate_cap30(self,num = 1):
        output = []
        predict = copy.deepcopy(self.model_predict_proba)
        columns_prob = []
        for i in range(self.classes):
            columns_prob.append('prob_' + str(i))
        df = pd.merge(pd.DataFrame(list(self.y),columns = ['y']), 
                      pd.DataFrame(predict,columns = columns_prob),
                      left_index = True, right_index = True)
        if num <= self.classes: #Y處理
            df.loc[df.y == num, 'y'] = 1
            df.loc[df.y != num, 'y'] = 0
        else:
            print('num is wrong!')
        df_ = df.sort_values(by = columns_prob[num],ascending = False)
        cap30 = sum(df_['y'][:int(len(df_['y'])*3/10)])/sum(df['y'])
        return cap30
    
    def calculate_ks(self,num = 1):
        output = []
        predict = copy.deepcopy(self.model_predict_proba)
        columns_prob = []
        for i in range(self.classes):
            columns_prob.append('prob_' + str(i))
        df = pd.merge(pd.DataFrame(list(self.y),columns = ['y']), 
                      pd.DataFrame(predict,columns = columns_prob),
                      left_index = True, right_index = True)
        if num <= self.classes: #Y處理
            df.loc[df.y == num, 'y'] = 1
            df.loc[df.y != num, 'y'] = 0
        else:
            print('num is wrong!')
        output = []
        for i in range(1,11): #排序並計算
            output.append(abs(round(len(df)*i/10)/len(df) - \
                          sum(df.sort_values(by = columns_prob[num],ascending=False)[:round(len(df)*i/10)]['y'])/sum(df['y'])))
        ks = max(output)
        return ks
    
    def calculate_gini(self,num = 1):
        output = []
        predict = copy.deepcopy(self.model_predict_proba)
        columns_prob = []
        for i in range(self.classes):
            columns_prob.append('prob_' + str(i))
        df = pd.merge(pd.DataFrame(list(self.y),columns = ['y']), 
                      pd.DataFrame(predict,columns = columns_prob),
                      left_index = True, right_index = True)
        if num <= self.classes: #Y處理
            df.loc[df.y == num, 'y'] = 1
            df.loc[df.y != num, 'y'] = 0
        else:
            print('num is wrong!')
        gini_list = []
        for i in range(1,11): #計算Gini
            if i == 1:
                gini_list.append((sum(df.sort_values(by = columns_prob[num],ascending=False)[:round(np.shape(df)[0]*i/10)]['y'])/ \
                                        sum(df['y']))*(round(len(df)*i/10)/len(df)))
            else:
                gini_list.append(((sum(df.sort_values(by = columns_prob[num],ascending=False)[:round(np.shape(df)[0]*i/10)]['y']) - \
                                     sum(df.sort_values(by = columns_prob[num],ascending=False)[:round(np.shape(df)[0]*(i-1)/10)]['y']))/sum(df['y']))* \
                                    ((round(len(df)*i/10)-round(len(df)*(i-1)/10))/len(df)))
        gini = 1-sum(gini_list)
        return gini
    
    def calculate_all(self, num = 1):
        result = []
        result.append([self.calculate_ks(num = num),self.calculate_gini(num = num),
                       self.calculate_cap30(num = num),self.calculate_auc(num = num)])
        df = pd.DataFrame(result, columns = ['ks','gini','cap30','auc'])
        return df

############################計算num類別預測結果,以機率排序切成十等分############################
#########################################分類每等分細節#########################################
    def calculate_detail(self, num = 1):
        tStart = time.time()
        output = []
        predict = copy.deepcopy(self.model_predict_proba)
        columns_prob = []
        for i in range(self.classes):
            columns_prob.append('prob_' + str(i))
        df = pd.merge(pd.DataFrame(list(self.y),columns = ['y']), 
                      pd.DataFrame(predict,columns = columns_prob),
                      left_index = True, right_index = True)
        if num <= self.classes: #Y處理
            df.loc[df.y == num, 'y'] = 1
            df.loc[df.y != num, 'y'] = 0
        else:
            print('num is wrong!')
        for i in range(1,11): #排序並計算
            df_ = df.sort_values(by = columns_prob[num],ascending = False)[round(np.shape(df)[0]*(i-1)/10):round(np.shape(df)[0]*i/10)]
            output.append([i, #rank
                           round(np.shape(df)[0]*i/10)-round(np.shape(df)[0]*(i-1)/10), #人數
                           round(np.shape(df)[0]*i/10), #累積人數
                           sum(df_['y']), #y數量
                           sum(df.sort_values(by = columns_prob[num],ascending=False)[:round(np.shape(df)[0]*i/10)]['y']), #累積y數量
                           sum(df_['y'])/(round(np.shape(df)[0]*i/10)-round(np.shape(df)[0]*(i-1)/10)), #y率
                           abs(round(np.shape(df)[0]*i/10)/np.shape(df)[0] - sum(df.sort_values(by = columns_prob[num],ascending=False)[:round(np.shape(df)[0]*i/10)]['y'])/sum(df['y'])),
                           0.]) #預留Gini位置
        df_output = pd.DataFrame(output,columns = ['rank','人數','累積人數','y','累積y','y率','KS','Gini'])
        for i in range(10): #計算Gini
            if i == 0:
                df_output['Gini'][i] = (df_output['累積y'][i]/df_output['累積y'][9])*(df_output['累積人數'][i]/df_output['累積人數'][9])
            else:
                df_output['Gini'][i] = ((df_output['累積y'][i]-df_output['累積y'][i-1])/df_output['累積y'][9])*((df_output['累積人數'][i]-df_output['累積人數'][i-1])/df_output['累積人數'][9])
        
        tEnd = time.time()
        print('Cost ' + str(round(tEnd - tStart,2)) + 's')
        return df_output
    
    
############################計算num類別預測結果,以機率排序切成十等分############################
#####################################只Output出最後衡量數值#####################################
    def calculate_result(self,num = 1,do_all = False):
        tStart = time.time()
        predict = copy.deepcopy(self.model_predict_proba)
        columns_prob = []
        for i in range(self.classes):
            columns_prob.append('prob_' + str(i))
        if do_all == False:
            df_output = self.calculate_detail(num)
            output_1 = []
            cap30 = sum(df_output['y'][:3])/df_output['累積y'][9]
            output_1.append([max(df_output['KS']),
                             1 - sum(df_output['Gini']),
                             cap30,
                             self.roc_auc[num]])
            df_output_1 = pd.DataFrame(output_1,columns = ['KS','Gini','cap30','AUC'])
            df_output_1 = df_output_1.set_index(pd.Index([num]))
        elif do_all == True:
            output_1 = []
            for j in range(1,self.classes):
                df_output = self.calculate_detail(j)
                cap30 = sum(df_output['y'][:3])/df_output['累積y'][9]
                output_1.append([max(df_output['KS']),
                                 1 - sum(df_output['Gini']),
                                 cap30,
                                 self.roc_auc[j]])
            #計算平均
            output_1 = np.array(output_1).T.tolist()
            for i in range(np.shape(output_1)[0]):
                output_1[i].append(np.average(df_output[i]))
            output_1 = np.array(output_1).T.tolist()
            df_output_1 = pd.DataFrame(output_1,columns = ['KS','Gini','cap30','AUC'])
            index = []
            for i in range(1,self.classes):
                index.append(i)
            index.append('Average')
            df_output_1 = df_output_1.set_index(pd.Index(index))
        else:
            print('do all? Enter True or False.')
        
        tEnd = time.time()
        print('Cost ' + str(round(tEnd - tStart,2)) + 's')
        return df_output_1
    

#############################劃出各類別的ROC CURVE及AUC#############################
    def ROC_AUC_plot(self, Title = '', fontsize = 12):
        tStart = time.time()
        plt.figure(figsize = (10,8))
        plt.style.use('seaborn')
        plt.plot(self.fpr['macro'], self.tpr['macro'],
                 label = 'macro-avg ROC curve(AUC={0:0.2f})'.format(self.roc_auc['macro']),
                 color = 'navy', linestyle = ':', linewidth = 4)
        for i in range(len(self.roc_auc)-2):
            plt.plot(self.fpr[i],self.tpr[i],
                     label = 'ROC curve of class{0}(AUC={1:0.2f})'.format(i, self.roc_auc[i]))
        plt.plot([0,1],[0,1],'k--')
        plt.legend(loc = 'lower right', fontsize = fontsize)
        plt.title(Title, fontsize = fontsize)
        plt.xlabel('False Positive Rate', fontsize = fontsize)
        plt.ylabel('True Positive Rate', fontsize = fontsize)
        plt.tick_params(axis = 'x', labelsize = fontsize)
        plt.tick_params(axis = 'y', labelsize = fontsize)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        tEnd = time.time()
        print('Cost ' + str(round(tEnd - tStart,2)) + 's')
        plt.show()
    