#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import math


def rating_rmse(actual, predicted):
    '''
    Function to calculate the RMSE of actual rating matrix(R) vs predicted rating matrix(R^*)
    If the rating user u gives to item i is not known R(u,i)=0.
    
    actual = actual rating matrix of size (users x items)   
    predicted = predicted rating matrix of size (users x items)
    '''
    e=0
    n=0
    max_ac = actual.max()
    for u in range(actual.shape[0]):
        for i in range(actual.shape[1]):
            a =actual[u][i]
            if a!=0:
                n+=1
                if predicted[u][i]>max_ac:
                    e += (a-max_ac)**2
                else:
                    e += (a-predicted[u][i])**2
    rmse = math.sqrt(e/n)
    return rmse


def rating_mae(actual, predicted):
    '''
    Function to calculate the MAE of actual rating matrix(R) vs predicted rating matrix(R^*)
    If the rating user u gives to item i is not known R(u,i)=0.
    
    actual = actual rating matrix of size (users x items)   
    predicted = predicted rating matrix of size (users x items)
    '''
    e=0
    n=0
    max_ac= actual.max()
    for u in range(actual.shape[0]):
        for i in range(actual.shape[1]):
            a =actual[u][i]
            if a!=0:
                n+=1
                if predicted[u][i]>max_ac:
                    e += abs(a-max_ac)
                else:
                    e += abs(a-predicted[u][i])
    mae = e/n
    return mae

def accuracy_in_recommendations(correct, new):
    '''
    Function to calculate the accuracy of the optimal TopN list vs the re-ranked TopN list
    
    actual = optimal TopN list matrix of size (users x N)   
    predicted = re-ranked TopN matrix of size (users x N)
    '''
    u = len(correct)
    return sum([len(set(correct[i]) & set(new[i]))/len(correct[i]) for i in range(u)])/u


def get_average_profit(rec_list, prices):
    '''
    Function to calculate the average profit of the TopN list and the list of profit of every user
    
    rec_list = TopN list matrix of size (users x N)   
    prices = prices vector of size (users)
    '''
    prof_list = []
    for u_list in rec_list:
        profit = 0
        for item in u_list:
            profit += prices[item]
        prof_list.append(profit)
    return sum(prof_list)/len(prof_list), prof_list


def get_NDCG(Rp,ideal,ranked):
    '''
    Function to calculate the NDCG and DCG of the re-ranked TopN list taking into account
    the predicted ratings.
    
    Rp = Predicted rating matrix of size (users x items)
    ideal = optimal TopN list matrix of size (users x N)   
    ranked = re-ranked TopN matrix of size (users x N)
    '''
    users = len(ideal)
    N = len(ideal[0])
    dcg=0
    idcg=0
    for u in range(users):
        for n in range(N):
            dcg += ((2**(Rp[u][ranked[u][n]]))-1)/math.log((1+(n+1)),2)
            idcg += ((2**(Rp[u][ideal[u][n]]))-1)/math.log((1+(n+1)),2)
    avg_DCG = dcg/users
    avg_NDCG = avg_DCG/(idcg/users)
    return avg_NDCG, avg_DCG

def get_NDCG2(R,ideal,ranked):
    '''
    Function to calculate the NDCG and DCG of the re-ranked TopN list but taking into account
    the actual ratings given by the users (R). If there is no actual rating R(u,i), it is not
    counted for the final result.
    
    R = Actual rating matrix of size (users x items) (With 0 meaning there is no actual rating)
    ideal = optimal TopN list matrix of size (users x N)   
    ranked = re-ranked TopN matrix of size (users x N)
    '''
    users = len(ideal)
    N = len(ideal[0])
    dcg=0
    idcg=0
    rc=0
    ic=0
    for u in range(users):
        for n in range(N):
            if R[u][ranked[u][n]]!=0:
                rc +=1
                dcg += ((2**(R[u][ranked[u][n]]))-1)/math.log((1+(n+1)),2)
            if R[u][ideal[u][n]]!=0:
                ic +=1
                idcg += ((2**(R[u][ideal[u][n]]))-1)/math.log((1+(n+1)),2)
    avg_DCG = dcg/(users*rc)
    avg_NDCG = avg_DCG/(idcg/(users*ic))
    return avg_NDCG, avg_DCG


def rmse_mae(ideal, reranked):
    '''
    Function to calculate the positional RMSE and MAE between the position of the items in
    the optimal TopN list and the new re-ranked TopN list.
    
    ideal = optimal TopN list matrix of size (users x N)   
    reranked = reranked TopN list matrix of size (users x N)
    '''
    users = ideal.shape[0]
    items = ideal.shape[1]
    n = len(reranked[0])
    act_position = np.zeros((users,n))
    pred_position = act_position.copy()
    
    act_order = [(-u).argsort(kind='stable') for u in ideal]
        
    for u in range(users):
        for i in range(n):
            act_position[u][i] = np.where(act_order[u] == reranked[u][i])[0][0]+1
            pred_position[u][i] = i+1

    return np.sum((np.sum((act_position-pred_position)**2,axis=1)/n)**(0.5))/users, np.sum(np.sum(np.abs(act_position-pred_position),axis=1)/n)/users
            
    
def presicion_in_recommendations(rating_matrix, ranked,th):
    '''
    Function to calculate the positional RMSE and MAE between the position of the items in
    the optimal TopN list and the new re-ranked TopN list.
    
    rating_matrix = actual rating matrix of size (users x items)   
    ranked = ranked TopN list matrix of size (users x N)
    th = high rating threshold T_H
    '''
    correct = 1
    total = 1
    for u in range(len(rating_matrix)):
        for i in ranked[u]:
            if rating_matrix[u][i]!=0:
                total += 1
                if rating_matrix[u][i]>=th:
                    correct +=1
    return correct/total,correct/(len(ranked[u])*len(rating_matrix))
                    