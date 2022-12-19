#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas
import numpy as np
import random
import math
from sklearn.preprocessing import MinMaxScaler
import random
import progressbar as pb


# In[52]:

def baseline(rec_mat, rank_x, th,tr):
    '''
       Method proposed by Jannach & Adomavicius (2017)
       Function that re-rank items based on a recomendation matrix (rec_mat),
       a ranking score of the items (rank_x) and a threshold T_R(tr).
       Threshold (T_H) is not used, it still produce the same results.
    '''
    rank_std = 1/rec_mat
    
    pars_mat = np.zeros((rec_mat.shape[0],rec_mat.shape[1]))
    au = rank_x.max()

    
    for u in range(rec_mat.shape[0]):
        for i in range(rec_mat.shape[1]):
            rat = rec_mat[u][i]
            if rat>=tr:
                pars_mat[u][i] = rank_x[i]
            else:
                pars_mat[u][i] = rank_std[u][i]+au
    return pars_mat


def topN_rec(n,rec_mat):
    '''
    TopN recommendation for all users based on recommendation matrix (rec_mat)
    return list of n items for u users
    '''
    topn_rec = [(-u).argsort(kind='stable')[:n] for u in rec_mat]
    return topn_rec

def topN_rec_user(user,n,rec_mat):
    '''
    TopN recommendation for user based on recommendation matrix
    return n items
    '''
    return (-rec_mat[user]).argsort()[:n]

def top_rec_T(user,rec_mat,th):
    '''
    Get items with rating higher than th, items ordered from higher to lower
    '''
    final_rec = []
    gtt_rec = []
    u_rec = rec_mat[user]
    sort_rec = (-u_rec).argsort(kind='stable')
    for rec in sort_rec:
        if u_rec[rec]>=th:
            final_rec.append(rec)
        else:
            return final_rec
    return final_rec


# In[124]:

def impact_in_recommendations(correct, new):
    '''
    This function returns a list with the corresponding computed impact for every user.
    It compares the correct list with the new list.
    '''
    u = len(correct)
    return np.array([1-(len(set(correct[i]) & set(new[i]))/len(correct[i])) for i in range(u)])


def initialize_weights(dim,users):
    '''
    Function to initialize a weight matrix of size (users x dim) with random numbers between 0 and 1.
    '''
    return np.random.uniform(0,1,(users,dim))


def get_cost(Y,Y_hat):
    '''
    Cost function, compares the desired values (Y) to the actual obtained values (Y_hat).
    It gets the average of the cost of all the users.
    '''
    Y_res=Y-Y_hat  
    return np.sum(np.abs(Y_res))/len(Y_res)

def update_weights(W,Y,Y_hat,lr,tol):
    '''
    Function to update the weights (W)
    Y = Vector of size (users) containing the desired impact for each user
    Y_hat = Vector of size (users) containing the obtained impact for each user
    lr = learning rate
    tol= Tolerance on the impact value (desired impact+-tol is accepted)
    
    '''
    Y_res=Y-Y_hat
    k = len(W[0])-1
    for u in range(len(Y_res)):
        u_res = Y_res[u]
        if np.abs(u_res)>tol:
            if u_res>0:
                for c in range(k):
                    W[u][c+1]=W[u][c+1]+((lr*u_res))
            else:
                W[u][0]=W[u][0]-((lr*u_res))  
    return W


# In[109]:


class MARS():
    def __init__(self, Rp):
        self.Rp = Rp                #Predicted rating matrix (R^*)
        self.users = Rp.shape[0]    #Total number of users
        self.items = Rp.shape[1]    #Total number of items
        self.rank_vectors = None
        self.rank_tensor = None       
        
    
    def to_rank(self, vectors=None, tensor=None):
        '''
        Transform attribute vectors and attribute tensors to rank scores
        rank_x = 1/(x+1)
        '''
        tmax = self.Rp.max()
        tmin = self.Rp.min()
        self.rank_std = 1/(self.Rp.copy()+1)
        
        if vectors is not None:
            v_min = vectors.min(axis=0)
            self.rank_vectors = 1/(((((vectors-v_min)/(vectors.max(axis=0)-v_min))*(tmax-tmin)) + tmin)+1)
        
        if tensor is not None:
            self.rank_tensor = [1/(((((rec-rec.min())/(rec.max()-rec.min()))*(tmax-tmin)) + tmin)+1) for rec in tensor]

            
    def get_mars(self,tr,weights):
        '''
        MARS proposed method explained in the paper
        tr = Ranking threshold T_R
        weights = Weights given for the weighted re-ranking processs
        '''
        self.weights = weights/np.amax(weights, axis=1)[:,None]
        self.tr = tr
        
        #Calculate au from the minimum value in Rp (1/(min R^*+1) x #attributes)
        au = 1/(self.Rp.min()+1)*self.weights.shape[1]
        mars_mtx = np.zeros((self.users,self.items))
        
        #Case where there are attribute vectors and attribute matrices
        if self.rank_vectors is not None and self.rank_tensor is not None:
            for u in range(self.users):      
                att = np.append(self.rank_vectors,np.array([rank_mat[u] for rank_mat in self.rank_tensor]).T,axis=1)
                x = np.insert(att, 0, self.rank_std[u], axis=1)
                x = x*self.weights[u]
                x = np.sum(x, axis=1)
                for i in range(self.items):
                    if self.Rp[u][i]<self.tr:
                        x[i] = self.rank_std[u][i]+au 
                mars_mtx[u] = x
                
        #Case where there are only attribute vectors       
        elif self.rank_vectors is not None:
            for u in range(self.users):
                att = self.rank_vectors
                x = np.insert(att, 0, self.rank_std[u], axis=1)

                x = x*self.weights[u]
                x = np.sum(x, axis=1)
                for i in range(self.items):
                    if self.Rp[u][i]<self.tr:
                        x[i] = self.rank_std[u][i]+au 
                mars_mtx[u] = x
                
        #Case where there are only attribute matrices     
        elif self.rank_tensor is not None:
            for u in range(self.users):
                att = np.array([rank_mat[u] for rank_mat in self.rank_tensor]).T
                x = np.insert(att, 0, self.rank_std[u], axis=1)

                x = x*self.weights[u]
                x = np.sum(x, axis=1)
                for i in range(self.items):
                    if self.Rp[u][i]<self.tr:
                        x[i] = self.rank_std[u][i]+au 
                mars_mtx[u] = x
        
        #Case where there are no attribute vectors nor attribute matrices
        #MARS method == Standard method
        else:
            mars_mtx = np.array(self.rank_std)
        self.mars_matrix = mars_mtx
        return self.mars_matrix
    
    def mars_user(self,tr,weights,user,from_matrix=False):
        u = user
        
        au = 1/(self.Rp.min()+1)*len(weights)

        if self.rank_vectors is not None and self.rank_tensor is not None:
                  
            att = np.append(self.rank_vectors,np.array([rank_mat[u] for rank_mat in self.rank_tensor]).T,axis=1)
            x = np.insert(att, 0, self.rank_std[u], axis=1)

            x = x*weights
            x = np.sum(x, axis=1)
            for i in range(self.items):
                if self.Rp[u][i]<tr:
                    x[i] = self.rank_std[u][i]+au 
                    
        elif self.rank_vectors is not None:
            
            att = self.rank_vectors
            x = np.insert(att, 0, self.rank_std[u], axis=1)

            x = x*weights
            x = np.sum(x, axis=1)
            for i in range(self.items):
                if self.Rp[u][i]<tr:
                    x[i] = self.rank_std[u][i]+au 
            
        elif self.rank_tensor is not None:
            
            att = np.array([rank_mat[u] for rank_mat in self.rank_tensor]).T
            x = np.insert(att, 0, self.rank_std[u], axis=1)

            x = x*weights
            x = np.sum(x, axis=1)
            for i in range(self.items):
                if self.Rp[u][i]<tr:
                    x[i] = self.rank_std[u][i]+au 
        else:
            x = self.rank_std[u] 
        return np.array(x,dtype='float16')
    

    
    def weights_optimization(self,impact,tol,W,tr,N,iterations,s_iterations,lr,uf,auf):
        '''
        Function that returns the weights to obtain the desired impact on the user
        impact = The value of the desired impact
        tol= Tolerance on the impact value (impact+-tol is accepted)
        W is the weight and can have 2 data types
        W = integer of the number of attributes being considered +1 to initialize the weight matrix
        W = Initial weight matrix that is going to be optimized
        tr = Ranking threshold T_R
        N = Value for TopN items
        iterations = Number of iterations of the first optimization process
        s_iterations = Number of iterations of the second optimization process, aggresive update
        lr = Learning rate of the optimization process
        uf = Learning rate update factor (lr decreases each iteration if 0<uf<1)
        auf = Aggresive update factor (auf>1)
        '''
        Y = np.array([impact]*self.users)
        rec_list_usersRp = topN_rec(N,self.Rp)
        c1 = float('inf')
        
        if type(W) == int:
            self.weights=initialize_weights(W,self.users)
        else:
            self.weights = W/np.amax(W, axis=1)[:,None]
        
        weights_up = self.weights
        for i in range(iterations):
            mars_rat_mat = self.get_mars(tr, self.weights)
            rec_list_users = topN_rec(N,-mars_rat_mat)
            Y_hat=impact_in_recommendations(rec_list_usersRp,rec_list_users)
            c2=get_cost(Y,Y_hat)
            print('Cost= ', c2, 'Impact= ', sum(Y_hat)/len(Y_hat))

            if (c2>=c1) | (c2<tol) | (c1-c2<0.001):
                self.weights=weights_up
                print('Stopped at iteration:', i+1)
                break
            else:
                weights_up = self.weights
                self.weights=update_weights(self.weights,Y,Y_hat,lr,tol)
                c1=c2
                lr = lr*uf
         
        for _ in range(s_iterations):
            mars_rat_mat = self.get_mars(tr, self.weights)
            rec_list_users = topN_rec(N,-mars_rat_mat)
            Y_hat=impact_in_recommendations(rec_list_usersRp,rec_list_users)
            hiu = np.where(Y_hat>impact+tol)[0]
            if len(hiu)>0:
                for u in hiu:
                    self.weights[u,0]=self.weights[u,0]*auf
                auf=auf*1.5    #Update of the Aggresive update factor
            else:
                break
    
    def save_weights(self,path):
        '''
        Fuction to save the weights in the defined path.
        '''
        np.savetxt(path, self.weights, delimiter=',')
        
    def accuracy_in_recommendations(correct, new):
        u = len(correct)
        return sum([len(set(correct[i]) & set(new[i]))/len(correct[i]) for i in range(u)])/u


