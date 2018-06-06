import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import time
from pulp import *
import math
import random as rd
import dataset_generation 


# ----------------------- helper functions -------------------------

# cross products
def cross(a,b): 
    p = 0
    for (x,y) in zip(a,b):
        p+=(x*y)
    return p


# formalize the output from linear program solver
def output(arr): 
    new = np.zeros((len(arr),len(arr[0])))
    for i in range(len(new)):
        for j in range(len(new[0])):
            new[i][j] = value(arr[i][j])
    return new


# histogram plot: rareness against number of skills 
# ex. how many skills are only owned by one person
# scatter plot: demand and supply of skills
# ex. point(1,2) means this skill is owned by one person and required by two tasks
def rareness_inspectation(parr, tarr):
    # rareness = 1
    tarr.sum(0)[parr.sum(0)==1]

    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.title('Demand and supply plot of skills')
    plt.scatter(parr.sum(0),tarr.sum(0))
    plt.xlabel("# of people")
    plt.ylabel("# of tasks")
    # plt.xlim(0,50)
    # plt.ylim(0,50)


    # rareness plot
    plt.subplot(122)
    plt.title('Rareness Plots of skills')
    temp = np.bincount(parr.sum(0))
    temp2 = np.arange(len(np.bincount(parr.sum(0))))
    #temp2 = temp2[temp>0]
    #temp = temp[temp>0]
    plt.bar(temp2, temp)
    plt.xlabel('rareness (rareness=1 means only 1 person owns the skill)')
    plt.ylabel('number of skills')
    plt.show()
    print('Least L:', max(tarr.sum(0)/parr.sum(0)))
    

# calculate and print how well the given people cover the tasks
def coverage_inspectation(people,tasks,ifprint = True):
    
    people_super = people.sum(0)>0
    people_super = people_super.astype(int)
    cover_rate = np.dot(tasks, people_super)/ tasks.sum(1)
    if ifprint:
        print('Percentege of tasks that can be fully covered: ', sum(cover_rate==1)/tasks.shape[0])
        print('Percentage of tasks that can not be covered: ', sum(cover_rate==0)/tasks.shape[0])
        print('Maximun coverage rate: ', sum(cover_rate)/tasks.shape[0])
    return [sum(cover_rate==1)/tasks.shape[0], sum(cover_rate==0)/tasks.shape[0], 
            sum(cover_rate)/tasks.shape[0]]


# ----------------------- core functions -------------------------

def LPRD(J,P, offset = []):
    """
    minimize personal loads.

    Args:
        J: An array with shape (n_tasks,n_features) containing info about tasks
        P: An array with shape (n_people,n_features) containing info about people
        offset: a list of index, indicating people that are not participating (for test purpose)
        
    Returns:
        A binary matrix X with shape (n_people, n_tasks), indicating the assignments.

    """
    # assignment matrix
    X = [[0 for x in range(len(J))] for y in range(len(P))]  

    # declare your variables
    L = LpVariable("L", 0, 1000)
    namestr = 0 
    for i in range(len(X)):
        for j in range(len(X[1])):
            X[i][j] = LpVariable('x'+ str(namestr), 0, 1) # 0=<x<=1
            namestr += 1

    # defines the problem
    prob = LpProblem("problem", LpMinimize)
    
    # defines the objective function to minimize
    prob += L
    
    # find able-cover
    able_cover = np.array(P).sum(0)>0
    print("able to cover:",sum(able_cover))
    
    # find needed-cover
    needed_cover = np.array(J).sum(0)>0
    print("need cover:",sum(needed_cover))
      
    # find those cannot be covered
    needed_delete = [x>y for (x,y) in zip(needed_cover,able_cover)]
    print("cannot cover:", sum(needed_delete))
    needed_delete_mutiplicity = \
            [np.array(J).sum(0)[i]*needed_delete[i] for i in range(len(needed_delete))]
    print("cannot cover(multiplicity):", sum(needed_delete_mutiplicity))
    
    # defines the regular constraints
    for i in range(len(X)):  # all people's loads subject to a uppper bound 
        prob += sum(X[i])<=L
        
    # offset people cannot participate    
    for i in offset:  
        prob += sum(X[i])==0
        
    # all skills in all tasks must be covered 
    for i in range(len(J)): 
        for j in range(len(J[0])):
            if needed_delete[j] == 0 and J[i][j]==1:
                prob += cross([a[i] for a in X],P[:,j]) >= J[i][j]    

    # solve the problem
    status = prob.solve(GLPK(msg=0))
    print(LpStatus[status])
    
    # print output 
    print("max Load: ",value(L))
    X = output(X)
    
    return output(X)

def thresholding(X,J,P,step):
    """use thresholding technique to integerize the matrix resulted from the linear program sovler.

    Args:
        X: Matrix resulted from the linear program sovler.
        J: An array with shape (n_tasks,n_features) containing info about tasks
        P: An array with shape (n_people,n_features) containing info about people
        step: a list of double in (0,1) as candidate thresholds
        
    Returns:
        An array with shape (len(step), 2). First column is L, second is un-covered percentage. 
    """    

    LP = []
    temp = X.copy()
    for thres in np.arange(0,1,step):
        # make integer output based on threshold
        for i in range(len(X)):
            for j in range(len(X[0])):
                if thres<X[i][j]:
                    temp[i][j] = 1 
                else:
                    temp[i][j] = 0
        
        # calculate L and percentage of covered 
        L = max(temp.sum(1))
        covered = np.dot(temp.transpose(), np.array(P))
        p_uncovered = np.bitwise_and(covered==0, np.array(J)>0).sum(1)/np.array(J).sum(1)
        LP.append([L,sum(p_uncovered)])

    return np.array(LP)

def thresholding_alpha(LP, alpha_range):
    """Given differrent alphas, return the min score
		by choosing among different thresholding results.

    Args:
        LP: Ouput array from "thresholding()"
        alpha_range: a list of double in (0,1)
        
    Returns:
        Three lists with length = len(alpha_range): scores, L, and un-covered percentage
    """    
    score = []
    c = []
    l = []
    for alpha in alpha_range:
        temp = 10000
        temp2 = -1
        temp3 = -1
        for row in LP:
            if temp>row[0]*alpha + row[1]*(1-alpha):
                temp = row[0]*alpha + row[1]*(1-alpha)
                temp2 = row[0]
                temp3 = row[1]
        score.append(temp)
        l.append(temp2)
        c.append(temp3)
    return score,l,c

def randomizing(X,J,P,delta,alpha_range):
    """use randomizing technique to integerize the matrix resulted from the linear program sovler.
		Given differrent alphas, return the min score

    Args:
	    X: Matrix resulted from the linear program sovler.
	    J: An array with shape (n_tasks,n_features) containing info about tasks
	    P: An array with shape (n_people,n_features) containing info about people
		delta: probability of giving a failing answer, used to calculate the times of tossing coin 
		alpha_range: a list of double in (0,1)
        
    Returns:
        Three lists with length = len(alpha_range): scores, L, and un-covered percentage
    """  
    R = round(math.log(2*max(len(P), len(J)*len(J[0]))/delta))
    temp = X.copy()
    # randomize process
    for i in range(len(X)):
        for j in range(len(X[0])):
            if X[i][j] == 0 or X[i][j] == 1: 
                temp[i][j] = X[i][j]
                continue
            for k in range(R):
                if rd.uniform(0,1)<X[i][j]: # if at least one time it's chosen
                    temp[i][j] = 1 
                    break
                if k == R-1: 
                    temp[i][j] = 0 # if not chosen in any round
    
    # calculate L and percentage of covered 
    L = max(temp.sum(1))
    covered = np.dot(temp.transpose(), np.array(P))
    p_uncovered = np.bitwise_and(covered==0, np.array(J)>0).sum(1)/np.array(J).sum(1)
    #print(L,sum(p_uncovered))
    
    # bring in alpha
    score = []
    for alpha in alpha_range:
        temp = L*alpha + sum(p_uncovered)*(1-alpha)
        score.append(temp)    
    return score, [L for i in alpha_range], [sum(p_uncovered) for i in alpha_range] 

def run_from_matrix(pp, tt, filename, alpha_range):
    """run CL and L using a range of alphas, save solution matrix and scores as csv files
       plot results
       
    Args:
        tt: An array with shape (n_tasks,n_features) containing info about tasks
        pp: An array with shape (n_people,n_features) containing info about people
        filename: solution matrix as LP_filename.csv, scores table as scores_file.csv
        alpha_range: a list of double in (0,1)
        
    Returns:
        solution matrix
    """ 
    
    alpha = alpha_range
    tt = tt[tt.sum(1)!=0 ,:]
    print('people: ', pp.shape[0], 'tasks: ', tt.shape[0] )
    t0 = time.time()
    
    # solve linear program
    # sol = np.genfromtxt('dec 13/results/LP_{}.csv'.format(filename), delimiter=",")
    sol = LPRD(tt,pp)
    np.savetxt('LP_{}.csv'.format(filename), sol, delimiter=",")
    print('number of Frac:', np.bitwise_and(sol>0, sol<1).sum())
    print('run time:', time.time()-t0)
    
    # run CL
    pairs = thresholding(sol.copy(), tt, pp, 0.05)
    score,CL_L,CL_C = thresholding_alpha(pairs, alpha)
    print('CL: ',score)
    
    # run L 
    score_rd,L_L,L_C = randomizing(sol.copy(), tt, pp, 0.05, alpha)
    print('L: ',score_rd)

    # save calculated scores 
    out = pd.DataFrame.from_records(np.array([score,CL_L, CL_C,score_rd,L_L,L_C,alpha]).transpose())
    out.columns = ['CL', 'CL_L', 'CL_C','L','L_L','L_C', 'alpha']
    out.to_csv('scores_{}.csv'.format(filename))
    
    # plot L vs. SPUC
    x_position,y_position = pairs[:,1],pairs[:,0]
    plt.figure(figsize=(16,8))
    
    plt.subplot(121)
    plt.title('CL Plot: Thresholding Effects')
    plt.grid()
    plt.plot(x_position,y_position,'rx')
    plt.ylabel('L: Max Load')
    plt.xlabel('SPUC: Sum of % Uncovered')
    # plt.xlim(0,10)
    # plt.ylim(0,20)

    # plot min_obj under different alpha
    plt.subplot(122)
    plt.title('Scores Plot: at different alpha')
    plt.plot(alpha, score_rd, label = 'L score')
    plt.plot(alpha, score, label = 'CL score')
    plt.grid()
    plt.ylabel('Obj')
    plt.xlabel('Alpha')
    plt.legend(loc=2, borderaxespad=0.)

    plt.show() 

def run_from_filename(filename,path, alpha_range):
    """given path and filename of the datasets, run 
       
    Args:
        filename: core name of files
        path: path of the file 
        alpha_range: a list of double in (0,1)
        
    Returns:
		N/A
    """ 
    # read
    tasks = np.genfromtxt(path + '/tasks{}.csv'.format(filename), delimiter=',')
    people = np.genfromtxt(path + '/people{}.csv'.format(filename), delimiter=',')
    people = people.astype(int)
    tasks = tasks.astype(int)
    
    # inspect
    rareness_inspectation(people,tasks)
    coverage_inspectation(people,tasks)
    
    # run alg. and output
    run_from_matrix(people, tasks, filename, alpha_range)


# ----------------------- main function -------------------------
if __name__ == '__main__':
    print('running main')
    dataset_generation.uniform_data_generation(10,10,10,10,10,'Jan 10')
    run_from_filename('(s10p10t10;10;10)', 'Jan 10', [0,0.2,0.4,0.6,0.8,1])