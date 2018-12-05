
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

def read_file(file, artificial = 1):
    if artificial == 1:
        with open("artificial/" + file, 'r') as f:
            data = f.read().split(" ")
            data.remove("")
        return data
    else:
        with open("20newsgroups/" + file, 'r') as f:
            data = f.read().split(" ")
            data.remove("")
        return data

K = 20
D = 100
alpha = 5.0/K
beta = 0.01
N_iters = 500

data = np.array([])
d = np.array([],dtype = int)
z = np.array([],dtype = int)

for i in range(1, D+1):
    data_temp = np.asarray(read_file(str(i), artificial=0))
    l_temp = data_temp.shape[0]
    data = np.append(data, data_temp)
    
    d_temp = i * np.ones([1,l_temp], dtype = int)
    d = np.append(d, d_temp)
    
    z_temp = np.random.randint(1, K+1, [1,l_temp])
    z = np.append(z, z_temp)

N_words = data.shape[0]
Vocab = list(set(data))
V = len(Vocab)
Vocab.sort()


# Initializing w(n) and then coverting the list of words to a list of indices of the words
word_index = defaultdict.fromkeys(Vocab,0)
for i in range(len(Vocab)):
    word_index[Vocab[i]] = i+1

w = np.zeros(N_words, dtype = int)
for i in range(N_words):
    w[i] = word_index[data[i]]

# Initialized pi, Cd, Ct
pi = np.random.permutation(range(0,N_words))
        
Ct = np.zeros((K,V))
Cd = np.zeros((D,K))

for i in range(D):
    for j in range(K):
        Cd[i,j] = sum((d == i+1) * (z == j+1))
for i in range(K):
    for j in range(V):
        Ct[i,j] = sum((z == i+1) * (w == j+1))

P = np.zeros(K)

# Gibbs sampling
for i in range(N_iters):
    print(i)
    for n in range(0, N_words):
        t = pi[n]
        word = w[t]
        doc = d[t]
        topic = z[t]
        Ct[topic-1,word-1] -= 1
        Cd[doc-1,topic-1] -= 1
        
        for k in range(0,K):
            P[k] = (Ct[k,word-1] + beta)/(V*beta + sum(Ct[k,:])) * (Cd[doc-1,k] + alpha)/(K*alpha + sum(Cd[doc-1,:]))
        P = P/sum(P)
        topic = np.random.choice(range(1,K+1),p = P)
        z[t] = topic
        Ct[topic-1,word-1] = Ct[topic-1,word-1] + 1
        Cd[doc-1,topic-1] = Cd[doc-1,topic-1] + 1

most_freq = np.zeros([K,5] ,dtype = object)
for i in range(K):
    freq = np.argsort(Ct[i], kind = 'mergesort')[-5:][::-1]
    for j in range(5):
        most_freq[i,j] = Vocab[freq[j]]
print(most_freq)
df = pd.DataFrame(most_freq)
df.to_csv("topicwords.csv",header = False, index = False)

X_bag = np.zeros([D,V])
X_lda = np.zeros([D,K])

for i in range(D):
    for j in range(V):
        X_bag[i,j] = (sum((d == i+1) * (data == Vocab[j])) + 0.0) / sum(d == i+1)
for i in range(D):
    for j in range(K):
        X_lda[i,j] = (Cd[i,j] + alpha)/(K*alpha + np.sum(Cd[i,:]))

df = pd.read_csv("20newsgroups/index.csv", header = None)
Y = df.iloc[:,[1]].values

def train_test_split(data,labels):
    n = data.shape[0]
    i = [i for i in range(n)]
    random.shuffle(i)
    test_i = i[0:int(n/3.0)]
    train_i = i[int(n/3.0): n]
    train_data = data[train_i]
    test_data = data[test_i]
    train_label = labels[train_i]
    test_label = labels[test_i]
    return train_data, test_data, train_label, test_label

def compute_acc(N_t, phi_test, test_label, SN, w):
    tp = tn = fp = fn = 0
    for i in range(N_t):
        mu_a = phi_test[i].dot(w)
        sigma_a_squared = phi_test[i].T.dot(SN.dot(phi_test[i]))
        kappa = (1 + np.pi * sigma_a_squared / 8) ** (-0.5)
        p = 1.0 / (1 + np.exp(- kappa * mu_a))
        if p >= 0.5:
            if test_label[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if test_label[i] == 0:
                tn += 1
            else:
                fn += 1
    acc = (tp + tn) / (tp + fp + tn + fn)
    return acc

def newton(train_data, train_label, test_data, test_label):
    t = train_label
    N = len(train_data)
    ones = np.array([[1]] * N)
    phi = np.concatenate((ones, train_data), axis=1)
    M = len(phi[0])
    w = np.array([[0]] * M)
    update = 1
    n = 1
    alpha = 0.1
    I = np.eye(M)

    while update > 10 ** -3 and n < 100:
        w_old = w
        a = phi.dot(w_old)
        y = 1.0 / (1 + np.exp(-a))
        r = y * (1 - y)
        R = np.diag(r.ravel())
        temp1 = phi.T.dot(y - t) + alpha * w_old
        temp2 = alpha * I + phi.T.dot(R.dot(phi))
        w_new = w_old - np.linalg.inv(temp2).dot(temp1)
        update = np.linalg.norm(w_new - w_old) / np.linalg.norm(w_old)
        w = w_new
        n += 1

    N_t = len(test_data)
    ones = np.array([[1]] * N_t)
    phi_test = np.concatenate((ones, test_data), axis=1)
    a = phi.dot(w)
    y = 1.0 / (1 + np.exp(-a))
    SN_inv = alpha * I
    for i in range(N):
        SN_inv += y[i] * (1 - y[i]) * np.outer(phi[i], phi[i])
    SN = np.linalg.inv(SN_inv)
    acc = compute_acc(N_t, phi_test, test_label, SN, w)

    return acc

# Plot the learning curves
sizes = [i/20 for i in range(2, 21)]
acc_lda = np.zeros([30,len(sizes)])
acc_bag = np.zeros([30,len(sizes)])

for k in range(30):
    train_data_lda, test_data_lda, train_label_lda, test_label_lda = train_test_split(X_lda,Y)
    train_data_bag, test_data_bag, train_label_bag, test_label_bag = train_test_split(X_bag,Y)
    j = 0
    for s in sizes:
        acc_lda[k,j] = newton(train_data_lda[0:int(s*D)], train_label_lda[0:int(s*D)], test_data_lda, test_label_lda)
        acc_bag[k,j] = newton(train_data_bag[0:int(s*D)], train_label_bag[0:int(s*D)], test_data_bag, test_label_bag)
        j += 1

lda_mean = np.mean(acc_lda, axis = 0)
lda_std = np.std(acc_lda, axis = 0)
bag_mean = np.mean(acc_bag, axis = 0)
bag_std = np.std(acc_bag, axis = 0)

plt.gcf().clear()
(_, caps, _) = plt.errorbar(sizes, lda_mean, yerr = lda_std, ecolor='r', color = 'b', capsize=20, label = "LDA")
for cap in caps:
    cap.set_markeredgewidth(1)

(_, caps, _) = plt.errorbar(sizes, bag_mean, yerr = bag_std, ecolor='c', color = 'g', capsize=20, label = "Bag of Words")
for cap in caps:
    cap.set_markeredgewidth(1)

plt.xlabel('Train Size')
plt.ylabel('Accuracy')
plt.title('LDA v/s Bag of words')
plt.legend(loc = "best")
plt.savefig("plot.png")




