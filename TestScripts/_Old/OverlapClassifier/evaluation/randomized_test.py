import numpy as np
import matplotlib.pyplot as plt
import sys

users = np.array([
[[
1,2.5,1.5],[
2,3,1.5],[
2,1.5,1],[
1.5,2,2],[
1,1,1],[
2,1,1.5],[
3,3,2.5],[
1.5,1,1],[
3,1,2],[
1.5,1,2]
],
[[
1,3,2],[
1,3,3],[
1,3,3],[
2,3,3],[
2,2,1],[
1,3,2],[
3,3,3],[
1,3,3],[
2,2,2],[
2,1,2]
],
[[
2.5,2,1.5],[
1.5,1.5,1.5],[
1,1.5,1.5],[
2.5,2.5,1.5],[
1.5,1,1.5],[
1,2,2],[
2,2.5,1.5],[
1,1.5,1],[
2,1.5,1.5],[
1,1,1]]
,[[
1,3,1],[
1.5,1,2],[
1,2,1],[
1.5,1,1.5],[
1,1,1],[
1,1,2.5],[
3,3,3],[
1.5,1,1],[
2,2,2.5],[
1,1,2.5]]
,[[
2,2,2],[
2,2,2],[
1,1,1],[
1,1,1],[
3,2,2],[
2,2,1],[
3,3,3],[
1,2,2],[
1,2,2],[
3,1,1]]
,[[
1,2,1],[
1,1.5,2],[
1,1,1.5],[
2.5,2.5,1],[
1,1,1],[
1,2.5,1],[
2,3,2],[
1.5,3,1],[
1,2,2],[
1,1,1]]
,[[
1,2,1],[
2,2,1],[
2,1,1],[
2,1,2],[
1,2,2],[
1,1,1],[
2,3,2],[
1,2,1],[
2,2,1],[
2,1,2]]
,[[
1,3,2],[
3,2,2],[
1.5,1,1],[
1.5,2,1],[
1,1,1],[
1,1,2],[
3,2.5,2.5],[
1,1.5,1.5],[
3,3,3],[
1,1,2]]
,[[
1,3,2],[
3,2,3],[
2,2,2],[
3,3,1],[
2,1,2],[
3,1,3],[
3,3,3],[
3,3,1],[
3,3,3],[
1,1,2]]
,[[
1,2,1],[
1,2,1],[
3,1,2],[
2,2,1],[
1,1,1],[
2,1,1],[
3,3,3],[
3,2,1],[
3,2,2],[
2,1,1]]
,[[
1,3,1],[
1,1,1],[
1,1,1],[
3,3,2],[
2,2,2],[
2,1,1],[
2,2,3],[
2,1,1],[
1,1,2],[
2,1,1]]
,[[
2,2,1],[
2,2,2],[
1,1,1],[
2,2,2],[
2,2,1],[
2,1,3],[
3,2,2],[
2,2,1],[
2,1,2],[
1,2,2]]
])


userscores = np.array([[
0.16667,2.20833,0.27083],[
0.29167,0.70833,0.58333],[
0.52083,0.45833,0.45833],[
0.89583,0.95833,0.20833],[
0.35417,0.25000,0.18750],[
0.75000,0.68750,1.12500],[
0.14583,0.27083,0.08333],[
0.89583,1.37500,0.43750],[
0.60417,0.25000,0.43750],[
0.68750,0.08333,0.68750]
])	#cf10

userdiffs = users  #- np.min(users,axis=2).reshape(12,10,1)
userscores = np.mean((userdiffs*2),axis=0)
print(userscores)

djtraj = [2,2,1,0,1,0,1,0,1,2]
excluded_list = [3,9]

N = 10000
n = 10
random_traj_scores = np.zeros(N,dtype='float')
random_traj_dj_scores = np.zeros(N,dtype='float')

import itertools
userscores_filtered = userscores[[i for i in range(n) if not i in excluded_list],:]
allpaths = itertools.product(range(3), repeat=n-len(excluded_list))
pathscores = np.zeros(3**(n-len(excluded_list)))
i = 0
#print(allpaths)
for curpath in allpaths:
	if i % 3**(n-len(excluded_list)-3) == 0:
		print(curpath)
	#print(curpath)
	pathscores[i] = np.mean([userscores_filtered[v][curpath[v]] for v in range(n-len(excluded_list))])
	#print(pathscores[i])
	i += 1
#print(pathscores)
dj_path_score = np.mean([userscores[v][djtraj[v]] for v in range(n) if not v in excluded_list])
print(dj_path_score)
print((1+np.sum(pathscores < dj_path_score)) / (1+len(pathscores)))

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

plt.hist(pathscores, bins=100)
plt.axvline(dj_path_score,c='red',linewidth=2)
plt.xlabel('Average rating over the 10 crossfades')
plt.ylabel('Number of crossfade selections')
plt.show()


# -------------------- below is bogus --------------
# for i in range(N):
	# for j in range(n):
		# if j in excluded_list:
			# continue
		# random_traj_scores[i] += userscores[j][np.random.choice(3)]
		# random_traj_dj_scores[i] += userscores[j][djtraj[j]]
# dj_traj_score = np.mean([userscores[i][djtraj[i]] for i in range(n) if i not in excluded_list])
# n = n - len(excluded_list)
# random_traj_scores /= n
# random_traj_dj_scores /= n
# print(np.mean(random_traj_scores), np.std(random_traj_scores))
# print(dj_traj_score, np.median(random_traj_scores))
# print(np.mean(random_traj_dj_scores))
# print((1+np.sum(random_traj_scores < dj_traj_score)) / (1+N))
# print((1+np.sum(random_traj_scores < random_traj_dj_scores)) / (1+N))

# plt.hist(random_traj_scores,bins=100,normed=True)
# plt.axvline(dj_traj_score,c='red')
# plt.show()


# BLUB


# usermins = np.zeros((12,10,3))
# for i in range(12):
	# for j in range(10):
		# usermin = min(users[i][j])
		# for k in range(3):
			# usermins[i][j][k] = users[i][j][k] - usermin

# for i in range(N):
	# for j in range(n):
		# randomuser = np.random.choice(12)
		# random_traj_scores[i] += usermins[randomuser][j][np.random.choice(3)]
		# random_traj_dj_scores[i] += usermins[randomuser][j][besttraj[j]]
# random_traj_scores /= n
# random_traj_dj_scores /= n
# print(random_traj_scores)
# print(random_traj_dj_scores)
# print(np.mean(random_traj_scores), np.std(random_traj_scores))
# mean_best_traj = np.mean([userscores[i][besttraj[i]] for i in range(n)])
# mean_rand_dj_traj = np.mean(random_traj_dj_scores)
# print(mean_best_traj, np.median(random_traj_scores))
# print(mean_rand_dj_traj)
# print((1+np.sum(random_traj_scores < mean_best_traj)) / (1+N))

# plt.hist(random_traj_scores,bins=100,normed=True)
# plt.hist(random_traj_dj_scores,bins=100,normed=True)
# #plt.axvline(mean_best_traj,c='red')
# #plt.axvline(mean_rand_dj_traj,c='green')
# plt.show()