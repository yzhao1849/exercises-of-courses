"""EM Algorithm for Image Segmentation: Gaussian Mixture Models"""

import numpy as np
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.stats import norm #normal distribution
from math import log
from scipy.cluster.vq import kmeans2

### Produce a binary mask that marks all pixels with an intensity greater than zero
brain=misc.imread('brain.png')
fig1=plt.figure()
fig1.add_subplot(1,2,1)
plt.axis('off')
plt.imshow(brain,cmap='gray')
fig1.add_subplot(1,2,2)
plt.axis('off')
brain_denoised=ndimage.median_filter(brain,size=4)
plt.imshow(brain_denoised,cmap='gray')

binary_mask=brain_denoised.astype(bool)
invert_binary_mask=np.invert(binary_mask)
plt.figure()
plt.title("mask")
plt.imshow(binary_mask,cmap='gray')

### Initialize the parameters of a three-compartment Gaussian mixture model to some reasonable values
### and use them to compute the responsibilities rho_ik of cluster k for pixel i
plt.figure()
onlyBrain=brain_denoised[binary_mask] #get only the part that has a value larger than 0

plt.hist(onlyBrain.ravel(),bins=256,range=(0,256),log=True,fc='k',ec='k')
plt.xlim(0,257)

#use kmeans to estimate the mean and sd:
onlyBrain=brain_denoised[binary_mask].astype(float)
means, labels=kmeans2(onlyBrain,3,minit='random')
mask1=(labels==0)
mask2=(labels==1)
mask3=(labels==2)
sds=[]
sds.append(onlyBrain[mask1].std())
sds.append(onlyBrain[mask2].std())
sds.append(onlyBrain[mask3].std())

pis=[]
total=onlyBrain.shape[0]
pis.append(onlyBrain[mask1].shape[0]/total)
pis.append(onlyBrain[mask2].shape[0]/total)
pis.append(onlyBrain[mask3].shape[0]/total)

print("kmeans result: means:",means)
print("kmeans result: sds:",sds)
print("kmeans result: pis:",pis)


### Visualize the responsibilities
def Responsibility1(data): #use means,sds,pis from lists, to make sure the used parameters are updated
    return pis[0]*norm.pdf(data,means[0],sds[0])
def Responsibility2(data):
    return pis[1]*norm.pdf(data,means[1],sds[1])
def Responsibility3(data):
    return pis[2]*norm.pdf(data,means[2],sds[2])

### E step
def e_step():
    Resp1 = np.vectorize(Responsibility1)  # function that applys to each elements of numpy array, to get the responsibility matrix of cluster1 for all data
    Resp2 = np.vectorize(Responsibility2)
    Resp3 = np.vectorize(Responsibility3)

    resp_matrices=[] # a list of the responsibility matrices
    resp_matrices.append(Resp1(brain_denoised))
    resp_matrices.append(Resp2(brain_denoised))
    resp_matrices.append(Resp3(brain_denoised))

    resp_sum = resp_matrices[0] + resp_matrices[1] + resp_matrices[2]
    resp_matrices[0] /= resp_sum
    resp_matrices[0][invert_binary_mask] = 0  # set the background to 0
    resp_matrices[1] /= resp_sum
    resp_matrices[1][invert_binary_mask] = 0
    resp_matrices[2] /= resp_sum
    resp_matrices[2][invert_binary_mask] = 0
    return resp_matrices, resp_sum

resp_matrices, resp_sum=e_step()
order_index=[] #the indices of the 3 means ordered from smallest to biggest (background, gray matter, white matter) in the means list
for i in sorted(means):
    order_index.extend(np.where(means==i)[0])
bg_index=order_index[0]
gm_index=order_index[1]
wm_index=order_index[2]
resp_total = np.stack((resp_matrices[bg_index], resp_matrices[wm_index], resp_matrices[gm_index]), axis=-1) #RGB. Corresponding to background, white matter, gray matter, respectively
plt.figure()
plt.title("Kmeans initialization")
plt.axis('off')
plt.imshow(resp_total)


fig=plt.figure()
fig.suptitle("Converging image (first 12 steps)")
fig.add_subplot(3, 4,1)
plt.title('1')
plt.axis('off')
plt.imshow(resp_total)


loge=np.vectorize(log)
resp_sum[invert_binary_mask]=1 #set background to 1 to avoid influence on log_likelihood
log_likelihood_list=[]
log_likelihood_list.append(np.sum(loge(resp_sum)))

### Update parameters: M step
def m_step(resp_matrices):
    n1 = np.sum(resp_matrices[0])
    n2 = np.sum(resp_matrices[1])
    n3 = np.sum(resp_matrices[2])
    num_pixel = np.count_nonzero(binary_mask)  # number of pixels inside the mask
    means[0] = np.sum(
        brain_denoised * resp_matrices[0]) / n1  # no need to use the mask because 0 multiplied with any number will still be 0
    means[1] = np.sum(brain_denoised * resp_matrices[1]) / n2
    means[2] = np.sum(brain_denoised * resp_matrices[2]) / n3

    temp1 = (brain_denoised - means[0]) ** 2 * resp_matrices[0]
    #print("np.sum(temp1)_before",np.sum(temp1))
    temp1[invert_binary_mask] = 0  # exclude the values in the background
    sds[0] = (np.sum(temp1) / n1)**0.5 #square root!!!!
    #print("mean1",mean1)
    #print("n1",n1)
    #print("sd1",sd1)
    temp2 = (brain_denoised - means[1]) ** 2 * resp_matrices[1]
    temp2[invert_binary_mask] = 0
    sds[1] = (np.sum(temp2) / n2)**0.5 #square root!!!!
    temp3 = (brain_denoised - means[2]) ** 2 * resp_matrices[2]
    temp3[invert_binary_mask] = 0
    sds[2] = (np.sum(temp3) / n3)**0.5 #square root!!!!
    pis[0] = n1 / num_pixel
    pis[1] = n2 / num_pixel
    pis[2] = n3 / num_pixel
    return

m_step(resp_matrices)

print("new means:",means)
print("new sds:",sds)
print("new pis:",pis)

resp_matrices, resp_sum=e_step()
resp_total = np.stack((resp_matrices[bg_index], resp_matrices[wm_index], resp_matrices[gm_index]), axis=-1) #RGB. Corresponding to background, white matter, gray matter, respectively

fig.add_subplot(3, 4,2)
plt.title('2')
plt.axis('off')
plt.imshow(resp_total)

resp_sum[invert_binary_mask]=1 #set background to 1 to avoid influence on log_likelihood
log_likelihood_list.append(np.sum(loge(resp_sum)))

rounds=2 #record the rounds of EM cycles
print("new and old log likelihoods:",log_likelihood_list[1],log_likelihood_list[0])

### Iterate the E and M steps of the algorithm until convergence
while abs((log_likelihood_list[rounds-1]-log_likelihood_list[rounds-2])/log_likelihood_list[rounds-2])>0.00001: #not converged, used an arbitrary threshold
    m_step(resp_matrices) #update the parameters
    resp_matrices, resp_sum = e_step()
    resp_sum[invert_binary_mask] = 1  # set background to 1 to avoid influence on log_likelihood
    log_likelihood_list.append(np.sum(loge(resp_sum)))
    print("log_likelihood_new:",log_likelihood_list[rounds]) #the newly computed log_likelihood
    rounds+=1
    if rounds<=12:
        fig.add_subplot(3, 4, rounds)
        plt.title(rounds)
        plt.axis('off')
        plt.imshow(resp_total)

plt.figure()
plt.axis('off')
plt.title('Converged result')
resp_total = np.stack((resp_matrices[bg_index], resp_matrices[wm_index], resp_matrices[gm_index]), axis=-1) #RGB. Corresponding to background, white matter, gray matter, respectively
plt.imshow(resp_total)
print("Number of EM cycles before convergence:",rounds)
print("mean after convergence:",means)
print("sd after convergence:",sds)
print("pi after convergence:",pis)

plt.figure()
index=[0]*len(log_likelihood_list)
for i in range(len(log_likelihood_list)):
    index[i]=i+1
plt.plot(index,log_likelihood_list)
plt.title("log likelihood convergence")
plt.xlabel("rounds")
plt.show()

