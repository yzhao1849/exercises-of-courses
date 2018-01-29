"""Markov Random Fields"""

import numpy as np
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.stats import norm #normal distribution
from math import log
from math import pi
from math import exp
from scipy.cluster.vq import kmeans2
from copy import deepcopy


def plot_hard_label(hard_label,mask,means,title):
    order_index = []  # the indices of the 3 means ordered from smallest to biggest (background, gray matter, white matter) in the means list
    for i in sorted(means):
        order_index.extend(np.where(means == i)[0])
    bg_index = order_index[0]
    gm_index = order_index[1]
    wm_index = order_index[2]
    ix,iy=hard_label.shape
    for_plot=np.zeros((ix,iy),dtype='3float32')
    for x in range(ix):
        for y in range(iy):
            if mask[x,y]==False:
                if hard_label[x,y]==bg_index:
                    for_plot[x,y,0]=1 #Red
                elif hard_label[x,y]==gm_index:
                    for_plot[x,y,1]=1 #Green
                else:
                    for_plot[x,y,2]=1 #Blue

    plt.figure()
    plt.axis('off')
    plt.title(title)
    plt.imshow(for_plot)
    return

def kmeans_estimation(image):
    means, labels = kmeans2(image, 3, minit='random')
    while means.shape[0]<3: #if some cluster collapses
        means, labels = kmeans2(image, 3, minit='random')
    masks = []
    sds=[]
    pis=[]
    masks.append(labels == 0)
    masks.append(labels == 1)
    masks.append(labels == 2)

    total = image.shape[0]
    for m in masks:
        sds.append(image[m].std())
        pis.append(image[m].shape[0] / total)
    return means,sds,pis,labels

### Iterated Conditional Modes (ICM) algorithm and use it to update the label image
def ICM(image,mask,means,sds,beta,arbitrary_rounds=None): #hard_label1: initial hard label
    hard_label1 = np.zeros(image.shape)
    update_label(image, mask,hard_label1, means, sds, beta, firstICM=True)
    rounds = 0  # the number of updating rounds
    label_before_iter = np.ones(image.shape) #make sure it is different from original hard_label1 to be able to enter the while loop

    if arbitrary_rounds == None:  # default: runs until convergence (update the same label matrix, not using a new one to store the new labels!)
        while not np.array_equal(hard_label1, label_before_iter) and rounds<50:  # not converged #
            label_before_iter=deepcopy(hard_label1)
            update_label(image, mask, hard_label1, means, sds, beta) #update hard_label1
            rounds += 1

        return hard_label1
    else: #if there is input number of rounds from the user
        while rounds<arbitrary_rounds:
            original_label=deepcopy(hard_label1)
            update_label(image, mask, hard_label1, means, sds, beta)
            rounds += 1
            print("Iteration,",rounds)
            count=image_diff_count(original_label,hard_label1)
            print("Number of differently labeled pixel after iteration:",count)

        return hard_label1


def update_label(image,mask,hard_label,means,sds,beta,firstICM=False): #one round of ICM (update the original label matrix)
    #hard_label is the hard labels of all data points
    #mask is True in background, beta is the penalty level
    ix,iy=image.shape
    for x in range(ix):
        for y in range(iy):
            if mask[x,y]==False: #if the point is not in background
                if firstICM==True:
                    curr_energy = calc_energy(x, y, image, hard_label, mask, means, sds, beta, firstICM=True)
                else: curr_energy=calc_energy(x, y, image, hard_label, mask, means, sds, beta)

                hard_label[x,y]= curr_energy.argmin() #Returns the index of the minimum value!!! and date the same label matrix
    #recalculate means,sds:
    image_no_bg=image[np.invert(mask)] #without background
    masks = []
    masks.append(hard_label[np.invert(mask)] == 0)
    masks.append(hard_label[np.invert(mask)] == 1)
    masks.append(hard_label[np.invert(mask)] == 2)
    for i in range(len(masks)):
        means[i]=image_no_bg[masks[i]].mean()
        sds[i]=image_no_bg[masks[i]].std()
    return

def delta(boolean): #Penalty: return 0 if the 2 labels are the same (True), otherwise 1
    if boolean: return 0
    else: return 1

def calc_energy(x,y,image,hard_label,mask,means,sds,beta,firstICM=False):
    external_energys = np.zeros(3)  # a list of external energy if the center label is assigned as 0, 1 or 2
    internal_energys = np.zeros(3)  # a list of internal energy if the center label is assigned as 0, 1 or 2

    external_energys[0] = log((2 * pi) ** 0.5 * sds[0]) + (image[x, y] - means[0]) ** 2 / (2 * sds[0] ** 2)
    external_energys[1] = log((2 * pi) ** 0.5 * sds[1]) + (image[x, y] - means[1]) ** 2 / (2 * sds[1] ** 2)
    external_energys[2] = log((2 * pi) ** 0.5 * sds[2]) + (image[x, y] - means[2]) ** 2 / (2 * sds[2] ** 2)

    if firstICM==False: #Do not update internal energy when it's the first round of ICM
        if mask[x - 1, y] == False:  # upper neighbor not in background
            internal_energys[0] += beta * delta(hard_label[x - 1, y] == 0)
            internal_energys[1] += beta * delta(hard_label[x - 1, y] == 1)
            internal_energys[2] += beta * delta(hard_label[x - 1, y] == 2)
        if mask[x, y - 1] == False:  # left neighbor not in background
            internal_energys[0] += beta * delta(hard_label[x, y - 1] == 0)
            internal_energys[1] += beta * delta(hard_label[x, y - 1] == 1)
            internal_energys[2] += beta * delta(hard_label[x, y - 1] == 2)
        if mask[x + 1, y] == False:  # lower neighbor not in background
            internal_energys[0] += beta * delta(hard_label[x + 1, y] == 0)
            internal_energys[1] += beta * delta(hard_label[x + 1, y] == 1)
            internal_energys[2] += beta * delta(hard_label[x + 1, y] == 2)
        if mask[x, y + 1] == False:  # right neighbor not in background
            internal_energys[0] += beta * delta(hard_label[x, y + 1] == 0)
            internal_energys[1] += beta * delta(hard_label[x, y + 1] == 1)
            internal_energys[2] += beta * delta(hard_label[x, y + 1] == 2)

    energy = external_energys + internal_energys  # a list of total energy if the center label is assigned as 0, 1 or 2
    return energy

def image_diff(image1,image2): #plot the difference between 2 images (same shape), black=same, white=diff
    assert image1.shape==image2.shape
    ix,iy=image1.shape
    diff=np.zeros((ix,iy),dtype=np.uint8)
    for x in range(ix):
        for y in range(iy):
            if image1[x,y]!=image2[x,y]:
                diff[x,y]=255
    plt.figure()
    plt.imshow(diff,cmap='gray')
    return
def image_diff_count(image1,image2):
    assert image1.shape == image2.shape
    ix, iy = image1.shape
    count=0
    for x in range(ix):
        for y in range(iy):
            if image1[x, y] != image2[x, y]:
                count+=1
    return count #,diff_pos_list
def HMRF_e_step(image, mask, means, sds, beta):
    hard_label_new=ICM(image, mask, means, sds, beta)
    HMRF_resp_matrices=HMRF_resp(image,hard_label_new,mask,means,sds,beta)
    return HMRF_resp_matrices,hard_label_new
def HMRF_m_step(image, mask, HMRF_resp_matrices, means, sds):
    n1 = np.sum(HMRF_resp_matrices[0])
    n2 = np.sum(HMRF_resp_matrices[1])
    n3 = np.sum(HMRF_resp_matrices[2])
    means[0] = np.sum(image * HMRF_resp_matrices[0]) / n1
    # no need to use the mask because 0 multiplied with any number will still be 0
    means[1] = np.sum(image * HMRF_resp_matrices[1]) / n2
    means[2] = np.sum(image * HMRF_resp_matrices[2]) / n3

    temp1 = (image - means[0]) ** 2 * HMRF_resp_matrices[0]
    temp1[mask] = 0  # exclude the values in the background
    sds[0] = (np.sum(temp1) / n1) ** 0.5  # square root!!!!
    temp2 = (image - means[1]) ** 2 * HMRF_resp_matrices[1]
    temp2[mask] = 0
    sds[1] = (np.sum(temp2) / n2) ** 0.5  # square root!!!!
    temp3 = (image - means[2]) ** 2 * HMRF_resp_matrices[2]
    temp3[mask] = 0
    sds[2] = (np.sum(temp3) / n3) ** 0.5  # square root!!!!
    return



def HMRF_resp(image,hard_label,mask,means,sds,beta):
    HMRF_resp_matrices=[]
    for i in range(3):
        HMRF_resp_matrices.append(np.zeros(image.shape))

    ix, iy = image.shape
    for x in range(ix):
        for y in range(iy):
            if mask[x, y] == False:  # if the point is not in background
                curr_energy=calc_energy(x,y,image,hard_label,mask,means,sds,beta)
                expe=np.vectorize(exp)
                curr_energy*=-1
                exp_energy=expe(curr_energy)
                sum_energy=exp_energy.sum()
                exp_energy/=sum_energy #normalize so that posterior probabilities sum to 1
                for i in range(3):
                    HMRF_resp_matrices[i][x,y]=exp_energy[i]
    return HMRF_resp_matrices

### The EM algorithm of hidden Markov random field
def HMRF_EM(image, mask, hard_label,means,sds,beta):
    log_likelihood_list = []
    log_likelihood_list.append(HMRF_log_likelihood(image,mask,hard_label,means,sds))

    HMRF_resp_matrices,hard_label_new=HMRF_e_step(image, mask, means, sds, beta)
    HMRF_m_step(image, mask, HMRF_resp_matrices, means, sds)
    log_likelihood_list.append(HMRF_log_likelihood(image,mask,hard_label_new,means,sds))
    rounds=1 #count the number of iterations

    while abs((log_likelihood_list[rounds]-log_likelihood_list[rounds-1])/log_likelihood_list[rounds-1])>0.0005: #arbitrary threshold
        HMRF_resp_matrices,hard_label_new = HMRF_e_step(image, mask, means, sds, beta)
        HMRF_m_step(image, mask, HMRF_resp_matrices, means, sds)
        log_likelihood_current=HMRF_log_likelihood(image, mask, hard_label_new, means, sds)
        log_likelihood_list.append(log_likelihood_current)
        rounds+=1
        #Warning! log_likelihood may be trapped in a loop and never escape-->solution: re-run

    hard_label_new=ICM(image, mask, means, sds, beta)
    plot_hard_label(hard_label_new,mask,means,"HMRF EM Converged result")
    print("HMRF_log_likelihood:",log_likelihood_list)


def HMRF_log_likelihood(image,mask,hard_label,means,sds):
    loge = np.vectorize(log)

    def norm0(data): return norm.pdf(data, means[0], sds[0])

    def norm1(data): return norm.pdf(data, means[1], sds[1])

    def norm2(data): return norm.pdf(data, means[2], sds[2])

    norm0 = np.vectorize(norm0)
    norm1 = np.vectorize(norm1)
    norm2 = np.vectorize(norm2)

    mask_clusters = []
    mask_clusters.append(hard_label[np.invert(mask)] == 0)  # the pixels labeled as cluster 0
    mask_clusters.append(hard_label[np.invert(mask)] == 1)
    mask_clusters.append(hard_label[np.invert(mask)] == 2)

    image_no_bg = image[np.invert(mask)]
    log_likelihood = 0
    log_likelihood += np.sum(loge(norm0(image_no_bg[mask_clusters[0]])))
    log_likelihood += np.sum(loge(norm1(image_no_bg[mask_clusters[1]])))
    log_likelihood += np.sum(loge(norm2(image_no_bg[mask_clusters[2]])))
    return log_likelihood
def main():
    #(a)
    brain=misc.imread('brain.png')
    msk=misc.imread('mask.png')

    mask_centerT=msk.astype(bool)
    mask_centerF=np.invert(mask_centerT)
    onlyBrain=brain[mask_centerT].astype(float) #get only the part that has a value larger than 0

    # use kmeans to estimate the mean and sd:
    means,sds,pis,labels=kmeans_estimation(onlyBrain)
    hard_label=np.zeros(brain.shape)
    hard_label[mask_centerT]=labels #use the result from kmeans as original hard labels

    # one iteration of the Iterated Conditional Modes (ICM) algorithm
    hard_label_1round = ICM(brain, mask_centerF, means, sds, 0.5, arbitrary_rounds=1)
    plot_hard_label(hard_label_1round,mask_centerF,means,"After 1 round of ICM")

    # integrate the ICM into the EM algorithm and run it until convergence
    HMRF_EM(brain, mask_centerF, hard_label, means, sds, 0.5)

    # repeat with different parameter β
    HMRF_EM(brain, mask_centerF, hard_label, means, sds, 5)
    HMRF_EM(brain, mask_centerF, hard_label, means, sds, 10)
    HMRF_EM(brain, mask_centerF, hard_label, means, sds, 15)

    plt.show()

main()

