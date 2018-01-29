"""Image Registration"""

import numpy as np
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
from copy import deepcopy

# a) L2 cost function to evaluate registration quality
def L2_loss(image1,image2): #image1,image2 are np.arrays
    image1_float=image1.astype(float)
    image2_float=image2.astype(float)
    loss_fun=np.mean((image1_float-image2_float)**2)
    return loss_fun

# b) Write a method that can translate and rotate an input image by arbitrary amounts
def translate_rotate(image,shift_vector): #shift_vector is a vector (list or 1D array) applying to each axis. shift_vector=[angle,shift_x,shift_y]
    angle = shift_vector[0] #counter-clockwise
    shift_x = shift_vector[1] #vertical (+:)
    shift_y = shift_vector[2] #horizontal (+:right,-:left)
    rotated=ndimage.rotate(image,angle=angle,reshape=False)
    rotated_shifted=ndimage.shift(rotated,shift=[shift_y,shift_x])
    return rotated_shifted


def calc_loss(var,x,img_ref,img_mov):
    if var=="angle": return L2_loss(translate_rotate(img_mov,[x,0,0]),img_ref)
    elif var=="shift_x": return L2_loss(translate_rotate(img_mov,[0,x,0]),img_ref)
    elif var=="shift_y": return L2_loss(translate_rotate(img_mov,[0,0,x]),img_ref)

# c-a) simple optimization method based on the golden ratio rule
def find_bracket(img_ref,img_mov,var): #var can be either "angle","shift_x", or "shift_y"
    #image1 is fixed, image2 is floating
    w=(3-5**0.5)/2
    end1 = 0  # fix at 0
    mid = 10 * w
    loss_end1=calc_loss(var,end1,img_ref,img_mov)
    loss_mid = calc_loss(var,mid,img_ref,img_mov)
    while loss_end1 == loss_mid:  # iterate until loss_end1!=loss_mid
        mid += 1
        loss_mid = calc_loss(var, mid, img_ref, img_mov)
    if loss_end1 > loss_mid:
        end2 = mid * (1 / w)  # Golden ratio
        loss_end2 = calc_loss(var, end2, img_ref, img_mov)
        while loss_end1 <= loss_mid or loss_end2 <= loss_mid:  # not a bracket, keep searching
            mid = end2
            end2 = mid * (1 / w)
            if (var=="angle" and end2 > 180) or (var=="shift_x" and end2>img_mov.shape[0]) or (var=="shift_y" and end2>img_mov.shape[1]):
                print("Search failed... Positive limit reached")
                return 0, 0, 0
            loss_mid = loss_end2
            loss_end2 = calc_loss(var, end2, img_ref, img_mov)
        return end1, mid, end2  # in the order of: small, middle, large

    else:
        end1, mid = mid, end1  # exchange to make loss_end1>loss_mid. end1 is fixed in the following steps
        end2 = mid - abs(mid - end1) * (1 - w) / w  # Golden ratio (the other direction)
        loss_end2 = calc_loss(var, end2, img_ref, img_mov)
        while loss_end1 <= loss_mid or loss_end2 <= loss_mid:  # not a bracket, keep searching
            mid = end2
            end2 = mid - abs(mid - end1) * (1 - w) / w  # Golden ratio (the other direction)
            if (var=="angle" and end2 < -180) or (var=="shift_x" and end2 < img_mov.shape[0]*(-1)) or (var=="shift_y" and end2<img_mov.shape[1]*(-1)):
                print("Search failed... Positive limit reached")
                print(end2)
                return 0, 0, 0
            loss_mid = loss_end2
            loss_end2 = calc_loss(var, end2, img_ref, img_mov)
        return end2, mid, end1  # in the order of: small, middle, large

def find_bracket_general(img_ref,img_mov,vector): #generalized. vector is a np.array representing the direction of change
    original_v=deepcopy(vector)

    w = (3 - 5 ** 0.5) / 2
    end1=np.zeros(3) #fixed
    mid=vector*w+end1
    loss_end1 = L2_loss(img_ref,translate_rotate(img_mov,end1))
    loss_mid = L2_loss(img_ref,translate_rotate(img_mov,mid))
    while loss_end1 == loss_mid:  # iterate until loss_end1!=loss_mid
        mid += 1
        loss_mid = L2_loss(img_ref, translate_rotate(img_mov, mid))
    if loss_end1 > loss_mid:
        end2 = mid * (1 / w)  # Golden ratio
        loss_end2 = L2_loss(img_ref, translate_rotate(img_mov, end2))
        while loss_end1 <= loss_mid or loss_end2 <= loss_mid:  # not a bracket, keep searching
            mid = deepcopy(end2)
            end2 = mid * (1 / w) #* returns a copy, so no need to deepcopy
            if end2[0]>180 or end2[1]>img_mov.shape[0] or end2[2]>img_mov.shape[1] \
                    or end2[0]<-180 or end2[1]<img_mov.shape[0]*(-1) or end2[2]<img_mov.shape[1]*(-1):
                print("Search failed... limit reached1")
                return 0, 0, 0
            loss_mid = loss_end2
            loss_end2 = L2_loss(img_ref, translate_rotate(img_mov, end2))
        return end1, mid, end2  # in the order of: small, middle, large
    else:
        temp=deepcopy(mid)
        mid=deepcopy(end1)
        end1=temp # exchange to make loss_end1>loss_mid. end1 is fixed in the following steps
        loss_end1 = L2_loss(img_ref, translate_rotate(img_mov, end1))
        loss_mid = L2_loss(img_ref, translate_rotate(img_mov, mid))
        end2 = mid + (mid - end1) * (1 - w) / w  # Golden ratio (the other direction)
        loss_end2 = L2_loss(img_ref, translate_rotate(img_mov, end2))
        while loss_end1 <= loss_mid or loss_end2 <= loss_mid:  # not a bracket, keep searching
            mid = deepcopy(end2)
            end2 = mid + (mid - end1) * (1 - w) / w  # Golden ratio (the other direction)
            if end2[0] > 180 or end2[1] > img_mov.shape[0] or end2[2] > img_mov.shape[1] \
                    or end2[0] < -180 or end2[1] < img_mov.shape[0] * (-1) or end2[2] < img_mov.shape[1] * (-1):
                print("Search failed... limit reached2")
                print("end1, mid, end2",end1, mid, end2)
                print("original_v",original_v)
                return 0, 0, 0
            loss_mid = loss_end2
            loss_end2 = L2_loss(img_ref, translate_rotate(img_mov, end2))
        return end2, mid, end1  # in the order of: small, middle, large

# c-b)
def optimize_1D(image_ref,image_mov,var): #end1,mid,end2 are the coordinates of the bracket in the order of: small, middle, large
    w = (3 - 5 ** 0.5) / 2
    end1, mid, end2=find_bracket(image_ref,image_mov,var)
    while end2-end1>1: #bracket is larger than 1
        interval1 = abs(mid - end1)
        interval2 = abs(end2 - mid)
        loss_mid = calc_loss(var, mid, image_ref, image_mov)
        if interval1>interval2:
            new=mid-interval1*w
            loss_new=calc_loss(var, new, image_ref, image_mov)
            while loss_new==loss_mid: #make sure loss_new!=loss_mid
                new+=1
                loss_new = calc_loss(var, new, image_ref, image_mov)
            if loss_new>loss_mid:
                end1=new
            else:
                end2=mid
                mid=new
        else:
            new = mid + interval2 * w
            loss_new = calc_loss(var, new, image_ref, image_mov)
            while loss_new == loss_mid:  # make sure loss_new!=loss_mid
                new += 1
                loss_new = calc_loss(var, new, image_ref, image_mov)
            if loss_new > loss_mid:
                end2 = new
            else:
                end1 = mid
                mid = new
    return mid

def optimize_1D_general(image_ref,image_mov,vector): #generalized. vector is a np.array representing the direction of change
    w = (3 - 5 ** 0.5) / 2
    end1, mid, end2=find_bracket_general(image_ref,image_mov,vector)
    while np.sum((end2 - end1)**2)>0.1: #bracket is larger than 1
        interval1 = np.sum((mid - end1)**2)**0.5
        interval2 = np.sum(((end2 - mid))**2)**0.5
        loss_mid = L2_loss(image_ref, translate_rotate(image_mov, mid))
        if interval1>interval2:
            new=mid-(mid - end1)*w
            loss_new = L2_loss(image_ref, translate_rotate(image_mov, new))

            if loss_new>loss_mid:
                end1=deepcopy(new)
            else:
                end2=deepcopy(mid)
                mid=deepcopy(new)
        else:
            new = mid + (end2 - mid) * w
            loss_new = L2_loss(image_ref, translate_rotate(image_mov, new))
            if loss_new > loss_mid:
                end2 = deepcopy(new)
            else:
                end1 = deepcopy(mid)
                mid = deepcopy(new)
    return mid

# c-c)
def Powell(image_ref,image_mov): #Powell‘s Method
    axis_list=[]
    axis_list.append(np.array([10,0,0],dtype=float))
    axis_list.append(np.array([0,10,0],dtype=float))
    axis_list.append(np.array([0,0,10],dtype=float))
    loss_before=L2_loss(image_ref,image_mov)
    min_shift_list=[0]*3
    operation_vector=np.zeros(3) #sum of all direction vectors
    loss_after=[0]*3
    total_loss_old =1

    for i in range(len(axis_list)):
        min_shift_list[i] = optimize_1D_general(image_ref, image_mov, axis_list[i])
        operation_vector+=min_shift_list[i]
        # image_mov = translate_rotate(image_mov, min_shift_list[i]) #try not change the image_mov, but rather record the direction vectors
        loss_after[i] = L2_loss(image_ref, translate_rotate(image_mov,operation_vector))

    new_direction = min_shift_list[0] + min_shift_list[1] + min_shift_list[
        2]  # new direction is the combination of 3 directions
    new_direction/=0.1*np.sum(new_direction) #normalize
    total_loss_new = loss_after[2] - loss_before
    loss_after[2] -= loss_after[1]
    loss_after[1] -= loss_after[0]
    loss_after[0] -= loss_before
    axis_list[loss_after.index(min(loss_after))] = deepcopy(
        new_direction)  # replace the most successful direction with the new direction

    while abs((total_loss_new-total_loss_old)/total_loss_old)>0.0001: #arbitrary threshold of convergence
        total_loss_old=total_loss_new
        for i in range(len(axis_list)):
            min_shift_list[i]=optimize_1D_general(image_ref,translate_rotate(image_mov,operation_vector), axis_list[i]) #use the new moving image as the new starting point
            operation_vector += min_shift_list[i]
            loss_after[i] = L2_loss(image_ref, translate_rotate(image_mov, operation_vector))

        new_direction=min_shift_list[0]+min_shift_list[1]+min_shift_list[2] #new direction is the combination of 3 directions
        total_loss_new=loss_after[2]-loss_before
        loss_after[2]-=loss_after[1]
        loss_after[1]-=loss_after[0]
        loss_after[0]-=loss_before

        axis_list[loss_after.index(max(loss_after))]=deepcopy(new_direction) #replace the most successful direction with the new direction
    return operation_vector


def main():
    #a)
    axial = misc.imread('axial.png')
    axial_t = misc.imread('axial_transformed.png')
    print("loss between axial and axial_transformed",L2_loss(axial,axial_t))

    #b)
    plt.figure()
    plt.title("Original axial vs. axial rotated 90 degrees")
    plt.imshow(axial,alpha=0.5,cmap='gray')
    plt.imshow(translate_rotate(axial,[90,0,0]),alpha=0.5,cmap='gray')


    loss_list_angle=[]
    index=[]
    for i in range(-180,180,10):
        loss_list_angle.append(L2_loss(axial,translate_rotate(axial_t,[i,0,0])))
        index.append(i)
    plt.figure()
    plt.xlabel("angle")
    plt.ylabel("loss function value")
    plt.title("Change of loss function")
    plt.plot(index,loss_list_angle)

    loss_list_x = []
    index = []
    for i in range(-255, 256, 10):
        loss_list_x.append(L2_loss(axial, translate_rotate(axial_t, [0,i,0])))
        index.append(i)
    plt.figure()
    plt.xlabel("shift_x")
    plt.ylabel("loss function value")
    plt.title("Change of loss function")
    plt.plot(index, loss_list_x)

    loss_list_y = []
    index = []
    for i in range(-255, 256, 10):
        loss_list_y.append(L2_loss(axial, translate_rotate(axial_t, [0,0,i])))
        index.append(i)
    plt.figure()
    plt.xlabel("shift_y")
    plt.ylabel("loss function value")
    plt.title("Change of loss function")
    plt.plot(index, loss_list_y)

    # c-a) Example of finding the initial bracket in the dimension of angle:
    end1, mid, end2=find_bracket(axial, axial_t, "angle")
    print("Angle bracket:",end1,mid,end2)
    print("Corresponding loss values:",calc_loss("angle",end1,axial,axial_t),calc_loss("angle",mid,axial,axial_t),calc_loss("angle",end2,axial,axial_t))

    # c-b) Example of optimizing in the dimension of angle:
    min_shift=optimize_1D(axial,axial_t,"angle")
    print("Optimized angle:", min_shift)
    print("Corresponding loss value:", L2_loss(axial,translate_rotate(axial_t,[min_shift,0,0])))


    # c-c) Using the Powell's method to optimize:
    oper_vec = Powell(axial, axial_t)  # optimal operation vector
    print(oper_vec)
    new_img=translate_rotate(axial_t,oper_vec)
    plt.figure()
    plt.title("Powell result")
    plt.imshow(new_img, cmap='gray',alpha=0.5)
    plt.imshow(axial,cmap='gray',alpha=0.5)
    print("Corresponding loss value:",L2_loss(axial,new_img))

    plt.show()


main()
