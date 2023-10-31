
# coding: utf-8

# # Script to create and save sampleframe and triframe to directory


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from TriFuncs import * #custom functions for following analysis
#get_ipython().run_line_magic('matplotlib', '')

#get variables from main script
from __main__ import *

#set path to access previously stored .mat files, etc. use scipy.io.loadmat()
datepath = filepath + date + '/'
savepath = datepath

#import arduino behavioral data and their timestamps
a = open(datepath+'arduinoraw'+str(filecount)+'.txt','r')
ardtext= a.read().splitlines()
a.close


#Make dataframe with all data organized by sample number
a = np.tile(0,(len(ardtext),8))
data = np.full_like(a, np.nan, dtype=np.float16) #make a sample number x variable number array of nans
block = 1
tris = 0
trial = 1
data[0,7] = 0
pA = -100
pB = -100
pC = -100
#get sample number of current trial to align phot data to port entries
#data = list(data)
#fill in reward and port info. Ports are coded by 1:A,2:B,3:C
for i in range(len(ardtext)):
    new = False
    if 'A Harvested' in ardtext[i] or "delivered at port A" in ardtext[i]:
        data[tris,0] = 0
        data[tris,1] = 1
        data[tris+1,7] = 0
        trial += 1
        tris += 1
    elif 'B Harvested' in ardtext[i] or "delivered at port B" in ardtext[i]:
        data[tris,0] = 1
        data[tris,1] = 1
        data[tris+1,7] = 1
        trial += 1
        tris += 1
    elif 'C Harvested' in ardtext[i] or "delivered at port C" in ardtext[i]:
        data[tris,0] = 2
        data[tris,1] = 1
        data[tris+1,7] = 2
        trial += 1
        tris += 1
    elif 'o Reward port A' in ardtext[i]:
        data[tris,0] = 0
        data[tris,1] = 0
        data[tris+1,7] = 0
        trial += 1
        tris += 1
    elif 'o Reward port B' in ardtext[i]:
        data[tris,0] = 1
        data[tris,1] = 0
        data[tris+1,7] = 1
        trial += 1
        tris += 1
    elif 'o Reward port C' in ardtext[i]:
        data[tris,0] = 2
        data[tris,1] = 0
        data[tris+1,7] = 2
        trial += 1
        tris += 1
    elif "Block" in ardtext[i]:
        block = int(ardtext[i][-1]) + 1
        new = True
        pA = ardtext[i+1][3:]
        pB = ardtext[i+2][3:]
        pC = ardtext[i+3][3:]
    data[tris,2] = block #block number
    data[tris,3] = pA
    data[tris,4] = pB
    data[tris,5] = pC
    if new == True:
        trial = 0 #reset trials within block
    data[tris,6] = trial

data = data[:tris]

#if only dLight
tridat = pd.DataFrame(data,columns = ['port','rwd','block','pA','pB','pC',\
    'tri','fromP'])

tridat.loc[:,"rat"] = animal

tridat.to_csv(savepath+'triframe_brief.csv')


def plot_prob_matching(tridat):
    x1 = np.linspace(0,len(tridat),len(tridat))
    A1 = tridat.rwd.loc[tridat.port==0] + 2
    yA1 = np.zeros(len(tridat))
    yA1[A1.index.values] = A1
    B1 = tridat.rwd.loc[tridat.port==1] + 2
    yB1 = np.zeros(len(tridat))
    yB1[B1.index.values] = B1
    C1 = tridat.rwd.loc[tridat.port==2] + 2
    yC1 = np.zeros(len(tridat))
    yC1[C1.index.values] = C1
    
    #frequency of each port visit over time
    window = 10
    #wintype = 'boxcar'
    
    tridat['ChooseA'] = np.where(tridat.port==0,1,0)
    tridat['ChooseB'] = np.where(tridat.port==1,1,0)
    tridat['ChooseC'] = np.where(tridat.port==2,1,0)
    
    bdt = plt.figure(figsize = (18,12))
    plt.suptitle('Within-Session Probability Matching',fontweight='bold',fontsize = 26)
    ax4 = plt.subplot2grid((18,1),(3,0),colspan = 1, rowspan =15)
    ax4.plot(x1,tridat.ChooseA.rolling(window,min_periods=1).sum().divide(window),label = 'A',alpha=0.8)
    ax4.plot(x1,tridat.ChooseB.rolling(window,min_periods=1).sum().divide(window),label = 'B',alpha=0.8)
    ax4.plot(x1,tridat.ChooseC.rolling(window,min_periods=1).sum().divide(window),label = 'C',alpha=0.8)
    ax4.set_ylabel('Port Visits/trial',fontsize=20,fontweight='bold')
    ax4.set_ylim(0,.7)
    #ax4.set_xlabel('Time (min)',fontsize=20,fontweight='bold')
    ax4.legend(bbox_to_anchor=(0.9, 1.4), loc=2, borderaxespad=0.)
    ax1 = plt.subplot2grid((18,1),(0,0),colspan = 1, rowspan =1,sharex=ax4)
    ax1.bar(x1,yA1,color = 'blue')
    ax1.axis('off')
    ax2 = plt.subplot2grid((18,1),(1,0),colspan = 1, rowspan =1,sharex=ax4)
    for b in tridat.block.unique():
        xstart = int((max(tridat.loc[tridat.block==b].index)+1))
        ind = max(tridat.loc[tridat.block==b].index)
        xmin = int(min(tridat.loc[tridat.block==b].index))
        xmid = int(np.mean(tridat.loc[tridat.block==b].index))#int(xmin+(xstart-xmin)/2)
        print(xmid)
        if b==1:
            ax4.axvline(x=xstart,color ='r',linestyle='--', label = 'Block Change')
        else:
            ax4.axvline(x=xstart,color ='r',linestyle='--')
        plt.text(xmid-12,8,str(int(tridat.pA[ind-1]))+': ',fontsize='xx-large',fontweight='bold',color = 'b')
        plt.text(xmid,8,str(int(tridat.pB[ind-1]))+': ',fontsize='xx-large',fontweight='bold',color = 'orange')
        plt.text(xmid+12,8,str(int(tridat.pC[ind-1])),fontsize='xx-large',fontweight='bold',color = 'g')
    ax4.legend()
    ax2.bar(x1,yB1,color='orange')
    ax2.axis('off')
    ax3 = plt.subplot2grid((18,1),(2,0),colspan = 1, rowspan =1,sharex=ax4)
    ax3.bar(x1,yC1,color='g')
    ax3.axis('off')
    return bdt


