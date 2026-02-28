#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:52:37 2025

@author: yao
"""

from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = 12,9

# ---- Find missing grid ---
#Zlist = ["1","3","10","31","100"]
Zlist = ["100"]

fig = plt.figure()
ax = plt.subplot(111)

n = 0
n_err = 0
for i in range(len(Zlist)):
    for Teq in range(6,10,1):
        for Tint in range(1,7,1):
            for grav in range(1,6,1):
                n = n+1
                fname = "PT_Z"+Zlist[i]+"_Teq"+str(Teq).zfill(2)+"_Tin"+str(Tint).zfill(3)+"_g"+str(grav*100)+".dat"                
                if( os.path.exists(fname) ):
                    #print(fname)
                    data = np.loadtxt(fname,skiprows=4)
                    ax.plot(data[:,1],data[:,2])
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([100,10000])
ax.set_xlim([1e4,1e-6])  
ax.set_ylabel('T (K)',fontsize=24)
ax.set_xlabel('P (bar)',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

#plt.savefig("TPs_100x.pdf", format="pdf", bbox_inches="tight")
