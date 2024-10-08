#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 00:17:53 2023

@author: charlesarnold
"""

# In[9]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from itertools import cycle

# In[9]:
    
df1 = pd.read_excel("R_week1.xlsx")
df2 = pd.read_excel("R_week2.xlsx")
df3 = pd.read_excel("R_week3.xlsx")
df4 = pd.read_excel("SLOPE_ANALYSIS.xlsx")
dfm = pd.read_excel("mean.xlsx")
print(df1.columns)
# In[9]:
def coupled_rk4(dxdt, dydt, x_0, y_0, dt, tf):
    
    def f(xy,t):
        x = xy[0]
        y = xy[1]
        return np.array([dxdt(x,y,t), dydt(x,y,t)], float)
    
    t_arr = np.arange(0, tf, dt)
    x_arr = []
    y_arr = []
    
    x_arr.append(x_0)
    y_arr.append(y_0)
    
    xy = np.array([x_0, y_0], float)
    
    for t in t_arr:
        k1 = f(xy, t) * dt
        k2 = f(xy + 0.5*k1, t + 0.5*dt) * dt
        k3 = f(xy + 0.5*k2, t + 0.5*dt) * dt
        k4 = f(xy + k3, t + dt) * dt
        xy += (k1 + 2*k2 + 2*k3 + k4)/6
        x_arr.append(xy[0])
        y_arr.append(xy[1])
    
    return np.array(x_arr), np.array(y_arr)

# In[9]:
theta1 = df1[ 'A']
omega1 = df1[ 'omega_dim']
print(df1.columns.values.tolist())

# In[9]:


# In[9]:

# In[9]:
    #EXCEL SHEET 1
dt = 0.01
tf = 15

x, y = np.meshgrid(np.linspace(-0.25, 1.8, 11), np.linspace(-0.4, 0.5, 11))
t = np.arange(0, tf, dt)

r = 2

def dxdt(x,y,t): return y

def dydt(x,y,t): return -r*y-x #acceleration

X = dxdt(x,y,t)
Y = dydt(x,y,t)

theta1 = df1[ 'A']
omega1 = df1[ 'omega_dim']
players1 = df1['PLAYER']
print(players1)
plt.figure(dpi=300)



for players1, group in df1.groupby('PLAYER'):
    plt.figure(dpi=300)
    plt.plot(-r*y,y,linestyle='-.')
    plt.plot(np.linspace(-0.25,1.75,11),np.linspace(0,0,11),color ='purple', linestyle='-.')    
    plt.plot([-2,2], [0, 0], color='black', linewidth=0.75, linestyle='--')
    plt.plot([0, 0], [-2, 2], color='black', linewidth=0.75, linestyle='--')
    plt.quiver(x, y, X, Y, color='grey', width=0.003)
    plt.title(f'Pitch Type Phase Portrait') #{players3}:
    #plt.title(r'')
    plt.ylabel(r'$\dot{\theta}$',fontsize=20)
    plt.xlabel(r'$\theta$',fontsize=20)
    plt.xlim(-0.25, 1.8)
    plt.ylim(-0.4, 0.5)
    for row in group.itertuples():
        point = (row.A, row.omega_dim)
        line = plt.plot(*coupled_rk4(dxdt, dydt, point[0], point[1], dt, tf), 
                         label=f'{row.PITCH}' r' ${\theta} = 'f'{round(point[0],2)}$, 'r'$\dot{\theta}$ = 'f'{round(point[1],2)}')
        plt.scatter(point[0], point[1], color=line[0].get_color())
        # Save the plot to a file directory
        plt.legend(loc = 'upper right', fontsize = 13, fancybox=True, framealpha=0.45)
        plt.savefig(f'/Users/charlesarnold/Library/Mobile Documents/com~apple~CloudDocs/Documents/RIT Academics/2023 Year/The Last Spring/Capstone ll/Photos/{players1}_phasew1.png')
    
    plt.show()



# In[9]:
    #EXCEL SHEET 2
    
dt = 0.01
tf = 15

x, y = np.meshgrid(np.linspace(-0.25, 1.25, 11), np.linspace(-0.3, 0.5, 11))
t = np.arange(0, tf, dt)


r = 2

def dxdt(x,y,t): return y

def dydt(x,y,t): return -r*y-x

X = dxdt(x,y,t)
Y = dydt(x,y,t)


theta2 = df2[ 'A']
omega2 = df2[ 'omega_dim']
players2 = df2['PLAYER']
print(players2)
plt.figure(dpi=300)



for players2, group in df2.groupby('PLAYER'):
    plt.figure(dpi=300)
    plt.plot(-r*y,y,linestyle='-.')
    plt.plot(np.linspace(-0.25,1.75,11),np.linspace(0,0,11),color ='purple', linestyle='-.')
    plt.plot([-10, 40], [0, 0], color='black', linewidth=0.75, linestyle='--')
    plt.plot([0, 0], [-30, 30], color='black', linewidth=0.75, linestyle='--')
    plt.quiver(x, y, X, Y, color='grey', width=0.003)
    plt.title(f'Pitch Type Phase Portrait') #{players2}:
    #plt.title(r'')
    plt.ylabel(r'$\dot{\theta}$',fontsize=20)
    plt.xlabel(r'$\theta$',fontsize=20)
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.3, 0.5)
    for row in group.itertuples():
       point = (row.A, row.omega_dim)
       line = plt.plot(*coupled_rk4(dxdt, dydt, point[0], point[1], dt, tf), 
                        label=f'{row.PITCH}' r' ${\theta} = 'f'{round(point[0],2)}$, 'r'$\dot{\theta}$ = 'f'{round(point[1],2)}')
       plt.scatter(point[0], point[1], color=line[0].get_color())
       plt.legend(loc = 'upper right', fontsize = 12.5, fancybox=True, framealpha=0.45)
       plt.savefig(f'/Users/charlesarnold/Library/Mobile Documents/com~apple~CloudDocs/Documents/RIT Academics/2023 Year/The Last Spring/Capstone ll/Photos/{players2}_phasew2.png')
    
    plt.show()

# In[9]:
    #EXCEL SHEET 3
    
dt = 0.01
tf = 15

x, y = np.meshgrid(np.linspace(-0.25, 1.35, 11), np.linspace(-0.3, 0.5, 11))
t = np.arange(0, tf, dt)

r = 2

def dxdt(x,y,t): return y

def dydt(x,y,t): return -r*y-x

X = dxdt(x,y,t)
Y = dydt(x,y,t)


theta3 = df3[ 'A']
omega3 = df3[ 'omega_dim']
players3 = df3['PLAYER']
print(players3)
plt.figure(dpi=300)




for players3, group in df3.groupby('PLAYER'):
    plt.figure(dpi=300)
    plt.plot(-r*y,y,linestyle='-.')
    plt.plot(np.linspace(-0.25,1.75,11),np.linspace(0,0,11),color ='purple', linestyle='-.') 
    plt.plot([-10, 40], [0, 0], color='black', linewidth=0.75, linestyle='--')
    plt.plot([0, 0], [-30, 30], color='black', linewidth=0.75, linestyle='--')
    plt.quiver(x, y, X, Y, color='grey', width=0.003)
    plt.title(f'Pitch Type Phase Portrait') #{players3}:
    #plt.title(r'')
    plt.ylabel(r'$\dot{\theta}$',fontsize=20)
    plt.xlabel(r'$\theta$',fontsize=20)
    plt.xlim(-0.25, 1.35)
    plt.ylim(-0.3, 0.5)
    for row in group.itertuples():
        point = (row.A, row.omega_dim)
        line = plt.plot(*coupled_rk4(dxdt, dydt, point[0], point[1], dt, tf), 
                         label=f'{row.PITCH}' r' ${\theta} = 'f'{round(point[0],2)}$, 'r'$\dot{\theta}$ = 'f'{round(point[1],2)}')
        plt.scatter(point[0], point[1], color=line[0].get_color())
        plt.legend(loc = 'upper right', fontsize = 12, fancybox=True, framealpha=0.45)
        plt.savefig(f'/Users/charlesarnold/Library/Mobile Documents/com~apple~CloudDocs/Documents/RIT Academics/2023 Year/The Last Spring/Capstone ll/Photos/{players3}_phasew3.png')
    
    plt.show()
         
# In[9]:

# In[9]:
    #EXCEL 1
tf = 0.5
t = np.linspace(0, tf, 50)

lines = ["o","x","s","^"]
linecycler = cycle(lines)

prev_val = None
for index, row in df1.iterrows():
    player = row['PLAYER']
    pitch_type = row['PITCH']
    A = row['A']
    B = row['B']
    v = row['Hand_v']
    alpha = row['damp']
    if player != prev_val:
        if prev_val is not None:
            plt.legend(fontsize=16)
            plt.draw()
            plt.pause(0.4)
        plt.figure()
        plt.title(f'{player} Pitch Type Comparasion') #{player}:
        plt.xlabel(r'$t$', fontsize=20)
        plt.ylabel(r'$\theta(t)$',fontsize=20)
        prev_val = player
    theta = (A + B*t) * np.exp(-alpha*t)
    plt.scatter(t, theta, marker = next(linecycler) ,label=f' {pitch_type}: ' r'$\alpha$=' f'{round(alpha,2)}'r' $v_H$ = 'f'{v}mph')
    plt.legend(fontsize=14)
    plt.savefig(f'/Users/charlesarnold/Library/Mobile Documents/com~apple~CloudDocs/Documents/RIT Academics/2023 Year/The Last Spring/Capstone ll/Photos/{player}_dampw1.png')


# In[9]:
    
    #EXCEL 2

tf = 0.45
t = np.linspace(0, tf, 50)

lines = ["o","x","s","^"]
linecycler = cycle(lines)

prev_val = None
for index, row in df2.iterrows():
    player = row['PLAYER']
    pitch_type = row['PITCH']
    A = row['A']
    B = row['B']
    v = row['Hand_v']
    alpha = row['damp']
    if player != prev_val:
        if prev_val is not None:
            plt.legend(fontsize=16)
            plt.draw()
            plt.pause(0.4)
        plt.figure()
        plt.title(f'Pitch Type Comparasion') #{player}:
        plt.xlabel(r'$t$', fontsize=20)
        plt.ylabel(r'$\theta(t)$',fontsize=20)
        prev_val = player
    theta = (A + B*t) * np.exp(-alpha*t)
    plt.scatter(t, theta, marker = next(linecycler) ,label=f'{pitch_type}: ' r'$\alpha$' f'={round(alpha,2)}')
    plt.legend(fontsize=14)
    plt.savefig(f'/Users/charlesarnold/Library/Mobile Documents/com~apple~CloudDocs/Documents/RIT Academics/2023 Year/The Last Spring/Capstone ll/Photos/{player}_dampw2.png')


# In[9]:
    #EXCEL 3

#Measure damping effects by player

tf = 0.5
t = np.linspace(0, tf, 50)

lines = ["o","x","s","^"]
linecycler = cycle(lines)


prev_val = None
for index, row in df3.iterrows():
    player = row['PLAYER']
    pitch_type = row['PITCH']
    A = row['A']
    B = row['B']
    v = row['Hand_v']
    alpha = row['damp']
    if player != prev_val:
        if prev_val is not None:
            plt.legend(fontsize=16)
            plt.draw()
            plt.pause(0.4)
        plt.figure()
        plt.title(f'Pitch Type Comparasion') #{player}:
        plt.xlabel(r'$t$', fontsize=20)
        plt.ylabel(r'$\theta(t)$',fontsize=20)
        prev_val = player
    theta = (A + B*t) * np.exp(-alpha*t)
    plt.scatter(t, theta, marker = next(linecycler) ,label=f'{pitch_type}: ' r'$\alpha$' f'={round(alpha,2)}' r' $v_H$ = 'f'{v}mph')
    plt.legend(fontsize=14)
    plt.savefig(f'/Users/charlesarnold/Library/Mobile Documents/com~apple~CloudDocs/Documents/RIT Academics/2023 Year/The Last Spring/Capstone ll/Photos/{player}_dampw3.png')





# In[9]:# In[9]:
    #PLAYER COMPARASION
tf = 0.5
t = np.linspace(0, tf, 50)
lines = ["o","x","s","^"]
linecycler = cycle(lines)

pitch_type = 'FS' # specify the pitch type you want to compare


plt.figure()
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$\theta(t)$', fontsize=18)

for player in df4['PLAYER'].unique():
    df_player = df4[df4['PLAYER'] == player]
    df_pitch = df_player[df_player['PITCH'] == pitch_type] # match pitch to pitch type in dataset
    if not df_pitch.empty:
        A = df_pitch['A'].values[0]
        B = df_pitch['B'].values[0]
        v = df_pitch['Hand_v'].values[0]
        alpha = df_pitch['alpha'].values[0]
        theta = (A + B*t) * np.exp(-alpha*t)
        #theta = theta_i/theta_i[0]
        plt.scatter(t, theta, marker = next(linecycler) ,label=f'{pitch_type}:' r' $\alpha'f'={round(alpha,2)}$'r' $v_H$ = 'f'{v}mph')
plt.legend(fontsize = 12, fancybox=True, framealpha=0.45)
plt.savefig(f'/Users/charlesarnold/Library/Mobile Documents/com~apple~CloudDocs/Documents/RIT Academics/2023 Year/The Last Spring/Capstone ll/Photos/damp_comparasion.png')

plt.show()
# In[9]:# In[9]:


# In[9]:# In[9]:

    #MEAN PLOT FOR ANOVA

# Calculate means and pitch names
mean, std = dfm.groupby('PITCH')['damp'].agg(['mean', 'std']).values.T
Pitches = ["Breaking Ball","FastBall"]
# Create the bar plot
plt.bar(Pitches, mean, color=['blue','brown'], yerr=std, edgecolor = "black")

# Add x and y axis labels
plt.xlabel('Pitch Type', fontsize=18)
plt.ylabel(r'$\alpha (\frac{1}{s})$', fontsize=18)

# Add a title

# Add value labels to the bars
for i, v in enumerate(mean):
    plt.text(i, v, f'{v:.2f}'r'$\pm$'f'{std[i]:.2f}'r' $\frac{1}{S}$', ha='left', va='bottom')

# Show the plot
plt.savefig(f'/Users/charlesarnold/Library/Mobile Documents/com~apple~CloudDocs/Documents/RIT Academics/2023 Year/The Last Spring/Capstone ll/Photos/anova_mean.png')
plt.show()
# Print the mean for each pitch



# In[9]:# In[9]:

    

# In[9]:# In[9]:

  
# In[9]:
    
tf = 1
t = np.linspace(0,tf,30)
def crit_damp(A,B, alpha, t):
    theta = (A+B*t)*exp(-alpha*t)
    return theta

A1 = df1[ 'A']
A2 = df2[ 'A']
A3 = df3[ 'A']

B1 = df1[ 'B']
B2 = df2[ 'B']
B3 = df3[ 'B']

alpha1 = df1[ 'alpha']
alpha2 = df2[ 'alpha']
alpha3 = df3[ 'alpha']

plt.figure()

theta_1 = crit_damp(A1[i],B1[i],alpha1[i],t)
theta_2 = crit_damp(A1[i],B1[i],alpha1[i],t)

plt.plot(t,theta_1)
plt.plot(t,theta_2)     

# In[9]:# In[9]:
'''    #MEASURE DAMPING BY PITCH
tf = 0.5
t = np.linspace(0, tf, 50)
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

pitch_type = 'CH' # specify the pitch type you want to compare

plt.figure()
plt.title(f'{pitch_type} Comparison for ALL')
plt.xlabel(f'$t$')
plt.ylabel(r'$\theta$')

for player in df1['PLAYER'].unique():
    df_player = df1[df1['PLAYER'] == player]
    df_pitch = df_player[df_player['PITCH'] == pitch_type] # match pitch to pitch type in dataset
    if not df_pitch.empty:
        A = df_pitch['A'].values[0]
        B = df_pitch['B'].values[0]
        alpha = df_pitch['alpha'].values[0]
        theta = (A + B*t) * np.exp(-alpha*t)
        #theta = theta_i/theta_i[0]
        plt.plot(t, theta, linestyle = next(linecycler) ,label=r'$\theta_i= $'f'{round(theta[0],1)}' r' $\alpha'f'={round(alpha,2)}$')
for player in df2['PLAYER'].unique():
    df_player = df2[df2['PLAYER'] == player]
    df_pitch = df_player[df_player['PITCH'] == pitch_type] # match pitch to pitch type in dataset
    if not df_pitch.empty:
        A = df_pitch['A'].values[0]
        B = df_pitch['B'].values[0]
        alpha = df_pitch['alpha'].values[0]
        theta = (A + B*t) * np.exp(-alpha*t)
        #theta = theta_i/theta_i[0]
        plt.plot(t, theta, linestyle = next(linecycler) ,label=r'$\theta_i= $'f'{round(theta[0],1)}' r' $\alpha'f'={round(alpha,2)}$')
for player in df3['PLAYER'].unique():
    df_player = df3[df3['PLAYER'] == player]
    df_pitch = df_player[df_player['PITCH'] == pitch_type] # match pitch to pitch type in dataset
    if not df_pitch.empty:
        A = df_pitch['A'].values[0]
        B = df_pitch['B'].values[0]
        alpha = df_pitch['alpha'].values[0]
        theta = (A + B*t) * np.exp(-alpha*t)
        #theta = theta_i/theta_i[0]
        plt.plot(t, theta, linestyle = next(linecycler) ,label=r'$\theta_i= $'f'{round(theta[0],1)}' r' $\alpha'f'={round(alpha,2)}$')
#plt.legend()
plt.show()'''
    

# In[9]:# In[9]:

    

# In[9]:# In[9]:
    

# In[9]:# In[9]:

    

# In[9]:# In[9]:

    

# In[9]:# In[9]:

    

# In[9]:# In[9]:

    

# In[9]:# In[9]:

    

# In[9]: