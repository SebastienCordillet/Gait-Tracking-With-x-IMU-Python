import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot, q2euler
import pyquaternion
import ximu_python_library.xIMUdataClass as xIMU
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main import getIMU, selTime
from plotFrames import labFrame, addFrames, addOrigin

import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default='browser'

# filePath = 'datasets/straightLine'
# startTime = 6
# stopTime = 26
# samplePeriod = 1/256

# filePath = 'datasets/stairsAndCorridor'
# startTime = 5
# stopTime = 53
# samplePeriod = 1/256

filePath = 'data_marche.c3d'




startTime = 0
stopTime = 60
# samplePeriod = 1/256


# def main():
    
    
    # xIMUdata = xIMU.xIMUdataClass(filePath, 'InertialMagneticSampleRate', 1/samplePeriod)
    
lfoot=getIMU(filePath, chanelName='Right_Tibialis Anterior', mag=False)
# lfoot=getIMU(filePath, chanelName='Left_Semitendinosus', mag=False)


# rate=lfoot.attrs['rate']
rate=120
samplePeriod=1/rate
# print(lfoot) # OK
lfoot=lfoot.meca.time_normalize(time_vector=selTime([startTime,stopTime],rate))


# print(lfoot)
time = lfoot['time'].values
gyrX = lfoot.sel(channel='GYRO_X').values
gyrY = lfoot.sel(channel='GYRO_Y').values
gyrZ = lfoot.sel(channel='GYRO_Z').values
accX = lfoot.sel(channel='ACC_X').values
accY = lfoot.sel(channel='ACC_Y').values
accZ = lfoot.sel(channel='ACC_Z').values


figAcc=go.Figure()
figAcc.add_trace(go.Scatter(x=time, y=accX, name='accX'))
figAcc.add_trace(go.Scatter(x=time, y=accY, name='accY'))
figAcc.add_trace(go.Scatter(x=time, y=accZ, name='accZ'))
figGyr=go.Figure()
figGyr.add_trace(go.Scatter(x=time, y=gyrX, name='gyrX'))
figGyr.add_trace(go.Scatter(x=time, y=gyrY, name='gyrY'))
figGyr.add_trace(go.Scatter(x=time, y=gyrZ, name='gyrZ'))


# Compute accelerometer magnitude
acc_mag = np.sqrt(accX*accX+accY*accY+accZ*accZ)

# LP filter accelerometer data
filtCutOff = 5
b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'lowpass')
acc_magFilt = signal.filtfilt(b, a, acc_mag-1, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

filtCutOff = 2
b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'lowpass')
accX = signal.filtfilt(b, a, accX, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
accY = signal.filtfilt(b, a, accY, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
accZ = signal.filtfilt(b, a, accZ, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

gyrX = signal.filtfilt(b, a, gyrX, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
gyrY = signal.filtfilt(b, a, gyrY, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
gyrZ = signal.filtfilt(b, a, gyrZ, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))




# Threshold detection
stationary = acc_magFilt < 0.20

# plt.figure()
# plt.plot(time,acc_magFilt)
# plt.plot(time,stationary)
# # plt.legend()
# plt.show

# fig=go.Figure()
# fig.add_trace(go.Scatter(x=time, y=acc_magFilt, name='accNorm'))
# fig.add_trace(go.Scatter(x=time,y=stationary.astype(int), name='IsStationnary'))
# plot(fig)

# Compute orientation
quat  = np.zeros((time.size, 4), dtype=np.float64)

# initial convergence
initPeriod = 0.5
indexSel = time<=time[0]+initPeriod
# gyr=np.zeros(3, dtype=np.float64)
acc = np.array([np.mean(accX[indexSel]), np.mean(accY[indexSel]), np.mean(accZ[indexSel])])
mahony = ahrs.filters.Mahony(Kp=1, Ki=0,KpInit=1, frequency=1/samplePeriod)
tilt = ahrs.filters.Tilt()
angular_rate = ahrs.filters.AngularRate(Dt=1/rate)
# -->initialize from mahony
# q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
# for i in range(0, 4000):
#     q = mahony.updateIMU(q, gyr=gyr, acc=acc)

# -->initialize from tilt
quat[0,:]=tilt.estimate(acc)

for t in range(1,time.size):
    acc = np.array([accX[t],accY[t],accZ[t]])
    gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
    q= quat[t-1,:]
    if(stationary[t]): 
        quat[t,:]=q
    else:
        quat[t,:]=angular_rate.update(q, gyr)

# plt.figure()
# plt.plot(np.array([q2euler(q)*180/np.pi for q in quat]))
# plt.legend(['X','Y','Z'])
# plt.show()


EA=np.array([q2euler(q)*180/np.pi for q in quat])
fig2=go.Figure()
fig2.add_trace(go.Scatter(x=time, y=EA[:,0], name='roll'))
fig2.add_trace(go.Scatter(x=time, y=EA[:,1], name='pitch'))
fig2.add_trace(go.Scatter(x=time, y=EA[:,2], name='yaw'))
# plot(fig2)

# fig=px.line(np.array([q2euler(q)*180/np.pi for q in quat]))


    # quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
# For all data
# for t in range(0,time.size):
#     if(stationary[t]):
#         mahony.Kp = 0.5
#     else:
#         mahony.Kp = 0
#     gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
#     acc = np.array([accX[t],accY[t],accZ[t]])
#     quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)

# -------------------------------------------------------------------------
# Compute translational accelerations

# Rotate body accelerations to Earth frame
fig3=go.Figure()
fig3.add_trace(go.Scatter(x=time,y= accX*9.81, line=dict(color='red', dash='dot'), name='X raw'))
fig3.add_trace(go.Scatter(x=time,y= accY*9.81, line=dict(color='green', dash='dot'), name='Y raw'))
fig3.add_trace(go.Scatter(x=time,y= accZ*9.81, line=dict(color='blue', dash='dot'), name='Z raw'))


# # fig = plt.figure(figsize=(10, 5))
# plt.plot(time,accX*9.81,lines(dict(color='r', dash='dot')))
# plt.plot(time,accY*9.81,c='g',linestyle=':',linewidth=0.5)
# plt.plot(time,accZ*9.81,c='b',linestyle=':',linewidth=0.5)

acc = []
for x,y,z,q in zip(accX,accY,accZ,quat):
    # acc.append(q_rot(np.array([x, y, z]), q_conj(q)))
    acc.append(q_rot(q_conj(q),np.array([x, y, z])))
acc = np.array(acc)
acc = acc - np.array([0,0,1])
acc = acc * 9.81

fig3.add_trace(go.Scatter(x=time,y= acc[:,0], line=dict(color='red'), name='X free'))
fig3.add_trace(go.Scatter(x=time,y= acc[:,1], line=dict(color='green'), name='Y free'))
fig3.add_trace(go.Scatter(x=time,y= acc[:,2], line=dict(color='blue'), name='Z free'))
# plot(fig3)


# plt.plot(time,acc[:,0],c='r',linewidth=0.5)
# plt.plot(time,acc[:,1],c='g',linewidth=0.5)
# plt.plot(time,acc[:,2],c='b',linewidth=0.5)
# plt.legend(["x","y","z"])
# plt.title("acceleration")
# plt.xlabel("time (s)")
# plt.ylabel("accerelation (m/s²)")
# plt.show(block=False)


# Compute translational velocities
# acc[:,2] = acc[:,2] - 9.81

# acc_offset = np.zeros(3)
vel = np.zeros(acc.shape)
for t in range(1,vel.shape[0]):
    vel[t,:] = vel[t-1,:] + acc[t,:]*samplePeriod
    if stationary[t] == True:
        vel[t,:] = np.zeros(3)

# Compute integral drift during non-stationary periods

# fig = plt.figure(figsize=(10, 5))
# plt.plot(time,vel[:,0],c='r',linestyle=':',linewidth=0.5)
# plt.plot(time,vel[:,1],c='g',linestyle=':',linewidth=0.5)
# plt.plot(time,vel[:,2],c='b',linestyle=':',linewidth=0.5)

velDrift = np.zeros(vel.shape)
stationaryStart = np.where(np.diff(stationary.astype(int)) == -1)[0]+1
stationaryEnd = np.where(np.diff(stationary.astype(int)) == 1)[0]+1
for i in range(0,stationaryEnd.shape[0]-1):
    driftRate = vel[stationaryEnd[i]-1,:] / (stationaryEnd[i] - stationaryStart[i])
    enum = np.arange(0,stationaryEnd[i]-stationaryStart[i])
    drift = np.array([enum*driftRate[0], enum*driftRate[1], enum*driftRate[2]]).T
    velDrift[stationaryStart[i]:stationaryEnd[i],:] = drift

# Remove integral drift
vel = vel - velDrift
# fig = plt.figure(figsize=(10, 5))
# plt.plot(time,vel[:,0],c='r',linewidth=0.5)
# plt.plot(time,vel[:,1],c='g',linewidth=0.5)
# plt.plot(time,vel[:,2],c='b',linewidth=0.5)
# plt.legend(["x","y","z"])
# plt.title("velocity")
# plt.xlabel("time (s)")
# plt.ylabel("velocity (m/s)")
# plt.show(block=False)

# -------------------------------------------------------------------------
# Compute translational position
pos = np.zeros(vel.shape)
for t in range(1,pos.shape[0]):
    pos[t,:] = pos[t-1,:] + vel[t,:]*samplePeriod

# fig = plt.figure(figsize=(10, 5))
# plt.plot(time,pos[:,0],c='r',linewidth=0.5)
# plt.plot(time,pos[:,1],c='g',linewidth=0.5)
# plt.plot(time,pos[:,2],c='b',linewidth=0.5)
# plt.legend(["x","y","z"])
# plt.title("position")
# plt.xlabel("time (s)")
# plt.ylabel("position (m)")
# plt.show(block=False)

# -------------------------------------------------------------------------
# Plot 3D foot trajectory
# print(type([q2R(q) for q in quat]))
# R=[q2R(q) for q in quat]
# figF=labFrame()
# figF=addOrigin(figF, colors=['darkgray','darkgray','darkgray'], lineLength=0.4)
# figF=addFrames(figF,
#                 pos,
#                 [q2R(q) for q in quat])   
# figF.show()


# posPlot = pos
# quatPlot = quat

# extraTime = 20
# onesVector = np.ones(int(extraTime*(1/samplePeriod)))

# Create 6 DOF animation
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection='3d') # Axe3D object
# ax.plot(posPlot[:,0],posPlot[:,1],posPlot[:,2])
# min_, max_ = np.min(np.min(posPlot,axis=0)), np.max(np.max(posPlot,axis=0))
# ax.set_xlim(min_,max_)
# ax.set_ylim(min_,max_)
# ax.set_zlim(min_,max_)
# ax.set_title("trajectory")
# ax.set_xlabel("x position (m)")
# ax.set_ylabel("y position (m)")
# ax.set_zlabel("z position (m)")
# plt.show(block=False)

# plt.show()

# if __name__ == "__main__":
#     main()