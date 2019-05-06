import numpy as np 
import matplotlib.pyplot as plt

parms_list=[
(0,-3.2999,-0.9092,-1.8537,0.8163,0,0,0),          #Tpi only,R2=0.979,C=700
(0,-3.2836,-0.9591,-1.8779,0.8014,0,0.1260,0),     #Tpi,Tsz, R2=0.979,C=800
(0,-3.3011,-0.9135,-1.8600,0.8125,-0.0096,0,0),    #Tpi,Tdz,R2=0.979,C=800
(0,-3.2487,-1.6910,-2.575,0.3782,-0.7785,1.1160,0), #Tpi,Tdz,Ts,R2=0.981,C=1540

(0,-2.4505,-1.4719,-1.6933,0,0,0,2.0757),          #Tpi only,R2=0.980,C=700
(0,-2.7723,-1.2603,-1.7578,0.3189,0,0,1.3095),     #Tpi only,R2=0.981,C=1610
(0,-2.5176,-1.4976,-1.7923,0,-0.1347,0,1.9594),    #Tpi only,R2=0.981,C=947
(0,-2.4457,-1.4105,-1.6489,0,-0.2115,0,2.1666),    #Tpi only,R2=0.981,C=730
]

i=0
for parms in parms_list:
  e4s,e3d,epi,ez,tpi,tdz,tsz,tds=parms
  #dz2, dd, dpi
  H=np.diag([e3d,e3d,e3d,e3d,e3d,ez,epi,epi,e4s])
  H[[3,4,6,7],[6,7,3,4]]=tpi
  H[[0,5],[5,0]]=tdz
  H[[5,8],[8,5]]=tsz
  H[[0,8],[8,0]]=tds

  w,__=np.linalg.eigh(H)
  print(w-w[0])

  plt.plot(np.ones(len(w))*i,w-w[0],'o')
  i+=1
plt.show()

#Anything else added would increase the correlation too much and not affect anything else
#Next step: is there any difference between these different model parameterizations in terms of 1-body properties or energies?

