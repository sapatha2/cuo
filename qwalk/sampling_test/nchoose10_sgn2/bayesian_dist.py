import numpy as np
import scipy
from scipy.integrate import dblquad
from covariance import gather_json

def evaluate_posterior(qvals,xavg,Scov,xlims=None,ylims=None):
  if xlims==None:
    xlims=(xavg[1]-20*Scov[1,1],xavg[1]+20*Scov[1,1])
  if ylims==None:
    ylims=(xavg[2]-20*Scov[2,2],xavg[2]+20*Scov[2,2])
  Scovinv=np.linalg.inv(Scov)
  allvals=[]
  allerr=[]
  for q in qvals:
    def func(y,x):
      xshift=xavg-np.array([q+x*y,x,y])
      #print(x,y,xshift,q)
      return np.exp(-0.5*np.dot(xshift.T,np.dot(Scovinv,xshift)))


    val,valerr=dblquad(func,xlims[0],xlims[1],lambda x:ylims[0],lambda x:ylims[1])
    allvals.append(val)
    allerr.append(valerr)
    #print(q,"done",val,valerr)
  allvals=np.array(allvals)
  allvals=allvals/allvals.max()
  return allvals,allerr


def avg_var(qvals,prob):
  avg=np.sum(qvals*prob)/np.sum(prob)
  var=np.sum((qvals-avg)**2*prob)/np.sum(prob)
  return avg,var
  
def test():
  xavg=np.ones(3)
  import matplotlib.pyplot as plt

  for cov in [-0.005,0.000,0.005,0.01,0.03]:
    Mcov=np.ones((3,3))
    for i in range(3):
      Mcov[i,i]=0.02
    for (i,j) in [(0,1),(1,2),(0,2)]:
      Mcov[i,j]=cov
      Mcov[j,i]=cov

    qest=xavg[0]-xavg[1]*xavg[2]
    qvals=np.linspace(qest-1,qest+1,30)
    allvals,allerr=evaluate_posterior(qvals,xavg,Mcov)
    plt.plot(qvals,allvals,label=str(cov))
  plt.legend()
  plt.show()
  

def testblock():
  import pandas as pd
  #df=pd.read_json("blocks.json")
  df=gather_json("Cuvtz0_B3LYP_s1_g0.9.vmc.json")
  cov=df.cov()
  import matplotlib.pyplot as plt
  
  for ip in range(9):
    A='dpenergy%i'%ip
    B='energy'
    C='dpwf%i'%ip
    cov=df[[A,B,C]].cov()/float(len(df))
    Mcov=cov.values
    xavg=np.ones(3)
    for i,nm in enumerate([A,B,C]):
      xavg[i]=df[nm].mean()
    qest=xavg[0]-xavg[1]*xavg[2]
    qvals=np.linspace(qest-1,qest+1,400)
    allvals,allerr=evaluate_posterior(qvals,xavg,Mcov)
    avg,var=avg_var(qvals,allvals)
    print(ip,"estimated",qest,"average",avg,"+/-",np.sqrt(var))
    plt.plot(qvals,allvals,label='parameter '+str(ip))

  plt.legend()
  plt.xlabel("Value")
  plt.savefig("dist.pdf",bbox_inches='tight')

  #import matplotlib.pyplot as plt
  #plt.hist(df[A])
  #plt.show()


def bootstrapblock():
  import pandas as pd
  #df=pd.read_json("blocks.json")
  df=gather_json("Cuvtz0_B3LYP_s1_g0.9.vmc.json")
  nsample=100
  for ip in range(9):
    qs=[]
    for i in range(nsample):
      dfs=df.sample(frac=1,replace=True)
      A='dpenergy%i'%ip
      B='energy'
      C='dpwf%i'%ip
      qs.append(dfs[A].mean()-dfs[B].mean()*dfs[C].mean())
    qs=np.array(qs)
    print(ip,np.mean(qs),np.std(qs))
    
if __name__=="__main__":
  import time
  start=time.time()
  testblock()
  middle=time.time()
  bootstrapblock()
  end=time.time()
  print("timing: Bayes ",middle-start, "bootstrap",end-middle)

