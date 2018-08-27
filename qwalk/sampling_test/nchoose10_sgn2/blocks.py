#Generates blocks.json to calculate error bars on d<H>/dp accurately
from covariance import gather_json
import itertools 
import pandas as pd

#Read files
basename="Cuvtz0_B3LYP_s"
labels=['E',['dpenergy'+str(i) for i in range(10)],['dpwf'+str(i) for i in range(10)]]
labels=list(itertools.chain.from_iterable(labels))
blks=pd.DataFrame(columns=labels)

for i in range(1,10):
  for j in range(1,10):
    fname=basename+str(i)+"_g0."+str(j)+".vmc.json"
    tmp=gather_json(fname)
    blks=blks.append(tmp.mean(),ignore_index=True)
    print(blks)

blks.to_json("blocks.json")
