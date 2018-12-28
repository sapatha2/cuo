import numpy as np 
import pandas as pd
import json

def gather_json_df(jsonfn):
  ''' 
  Args:
    jsonfn (str): name of json file to read.
  Returns:
    DataFrame: dataframe indexed by block with columns for energy, and each dpenergy and dpwf.
  '''
  blockdict={
      'energy':[],
  }
  obdmdict={
      'obdm_up':[],
      'obdm_down':[],
      #'normalization':[],
  }
  tbdmdict={
      'tbdm_upup':[],
      'tbdm_updown':[],
      'tbdm_downup':[],
      'tbdm_downdown':[],
  }
  states = None
  with open(jsonfn) as jsonf:
    for blockstr in jsonf.read().split("<RS>"):
      if '{' in blockstr:
        block = json.loads(blockstr.replace("inf","0"))['properties']
        blockdict['energy'].append(block['total_energy']['value'][0])

        for s in ['up','down']:
          obdmdict['obdm_%s'%s].append(np.array(block['tbdm_basis']['obdm'][s]))
        #obdmdict['normalization'].append(block['tbdm_basis']['normalization']['value'])
        states = np.array(block['tbdm_basis']['states'])
   
        for s in ['upup','updown','downup','downdown']:
          tbdmdict['tbdm_%s'%s].append(block['tbdm_basis']['tbdm'][s])
  
  print('Dict loaded from json') 
  blockdict.update(obdmdict)
  blockdict.update(tbdmdict)
  blockdf = pd.DataFrame(blockdict)
  blockdf = flatten_matrices(blockdf,states)
  return blockdf

def flatten_matrices(blockdf,labels):
  '''
  Flatten 1- and 2-rdms
  input: 
  blockdf - dataframe with matrices
  labels -  labels for one axis of the matrix
  output:
  blockdf - dataframe with flattened values
  '''
  newdf=None  
  for x in list(blockdf):
    if('bdm' in x): 
      flat=np.stack(blockdf[x],axis=0)
      flat=np.reshape(flat,(flat.shape[0],np.prod(flat.shape[1:])))
      if('obdm' in x): newlabels=[x+'_'+str(i)+'_'+str(j) for i in labels for j in labels]
      elif('tbdm' in x): newlabels=[x+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l) \
                    for i in labels for j in labels for k in labels for l in labels]
      if(newdf is None): newdf=pd.DataFrame(flat,columns=newlabels)
      else: newdf=pd.concat((newdf,pd.DataFrame(flat,columns=newlabels)),axis=1)
      blockdf=blockdf.drop(columns=[x])
  blockdf=pd.concat((blockdf,newdf),axis=1)
  print('reshaped columns')
  return blockdf

import time 
def gather_all(detgen,N,Ndet,gsw,basename):
  '''
  Gathers all your data and stores into 
  basename/detgen_Ndet_gsw_df.pickle
  '''
  data=None
  labels=[]
  
  for state in ['','2']:
    #Excited states
    for j in range(1,N+1):
      f=basename+'/ex'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc.json'
      
      t0=time.time()
      df=gather_json_df(f)
      m=np.mean(df.values,axis=0)
      s=np.std(df.values,axis=0)/np.sqrt(df.shape[0])
      print('Gather time: ', time.time()-t0)

      print(m[:10])
      print(list(df)[:10])
      print(m[1050:1100])
      print(list(df)[1050:1100])
      exit(0) 
      tot=np.concatenate((m,s))
      if(data is None): data=tot
      else: 
        if(len(data.shape)<2): data=data[:,np.newaxis]
        data=np.concatenate((data,tot[:,np.newaxis]),axis=1)
    #Ground states
    f='base/gs'+state+'.vmc_tbdm.json'
    df=gather_json_df(f)
    m=np.mean(df.values,axis=0)
    s=np.mean(df.values,axis=0)/np.sqrt(df.shape[0])
    tot=np.concatenate((m,s))
    data=np.concatenate((data,tot[:,np.newaxis]),axis=1)

  labels=list(df)
  labels+=[x+'_err' for x in labels]
  df=pd.DataFrame(data.T,columns=labels)
  df.to_pickle(basename+'/'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_df2rdm.pickle')

if __name__=='__main__':
  detgen='s'
  N=50
  Ndet=10
  gsw=0.7
  basename='run1s'
  gather_all(detgen,N,Ndet,gsw,basename)
