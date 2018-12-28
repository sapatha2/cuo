import numpy as np 
import pandas as pd
import json
import copy 

baredict={
  'energy':[],
  'energy_err':[],
  'obdm_up':[],
  'obdm_up_err':[],
  'obdm_down':[],
  'obdm_down_err':[],
  'tbdm_upup':[],
  'tbdm_upup_err':[],
  'tbdm_updown':[],
  'tbdm_updown_err':[],
  'tbdm_downup':[],
  'tbdm_downup_err':[],
  'tbdm_downdown':[],
  'tbdm_downdown_err':[],
}
full_labels=np.array(["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"])
def readjson(jsonfn):
  #Get dictionary
  dict=copy.deepcopy(baredict)  
  assert(len(dict['energy'])==0) #Make sure everything resets properly

  data=json.load(open(jsonfn))['properties']
  dict['energy'].append(data['total_energy']['value'][0])
  dict['energy_err'].append(data['total_energy']['error'][0])
  for s in ['up','down']:
    dict['obdm_%s'%s].append(data['tbdm_basis']['obdm'][s])
    dict['obdm_%s_err'%s].append(data['tbdm_basis']['obdm'][s+'_err'])
  for s in ['upup','updown','downup','downdown']:
    dict['tbdm_%s'%s].append(data['tbdm_basis']['tbdm'][s])
    dict['tbdm_%s_err'%s].append(data['tbdm_basis']['tbdm'][s+'_err'])
  states=np.array(data['tbdm_basis']['states'])-1
  labels=full_labels[states]
  
  #Convert to DF and serialize
  df=pd.DataFrame(dict)
  df=flatten_matrices(df,labels) 
  return df

def flatten_matrices(matdf,labels):
  '''
  Flatten 1- and 2-rdms
  input: 
  matdf - dataframe with matrices
  labels -  labels for one axis of the matrix
  output:
  newdf - df object with flattened values
  '''
  newdf=None
  for x in list(matdf)[::-1]:
    if('bdm' in x): 
      flat=np.array(matdf[x][0])
      flat=np.reshape(flat,np.prod(flat.shape))
      if('obdm' in x): newlabels=[x+'_'+str(i)+'_'+str(j) for i in labels for j in labels]
      elif('tbdm' in x): newlabels=[x+'_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l) \
                    for i in labels for j in labels for k in labels for l in labels]
      if(newdf is None): newdf=pd.DataFrame(flat[:,np.newaxis].T,columns=newlabels)
      else: newdf=pd.concat((newdf,pd.DataFrame(flat[:,np.newaxis].T,columns=newlabels)),axis=1)
    else: newdf[x]=matdf[x]
  print('reshaped columns')
  return newdf

def gather_all(detgen,N,Ndet,gsw,basename):
  '''
  Gathers all your data and stores into 
  basename/detgen_Ndet_gsw_df.pickle
  '''
  totdf=None 
  for state in ['','2']:
    #Excited states
    for j in range(1,N+1):
      f=basename+'/ex'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc.gosling.json'
      df=readjson(f)
      if(totdf is None): totdf=df
      else: totdf=pd.concat((totdf,df),axis=0)
    #Ground states
    f='base/gs'+state+'.vmc_tbdm.gosling.json'
    df=readjson(f)
    totdf=pd.concat((totdf,df),axis=0)
  
  print(totdf.shape) 
  totdf.to_pickle(basename+'/'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_gosling.pickle')
  return 1

if __name__=='__main__':
  detgen='a'
  N=50
  Ndet=10
  gsw=0.8
  basename='run2a'
  gather_all(detgen,N,Ndet,gsw,basename)
