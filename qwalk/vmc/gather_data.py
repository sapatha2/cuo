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
      'normalization':[],
  }
  states = None
  with open(jsonfn) as jsonf:
    for blockstr in jsonf.read().split("<RS>"):
      if '{' in blockstr:
        block = json.loads(blockstr.replace("inf","0"))['properties']
        blockdict['energy'].append(block['total_energy']['value'][0])

        for s in ['up','down']:
          #obdmdict['obdm_%s'%s].append(block['tbdm_basis']['obdm'][s])
          obdmdict['obdm_%s'%s].append(np.array(block['tbdm_basis']['obdm'][s]))
        obdmdict['normalization'].append(block['tbdm_basis']['normalization']['value'])
        states = np.array(block['tbdm_basis']['states'])

  print('dict loaded from jsons')
  blockdict.update(obdmdict)
  blockdf = pd.DataFrame(blockdict)
  blockdf = unpack_matrices(blockdf,states=states)
  return blockdf

def unpack_matrices(blockdf, states=None):
  def unpack(vec,key,states=None):
    # expand vector of arrays into series labeled by index
    avec = np.array(vec)
    meshinds = np.meshgrid(*list(map(np.arange,avec.shape)))
    if states is not None and (key.find('bdm')>0 or key=='normalization'):
      if not key.startswith('dp'):
        meshinds[0] = states[meshinds[0]]
      for i in range(1,len(meshinds)):
        meshinds[i] = states[meshinds[i]] 
    labels = list(zip(*list(map(np.ravel,meshinds))))
    dat = pd.Series(dict(zip([key+'_'+'_'.join(map(str,i)) for i in labels], avec.ravel())))
    return dat

  def lists_to_cols(blockdf, key, states=None):
    # expand columns of arrays into separate columns labeled by index and remove original cols
    expanded_cols = blockdf[key].apply(lambda x:unpack(x,key=key,states=states))
    return blockdf.join(expanded_cols).drop(key,axis=1)

  for key in blockdf.keys():
    if key in ['energy', 'states']: continue
    blockdf = lists_to_cols(blockdf, key, states=states)
  print('reshaped columns')
  return blockdf

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
      df=gather_json_df(f)
      m=df.mean()
      s=df.std()/np.sqrt(df.shape[0])
      tot=np.concatenate((m.values,s.values))
      if(data is None): data=tot
      else: 
        if(len(data.shape)<2): data=data[:,np.newaxis]
        data=np.concatenate((data,tot[:,np.newaxis]),axis=1)
    #Ground states
    f='base/gs'+state+'.vmc.json'
    df=gather_json_df(f)
    m=df.mean()
    s=df.std()/np.sqrt(df.shape[0])
    tot=np.concatenate((m.values,s.values))
    data=np.concatenate((data,tot[:,np.newaxis]),axis=1)

  labels=list(df)
  labels+=[x+'_err' for x in labels]
  df=pd.DataFrame(data.T,columns=labels)
  df.to_pickle(basename+'/'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_df.pickle')

if __name__=='__main__':
  detgen='s'
  N=50
  Ndet=10
  gsw=0.7
  basename='run2s'
  gather_all(detgen,N,Ndet,gsw,basename)
