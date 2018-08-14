from pandas import DataFrame,Series
from json import loads
from numpy import array,einsum,diag,zeros,eye

def gather_json(jsonfn):
  ''' 
  Computing covariance is as easy as gather_json('my.json').cov()

  Args:
    jsonfn (str): name of json file to read.
  Returns:
    DataFrame: dataframe indexed by block with columns for energy, and each dpenergy and dpwf.
  '''
  blockdf={
      'energy':[],
      'dpenergy':[],
      'dpwf':[]
      }
  with open(jsonfn) as jsonf:
    for blockstr in jsonf.read().split("<RS>"):
     # print(blockstr)
      if '{' in blockstr:
        block=loads(blockstr.replace("inf","0"))['properties']
        blockdf['energy'].append(block['total_energy']['value'][0])
        blockdf['dpenergy'].append(block['derivative_dm']['dpenergy']['vals'])
        blockdf['dpwf'].append(block['derivative_dm']['dpwf']['vals'])

  def unpack(vec,key):
    indices=range(len(vec))
    dat=Series(dict(zip([key+'%d'%i for i in indices],vec)))
    return dat

  blockdf=DataFrame(blockdf)
  blockdf=blockdf.join(blockdf['dpenergy'].apply(lambda x:unpack(x,key='dpenergy'))).drop('dpenergy',axis=1)
  blockdf=blockdf.join(blockdf['dpwf'].apply(lambda x:unpack(x,key='dpwf'))).drop('dpwf',axis=1)

  return blockdf

def compute_totcov(cov,energy,dpwf):
  ''' 
  Compute the variance of the quantity dpenergy - energy*dpwf, which represents d<H>/dp.

  Args:
    cov (array-like): Covariance matrix.
    energy (float): total energy.
    dpwf (array-like): wave function derivatives.
  Returns:
    array: error of d<H>/dp indexed by p.
  '''
  #cov=diag(cov.values.diagonal()) # To check diagonal case works out.

  nparam=(cov.shape[0]-1)//2

  jacobian=zeros((nparam,1+2*nparam))
  # d/dE terms:
  jacobian[:,0]=-dpwf
  # d/dpenergy terms:
  jacobian[:,1:1+nparam]=eye(nparam)
  # d/dpwf terms:
  jacobian[:,1+nparam:]=-energy*eye(nparam)

  newcov=einsum('ij,jk,kl->il',jacobian,cov,jacobian.T)
  return newcov.diagonal()

def test_denergy_err(jsonfn):
  from subprocess import check_output
  blockdf=gather_json(jsonfn)
  newcov=compute_totcov(blockdf.cov(),blockdf['energy'].mean(),blockdf[[c for c in blockdf.columns if 'dpwf' in c]].values.mean(axis=0))
  newerr=(newcov/blockdf.shape[0])**0.5

  # Old (incorrect) error computation.
  gosling=loads(check_output(['/u/sciteam/sapatha2/mainline/bin/gosling','-json',jsonfn.replace('.json','.log')]).decode())
  energy    =gosling['properties']['total_energy']['value'][0]
  energy_err=gosling['properties']['total_energy']['error'][0]
  dpwf=    array(gosling['properties']['derivative_dm']['dpwf']['vals'])
  dpwf_err=array(gosling['properties']['derivative_dm']['dpwf']['err'])
  dpenergy=    array(gosling['properties']['derivative_dm']['dpenergy']['vals'])
  dpenergy_err=array(gosling['properties']['derivative_dm']['dpenergy']['err'])

  #print('value')
  val=(dpenergy-dpwf*energy)
  #print('old error')
  #print( (dpenergy_err**2 + (dpwf*energy_err)**2 + (dpwf_err*energy)**2)**0.5 )
  #print('new error')
  #print(newerr)
  return val,newerr

def test_stderr():
  from subprocess import check_output

  jsonfn='ext_0.50_6_10.json'
  blockdf=gather_json(jsonfn)

  nblocks=blockdf.shape[0]
  print(nblocks)
  stderr=Series(dict(zip(blockdf.columns,blockdf.cov().values.diagonal()/nblocks)))**0.5
  print(stderr)

  gosling=loads(check_output(['gosling','-json',jsonfn.replace('json','log')]).decode())
  print("energy")
  print(gosling['properties']['total_energy']['error'][0])
  print("der wave function")
  print(gosling['properties']['derivative_dm']['dpwf']['err'])
  print("der energy")
  print(gosling['properties']['derivative_dm']['dpenergy']['err'])

if __name__=='__main__':
  test_denergy_err()
