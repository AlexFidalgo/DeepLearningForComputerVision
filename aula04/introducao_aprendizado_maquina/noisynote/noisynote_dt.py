import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import os

current_directory = os.path.abspath(os.path.dirname(__file__))

def le(nomearq):
  folder = os.path.join(current_directory,nomearq)
  with open(folder,"r") as f:
    linhas=f.readlines()
  linha0=linhas[0].split()
  nl=int(linha0[0]); nc=int(linha0[1])
  a=np.empty((nl,nc),dtype=np.float32)
  for l in range(nl):
    linha=linhas[l+1].split()
    for c in range(nc):
      a[l,c]=np.float32(linha[c])
  return a

### main
ax=le("ax.txt"); ay=le("ay.txt")
qx=le("qx.txt"); qy=le("qy.txt")

arvore= tree.DecisionTreeClassifier()
arvore= arvore.fit(ax, ay)
qp=arvore.predict(qx)
erros=0;
for i in range(qp.shape[0]):
  if qp[i]!=qy[i]: erros+=1
print("Erros=%d/%d.   Pct=%1.3f%%\n"%(erros,qp.shape[0],100.0*erros/qp.shape[0]))