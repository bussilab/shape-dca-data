#Import modules, define constants
import numpy as np
import subprocess as sp
import math
import time
import scipy.optimize as opt
import sys
sys.path.append("/u/sbp/ncalonac/modules/ViennaRNA/2.4.10/lib/python3.6/site-packages/")
import RNA
import tempfile
import os
import pandas as pd
RT=0.61633119654

#Import data
df = pd.read_pickle('data.p')
from sklearn.model_selection import train_test_split
ds, test = train_test_split(df, test_size = 0.2, random_state = 1998)
tsystems = list(ds.pdb.values)

refdb=[]
train_rdata=[]
train_labels=[]
train_set=[]
train_seq=[]

for m,mol in enumerate(tsystems):
    train_rdata.append(ds[ds.pdb==mol].norm_r.values[0])
    refdb.append(ds[ds.pdb==mol].stru.values[0])
    train_labels.append(np.ones(len(refdb[m])))
    train_set.append(mol)
    train_seq.append(ds[ds.pdb==mol].seq.values[0])
    for nt in range(len(train_rdata[m])):
        if refdb[m][nt]=='.' or refdb[m][nt]=='[' or refdb[m][nt]==']':
            train_labels[m][nt]=0

#Pairing matrix
refdbmx=[]
for m,mol in enumerate(tsystems):
    refdbmx.append(np.zeros((len(refdb[m]),len(refdb[m]))))
    l=0
    a=np.zeros(len(refdb[m]))
    for nt in range(len(refdb[m])):
        if refdb[m][nt]=='(':
            a[l]=nt
            l+=1
        if refdb[m][nt]==')':
            l-=1
            refdbmx[-1][int(a[l]),nt]=1
            refdbmx[-1][nt,int(a[l])]=1
#Dca matrix
dcamx=[]
y=0
for mol in tsystems:
    dcamx.append(ds[ds.pdb==mol].dca.values[0])

#Activation functions
def lin(x):
    out=x
    dev=1.0
    return [out,dev]

def sigmoid(x):
    out=1/(1+np.exp(-x))
    dev=np.exp(-x)/((1+np.exp(-x))**2)
    return [out,dev]

def dca(coup,par):
    act=sigmoid(par[0]*coup+par[1])
    dev=np.zeros((len(par),coup.shape[0],coup.shape[1]))
    out=par[2]*act[0]+par[3]
    dev[0,:,:]=par[2]*act[1]*coup
    dev[1,:,:]=par[2]*act[1]
    dev[2,:,:]=act[0]
    dev[3,:,:]=1.0
    return [out,dev]

def loop(shape,seq,index,par,window):
    out=0.0
    dev=np.zeros(len(par))
    nn_left=-window
    nn_right=-nn_left+1
    width=len(range(nn_left,nn_right))+1
    x=np.zeros(width)
    if index < -nn_left:
        nn_left=-index
    if index >= len(shape)+nn_left:
        nn_right=len(shape)-index
    x[nn_left + window: nn_right + window]= shape[index+nn_left:index+nn_right]
    x[-1]=1.0
    activation=lin(np.dot(par[:width],x))
    out=activation[0]
    for i in range(len(dev)):
        dev[i]=activation[1]*x[i]
    return [out, dev]

def accuracy(bpmx,refmx):
    tp=0
    fp=0
    fn=0
    for i in range(bpmx.shape[0]):
        for j in range(i,bpmx.shape[1]):
            tp+=refmx[i,j]*bpmx[i,j]
            fp+=(1.0-refmx[i,j])*bpmx[i,j]
            fn+=refmx[i,j]*(1-bpmx[i,j])
    ppv=1/(1+(fp/tp))
    sens=1/(1+(fn/tp))
    mcc=np.sqrt(ppv*sens)
    return [tp,fp,fn,ppv,sens,mcc]

##Import hyper-paramters
dim_dca=4
if __name__ == "__main__":
    alpha_s = float(sys.argv[1])
    alpha_d = float(sys.argv[2])
    dim_shape = int(sys.argv[3])
    mod = sys.argv[4]

#Cost functions
def cost(pp,k):
    pseudoene_s=np.zeros(len(train_rdata[k]))
    fold=RNA.fold_compound(train_seq[k])
    qqq=pp[:dim_dca]
    ppp=pp[dim_dca:]
    dca_dev=np.zeros((len(train_rdata[k]),len(train_rdata[k]),len(qqq)))
    psen_dca=dca(dcamx[k],qqq)
    if np.isinf(alpha_d)==False:
        for i in range(len(train_rdata[k])):
            for j in range(i+4,len(train_rdata[k])):
                fold.sc_add_bp(i+1,j+1,RT*psen_dca[0][i,j])
                dca_dev[i,j,:]=psen_dca[1][:,i,j]

    shape_dev=np.zeros((len(train_rdata[k]),len(ppp)))
    if np.isinf(alpha_s)==False:
        for i in range(len(train_rdata[k])):
            psen_shape=loop(train_rdata[k],train_seq[k],i,ppp,dim_shape)
            fold.sc_add_up(i+1,RT*psen_shape[0])
            shape_dev[i]=psen_shape[1]
            pseudoene_s[i]=(psen_shape[0])
    (ss, mfe) = fold.mfe()
    fold.exp_params_rescale(mfe)
    part=fold.pf()[1]
    ene=fold.eval_structure(''.join(refdb[k]))
    foldbpp=fold.bpp()
    cost=ene-part
    if np.isinf(alpha_d)==False:
        cost+=alpha_d*np.sum(psen_dca[0]**2)
    if np.isinf(alpha_s)==False:
        cost+=alpha_s*np.sum(pseudoene_s**2)
    bpp=np.array(foldbpp)[1:,1:]
    bpp=bpp+bpp.transpose()

    dev=np.zeros(len(pp))
    diffp=np.sum(bpp-refdbmx[k],axis=1)
    if np.isinf(alpha_d)==False:
        for q in range(len(qqq)):
            dev[q]=RT*np.sum(((refdbmx[k]-bpp)+2*alpha_d*psen_dca[0])*dca_dev[:,:,q])
    if np.isinf(alpha_s)==False:
        for q in range(len(ppp)):
            dev[q+len(qqq)]=RT*np.dot(shape_dev[:,q],diffp+2*alpha_s*pseudoene_s)
    if ene>0:
        cost=100
        dev*=0
    return [cost,dev]

def total_ll(par,dset):
    cost_tot=0.0
    dev_tot=np.zeros(len(par))
    for k in dset:
        cc=cost(par,k)
        cost_tot+=cc[0]
        dev_tot+=cc[1]
    return [cost_tot/len(dset),dev_tot/len(dset)]

#Minimization

def minimize(initcond):
    sol=opt.minimize(total_ll,initcond,b,method='SLSQP',jac=True,options={'disp':True})
    return [sol.fun,sol.x,initcond]

train = False
test = False

if __name__ == "__main__":
    sys.stderr = open('next.log', 'w')
    sys.stdout = open('next.out', 'w')
    mol_indexes = range(len(train_set))
    np.random.seed(1992)
    if mod == 'train':
        train = True 
        resultfile = sys.argv[5]
        parameterfile = sys.argv[6]
    elif mod == 'test':
        test = True
        resultfile = sys.argv[5]
        parameterfile = sys.argv[6]
        leftout = int(sys.argv[7])

##Training
if train:
    b=mol_indexes
    start=np.random.rand(20,dim_dca+2*dim_shape+2)
    start[0]=np.zeros(dim_dca+2*dim_shape+2)
    start[:5]*=0.1
    start[5:10]*=0.01
    start[10:15]*=0.001
    start[15:20]*=0.0001
    if dim_shape>0:
        f=open('ewregl'+str(alpha_s)+'_'+str(alpha_d)+'_'+str(dim_shape-1)+'_train_par.dat').readlines()
        start_opt=np.array(f[f.index('# Optimal parameters\n')+1:f.index('# Optimal parameters\n')+dim_dca+2*(dim_shape-1)+2+1],dtype=float)
        start[-1]=np.zeros(dim_dca+2*dim_shape+2)
        start[-1][:dim_dca]=start_opt[:dim_dca]
        for i in range(1,2*dim_shape):
            start[-1][dim_dca+i]=start_opt[dim_dca+(i-1)]
    if alpha_s==np.inf:
        start[:,dim_dca:]*=0.0
    if alpha_d==np.inf:
        start[:,:dim_dca]*=0.0
    if dim_shape>=1:
        start[-1][-1]=start_opt[-1] # termine noto

    if(alpha_d!=np.inf):
        if(alpha_d==0.0):
            alpha_d_prev=0.0001
        elif(alpha_d==1.0):
            alpha_d_prev=np.inf
        else:
            alpha_d_prev=alpha_d*10
        f=open('ewregl'+str(alpha_s)+'_'+str(alpha_d_prev)+'_'+str(dim_shape)+'_train_par.dat').readlines()
        start_opt=np.array(f[f.index('# Optimal parameters\n')+1:f.index('# Optimal parameters\n')+dim_dca+2*(dim_shape)+2+1],dtype=float)
        start[-2]=start_opt

    if(alpha_s!=np.inf):
        if(alpha_s==0.0):
            alpha_s_prev=0.0001
        elif(alpha_s==1.0):
            alpha_s_prev=np.inf
        else:
            alpha_s_prev=alpha_s*10
        f=open('ewregl'+str(alpha_s_prev)+'_'+str(alpha_d)+'_'+str(dim_shape)+'_train_par.dat').readlines()
        start_opt=np.array(f[f.index('# Optimal parameters\n')+1:f.index('# Optimal parameters\n')+dim_dca+2*(dim_shape)+2+1],dtype=float)
        start[-3]=start_opt
    
    if __name__=='__main__':
        import multiprocessing as mp
        pool=mp.Pool()
        sol_set=pool.map(minimize,start)
    tf=[]
    tp=[]
    stp=[]
    for i in sol_set:
        tf.append(i[0])
        tp.append(i[1])
        stp.append(i[2])
        print(i[0],i[1],i[2])
    best_tf=np.nanmin([i for i in tf if i>=0])
    best_tp=tp[tf.index(best_tf)]
    best_stp=stp[tf.index(best_tf)]
    for lo in b:
        fold=RNA.fold_compound(train_seq[lo])
        (ss, mfe) = fold.mfe()
        fold.exp_params_rescale(mfe)
        part_init=fold.pf()
        ene_init=fold.eval_structure(''.join(refdb[lo]))
        cost_init=ene_init-part_init[1]
        fold=RNA.fold_compound(train_seq[lo])
        if np.isinf(alpha_d)==False:
            psen_dca=dca(dcamx[lo],best_tp[:dim_dca])
            for i in range(len(train_rdata[lo])):
                for j in range(i+4,len(train_rdata[lo])):
                    fold.sc_add_bp(i+1,j+1,RT*psen_dca[0][i,j])
        
        if np.isinf(alpha_s)==False: 
            for i in range(len(train_rdata[lo])):
                psen_shape=loop(train_rdata[lo],train_seq[lo],i,best_tp[dim_dca:],dim_shape)
                fold.sc_add_up(i+1,RT*psen_shape[0])
        (ss, mfe) = fold.mfe()
        fold.exp_params_rescale(mfe)
        part_fin=fold.pf()
        ene_fin=fold.eval_structure(''.join(refdb[lo]))
        cost_fin=ene_fin-part_fin[1]
        
        bpp=np.array(fold.bpp())[1:,1:]
        bpp=bpp+bpp.transpose()
        tp,fp,fn,ppv,sens,mcc=accuracy(bpp,refdbmx[lo])
        with open(resultfile,'a') as fout:
            fout.write(train_set[lo]+"\t%d\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n" % (lo, best_tf, cost_init, cost_fin, tp, fp, fn, ppv, sens,mcc))
    with open(parameterfile,'a') as fout:
        np.savetxt(fout,best_stp,header='Starting parameters')
        np.savetxt(fout,best_tp,header='Optimal parameters')
##Leave-One-Out
if test:
    np.random.seed(1992)
    start=np.random.rand(20,dim_dca+2*dim_shape+2)
    start[0]=np.zeros(dim_dca+2*dim_shape+2)
    start[:5]*=0.1
    start[5:10]*=0.001
    start[10:15]*=0.0001
    start[15:20]*=0.00001
    lo=leftout
    if dim_shape>0:
        f=open('ewregl'+str(alpha_s)+'_'+str(alpha_d)+'_'+str(dim_shape-1)+'_test_par.dat').readlines()
        start_opt=np.array(f[f.index('# best final pars for '+train_set[lo]+'\n')+1:f.index('# best final pars for '+train_set[lo]+'\n')+dim_dca+2*(dim_shape-1)+2+1],dtype=float)
        start[-1]=np.zeros(dim_dca+2*dim_shape+2)
        start[-1][:dim_dca]=start_opt[:dim_dca]
        for i in range(1,2*dim_shape):
            start[-1][dim_dca+i]=start_opt[dim_dca+(i-1)]
    if alpha_s==np.inf:
        start[:,dim_dca:]*=0.0
    if alpha_d==np.inf:
        start[:,:dim_dca]*=0.0
    if dim_shape>=1:
        start[-1][-1]=start_opt[-1] # termine noto
    
    if(alpha_d!=np.inf):
        if(alpha_d==0.0):
            alpha_d_prev=0.0001
        elif(alpha_d==1.0):
            alpha_d_prev=np.inf
        else:
            alpha_d_prev=alpha_d*10
        f=open('ewregl'+str(alpha_s)+'_'+str(alpha_d_prev)+'_'+str(dim_shape)+'_test_par.dat').readlines()
        start_opt=np.array(f[f.index('# best final pars for '+train_set[lo]+'\n')+1:f.index('# best final pars for '+train_set[lo]+'\n')+dim_dca+2*(dim_shape)+2+1],dtype=float)
        start[-2]=start_opt

    if(alpha_s!=np.inf):
        if(alpha_s==0.0):
            alpha_s_prev=0.0001
        elif(alpha_s==1.0):
            alpha_s_prev=np.inf
        else:
            alpha_s_prev=alpha_s*10
        f=open('ewregl'+str(alpha_s_prev)+'_'+str(alpha_d)+'_'+str(dim_shape)+'_test_par.dat').readlines()
        start_opt=np.array(f[f.index('# best final pars for '+train_set[lo]+'\n')+1:f.index('# best final pars for '+train_set[lo]+'\n')+dim_dca+2*(dim_shape)+2+1],dtype=float)
        start[-3]=start_opt
    b=list(mol_indexes[:lo])+list(mol_indexes[lo+1:])   #leave_one_out
    if __name__=='__main__':
        import multiprocessing as mp
        pool=mp.Pool()
        sol_set=pool.map(minimize,start)
    tf=[]
    tp=[]
    stp=[]
    for i in sol_set:
        tf.append(i[0])
        tp.append(i[1])
        stp.append(i[2])
    #best_tf=np.nanmin(np.array(tf))
    best_tf=np.nanmin([i for i in tf if i>=0])
    best_tp=tp[tf.index(best_tf)]
    best_stp=stp[tf.index(best_tf)]
    fold=RNA.fold_compound(train_seq[lo])
    (ss, mfe) = fold.mfe()
    fold.exp_params_rescale(mfe)
    part_init=fold.pf()
    ene_init=fold.eval_structure(''.join(refdb[lo]))
    cost_init=ene_init-part_init[1]
    fold.exp_params_rescale(mfe)
    if np.isinf(alpha_d)==False: 
        psen_dca=dca(dcamx[lo],best_tp[:dim_dca])
        for i in range(len(train_rdata[lo])):
            for j in range(i+4,len(train_rdata[lo])):
                fold.sc_add_bp(i+1,j+1,RT*psen_dca[0][i,j])

    if np.isinf(alpha_s)==False:
        for i in range(len(train_rdata[lo])):
            psen_shape=loop(train_rdata[lo],train_seq[lo],i,best_tp[dim_dca:],dim_shape)
            fold.sc_add_up(i+1,RT*psen_shape[0])
    (ss, mfe) = fold.mfe()
    fold.exp_params_rescale(mfe)
    part_fin=fold.pf()
    ene_fin=fold.eval_structure(''.join(refdb[lo]))
    cost_fin=ene_fin-part_fin[1]

    #dbn=fold.mfe()
    bpp=np.array(fold.bpp())[1:,1:]
    bpp=bpp+bpp.transpose()
    tp,fp,fn,ppv,sens,mcc=accuracy(bpp,refdbmx[lo])
    with open(resultfile,'a') as fout:
        fout.write(train_set[lo]+"\t%d\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\n" % (lo, best_tf, cost_init, cost_fin, tp, fp, fn, ppv, sens,mcc))
    with open(parameterfile,'a') as fout:
        np.savetxt(fout,best_stp,header='best starting pars for '+train_set[lo])
        np.savetxt(fout,best_tp,header='best final pars for '+train_set[lo])
