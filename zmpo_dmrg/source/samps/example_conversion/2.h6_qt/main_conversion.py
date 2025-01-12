import math
import time
import h5py
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.samps import mpo_dmrg_conversion
from zmpo_dmrg.source.qtensor import qtensor_api

sval = 1.0
sz = 0.0
ifQt = False
ifs2proj = True

flmpsQ = h5py.File('./lmpsQ1','r')
flmps0 = h5py.File('./lmps0','w')
qtensor_api.fmpsQtReverse(flmpsQ,flmps0,'L')

flmps1 = h5py.File('./lmps1','w')
# Conversion
t0 = time.time()
mpo_dmrg_conversion.sweep_projection(flmps0,flmps1,ifQt,sval,thresh=1.e-4)
t1 = time.time()
print('dt=',(t1-t0))

flmps2 = h5py.File('./lmpsQ2','w')
qtensor_api.fmpsQt(flmps1,flmps2,'L')

#=== Check ===#

import h5py
import shutil
import numpy
from mpi4py import MPI
from zmpo_dmrg.source.itools.molinfo import class_molinfo
from zmpo_dmrg.source import mpo_dmrg_class
from zmpo_dmrg.source import mpo_dmrg_schedule

#==================================
# Main program
#==================================
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
# MPI init
if size > 0 and rank ==0: print('\n[MPI init]')
comm.Barrier()
print(' Rank= %s of %s processes'%(rank,size))

mol=class_molinfo()
mol.comm=comm
fname = "mole.h5"
mol.loadHam(fname)
mol.isym =0 #2 #WhetherUseSym
mol.symSz=0 #1 #TargetSpin-2*Sz
mol.symS2=0.0 #Total Spin
# Tempory file will be put to this dir
mol.tmpdir = './'
mol.build()

dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas//2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec,sz]):1} 
#---------------------------
if ifs2proj:
   dmrg2.ifs2proj = True
   dmrg2.npts = 4
   dmrg2.s2quad(sval,sz)
#---------------------------
mol.build()
dmrg2.path = mol.path
dmrg2.ifQt = True
dmrg2.partition()
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
#dmrg2.checkMPS(flmps2)


from zmpo_dmrg.source import mpo_dmrg_init
pop = mpo_dmrg_init.genPops(dmrg2,flmpsQ,flmpsQ,'./tmp_sop','L')
pop = numpy.dot(dmrg2.qwts,pop)
dmrg2.ifs2proj = False
sop = mpo_dmrg_init.genSops(dmrg2,flmpsQ,flmps2,'./tmp_sop','L')
#
# pop= 0.455118136425
# sop= 0.674518871551
# Overlap: <Psi|P|Psi0>*N0= 0.999843513716
#
print()
print('pop=',pop)
print('sop=',sop)
print('Overlap: <Psi|P|Psi0>*N0=',sop/math.sqrt(pop))
exit()


# <S2>   
if not ifs2proj:
   info=None
else:
   info=[dmrg2.npts,sval,sz]
from zmpo_dmrg.source.properties import mpo_dmrg_propsItrf
expect = mpo_dmrg_propsItrf.eval_S2Global(dmrg2,flmps2,spinfo=info)
print('expect_S2=',expect)

# New L-MPS
dmrg2.final()

flmpsQ.close()
flmps0.close()
flmps1.close()
flmps2.close()
