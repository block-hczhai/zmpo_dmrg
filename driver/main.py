
import numpy as np
from mpi4py import MPI
import h5py
import shutil
import os

comm = MPI.COMM_WORLD

class ZMPODMRG:

    def __init__(self, filename='FCIDUMP', bdims=[50]*8, dav_tols=[1E-4]*8, noises=[1E-4]*4 + [0] * 4, scartch="./tmp"):
        from zmpo_dmrg.source.itools.molinfo import class_molinfo
        info = self.loadERIs(filename)
        if comm.rank == 0:
            self.dump(info)
        comm.Barrier()

        self.mol = class_molinfo()
        self.mol.comm = comm
        self.mol.verbose = 0
        fname = "mole.h5"
        self.mol.loadHam(fname)

        self.mol.tmpdir = scartch + "/"
        self.mol.build()
        self.bdims = bdims
        self.dav_tols = dav_tols
        self.noises = noises

    def convert(self):

        if comm.rank == 0:
            from zmpo_dmrg.source.samps import mpo_dmrg_conversion, block_itrf
            from zmpo_dmrg.source.qtensor import qtensor_api
            fname = './flmpsQt'
            flmpsQ = h5py.File(fname, 'r')
            flmps0 = h5py.File(fname + '_NQt0', 'w')
            flmps1 = h5py.File(fname + '_NQt1', 'w')

            qtensor_api.fmpsQtReverse(flmpsQ, flmps0, 'L')
            ifQt = False

            # Conversion
            mpo_dmrg_conversion.sweep_projection(flmps0, flmps1, ifQt, 
                self.twos / 2, thresh=1.e-8, ifcompress=True,
                ifBlockSingletEmbedding=True, ifBlockSymScreen=True,
                ifpermute=True)
            path = './lmps_compact'
            block_itrf.compact_rotL(flmps1, path)

            flmpsQ.close()
            flmps0.close()
            flmps1.close()
        
        comm.Barrier()

    def clean(self):

        if comm.rank == 0:
            shutil.rmtree(self.mol.tmpdir)
            for x in os.listdir("."):
                if x.startswith("log_"):
                    os.remove(x)
        
        comm.Barrier()

    def prepare(self, hf_occ):
        from zmpo_dmrg.source.qtensor import qtensor_api
        from zmpo_dmrg.source import mpo_dmrg_class
        from zmpo_dmrg.source import mpo_dmrg_schedule

        mol = self.mol
        nelec = self.nelec
        twosz = self.twosz
        conf = list(hf_occ)
        
        dmrg = mpo_dmrg_class.mpo_dmrg()
        dmrg.const = mol.enuc + mol.ecor
        dmrg.iprt = 0
        dmrg.occun = np.array(conf)
        dmrg.path = mol.path
        dmrg.nsite = mol.sbas // 2
        dmrg.sbas  = mol.sbas
        dmrg.comm = comm
        dmrg.isym = 2
        dmrg.build()
        dmrg.qsectors = { str([nelec, twosz / 2]):1 } 
        sc = mpo_dmrg_schedule.schedule()
        sc.fixed(maxM=1, maxiter=0)
        sc.prt()
        dmrg.partition()
        dmrg.loadInts(mol)
        dmrg.dumpMPO()
        dmrg.default(sc)
        dmrg.checkMPS()

        if comm.rank == 0:
            flmps0 = dmrg.flmps
            flmps1 = h5py.File(dmrg.path+'/lmpsQt','w')
            qtensor_api.fmpsQt(flmps0,flmps1,'L')
            flmps0.close()
            flmps1.close()
            shutil.copy(dmrg.path+'/lmpsQt','./lmpsQ0')
        
        comm.Barrier()

    def run(self):
        from zmpo_dmrg.source import mpo_dmrg_class
        from zmpo_dmrg.source import mpo_dmrg_schedule

        mol = self.mol
        nelec = self.nelec
        twos = self.twos
        twosz = self.twosz

        # 1. Using an MPS in Qt form
        flmps1 = h5py.File('./lmpsQ0', 'r')
        dmrg2 = mpo_dmrg_class.mpo_dmrg()
        dmrg2.iprt = 0
        dmrg2.const = mol.enuc + mol.ecor
        dmrg2.nsite = mol.sbas // 2
        dmrg2.sbas  = mol.sbas
        dmrg2.isym = 2
        dmrg2.build()
        dmrg2.comm = mol.comm
        dmrg2.qsectors = { str([nelec, twosz / 2]):1 } 

        sc2 = mpo_dmrg_schedule.schedule()
        sc2.MaxMs = self.bdims
        ns = len(sc2.MaxMs)
        sc2.Sweeps = range(ns)
        sc2.Tols   = self.dav_tols
        sc2.Noises = self.noises
        sc2.coff = 0 

        sc2.Tag = 'Normal2'
        sc2.collect()
        sc2.maxiter = ns
        sc2.prt()

        #---------------------------
        dmrg2.ifs2proj = True
        dmrg2.npts = 10
        dmrg2.s2quad(twos / 2, twosz / 2)
        #---------------------------
        mol.build()
        dmrg2.path = mol.path
        dmrg2.ifQt = True
        dmrg2.partition()
        dmrg2.loadInts(mol)
        dmrg2.dumpMPO()
        dmrg2.default(sc2, flmps1)
        # New L-MPS
        dmrg2.checkMPS()
        dmrg2.final()
        flmps1.close()

        if comm.rank == 0:
            shutil.copy(dmrg2.path + '/lmps', './flmpsQt')
        
        comm.Barrier()

    def dump(self, info, ordering=None, fname='mole.h5'):
        ecore, int1e, int2e = info
        assert ordering is None
        # dump information
        nbas = int1e.shape[0]
        sbas = nbas * 2
        # print('\n[tools_itrf.dump] interface from FCIDUMP with nbas=', nbas)
        f = h5py.File(fname, "w")
        cal = f.create_dataset("cal", (1,), dtype='i')
        cal.attrs["nelec"] = self.nelec
        cal.attrs["sbas"]  = sbas
        cal.attrs["enuc"]  = 0.
        cal.attrs["ecor"]  = ecore
        cal.attrs["escf"]  = 0. # Not useful at all
        # Intergrals
        flter = 'lzf'
        # INT1e:
        h1e = np.zeros((sbas,sbas))
        h1e[0::2,0::2] = int1e # AA
        h1e[1::2,1::2] = int1e # BB
        # INT2e:
        h2e = np.zeros((sbas,sbas,sbas,sbas))
        h2e[0::2,0::2,0::2,0::2] = int2e # AAAA
        h2e[1::2,1::2,1::2,1::2] = int2e # BBBB
        h2e[0::2,0::2,1::2,1::2] = int2e # AABB
        h2e[1::2,1::2,0::2,0::2] = int2e # BBAA
        # <ij|kl> = [ik|jl]
        h2e = h2e.transpose(0,2,1,3)
        # Antisymmetrize V[pqrs]=-1/2*<pq||rs> - In MPO construnction, only r<s part is used. 
        h2e = -0.5*(h2e-h2e.transpose(0,1,3,2))
        int1e = f.create_dataset("int1e", data=h1e, compression=flter)
        int2e = f.create_dataset("int2e", data=h2e, compression=flter)
        # Occupation
        occun = np.zeros(sbas)
        orbsym = np.array([0]*sbas)
        spinsym = np.array([[0,1] for i in range(nbas)]).flatten()
        f.create_dataset("occun",data=occun)
        f.create_dataset("orbsym",data=orbsym)
        f.create_dataset("spinsym",data=spinsym)
        f.close()
        # print(' Successfully dump information for MPO-DMRG calculations! fname=', fname)
        # print(' with ordering', ordering)

    def loadERIs(self, filename='FCIDUMP'):
        # print('\n[tools_io.loadERIs] from FCIDUMP')
        with open(filename,'r') as f:
            line = f.readline()
            dic = {
                x.split('=')[0].split(' ')[-1].strip().lower():
                    x.split('=')[1].strip()
                for x in line.split(',') if x.strip() != '' }
            print(dic)
            line = dic['norb']
            self.n_sites = int(line)
            self.nelec = int(dic['nelec'])
            self.twos = self.twosz = int(dic['ms2'])
            print ('n_sites =', int(line))
            f.readline()
            f.readline()
            f.readline()
            n = int(line)
            e = 0.0
            int1e = np.zeros((n, n))
            int2e = np.zeros((n, n, n, n))
            for line in f.readlines():
                data = line.split()
                ind = [int(x) - 1 for x in data[1:]]
                if ind[2] == -1 and ind[3] == -1:
                    if ind[0] == -1 and ind[1] == -1:
                        e = float(data[0])
                    else:
                        int1e[ind[0], ind[1]] = float(data[0])
                        int1e[ind[1], ind[0]] = float(data[0])
                else:
                    int2e[ind[0], ind[1], ind[2], ind[3]] = float(data[0])
                    int2e[ind[1], ind[0], ind[2], ind[3]] = float(data[0])
                    int2e[ind[0], ind[1], ind[3], ind[2]] = float(data[0])
                    int2e[ind[1], ind[0], ind[3], ind[2]] = float(data[0])
                    int2e[ind[2], ind[3], ind[0], ind[1]] = float(data[0])
                    int2e[ind[3], ind[2], ind[0], ind[1]] = float(data[0])
                    int2e[ind[2], ind[3], ind[1], ind[0]] = float(data[0])
                    int2e[ind[3], ind[2], ind[1], ind[0]] = float(data[0])
        return e, int1e, int2e

def test_1():
    E = -2.190384218792720
    print(E)
    zd = ZMPODMRG('../data/H4.STO6G.R1.8.FCIDUMP',
        bdims=[50]*8, dav_tols=[1E-4]*8, noises=[1E-4]*4 + [0] * 4)
    zd.prepare([1, 1, 0, 0, 1, 1, 0, 0])
    zd.run()
    zd.convert()
    zd.clean()

def test_2():
    E = -107.654122436886396
    print(E)
    zd = ZMPODMRG('../data/N2.STO3G.FCIDUMP',
        bdims=[200]*8, dav_tols=[1E-4]*8, noises=[1E-4]*4 + [0] * 4)
    zd.prepare([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    zd.run()
    zd.convert()
    zd.clean()

if __name__ == "__main__":

    import sys
    import numpy as np
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_1()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test2':
        test_2()
    elif len(sys.argv) > 1:

        fname = sys.argv[1]
        with open(fname, 'r') as fin:
            lines = fin.readlines()
        dic = {}
        schedule = []
        schedule_start = -1
        schedule_end = -1
        for i, line in enumerate(lines):
            if "schedule" == line.strip():
                schedule_start = i
            elif "end" == line.strip():
                schedule_end = i
            elif schedule_start != -1 and schedule_end == -1:
                a, b, c, d = line.split()
                schedule.append([int(a), int(b), float(c), float(d)])
            elif not line.strip().startswith('!'):
                line_sp = line.split()
                if len(line_sp) != 0:
                    if line_sp[0] in dic:
                        raise ValueError("duplicate key (%s)" % line_sp[0])
                    dic[line_sp[0]] = " ".join(line_sp[1:])
        
        tmp = list(zip(*schedule))
        nsweeps = np.diff(tmp[0]).tolist()
        maxiter = int(dic["maxiter"]) - int(np.sum(nsweeps))
        assert maxiter > 0
        nsweeps.append(maxiter)
        
        schedule = [[], [], []]
        for nswp, M, tol, noise in zip(nsweeps, *tmp[1:]):
            schedule[0].extend([M] * nswp)
            schedule[1].extend([tol] * nswp)
            schedule[2].extend([noise] * nswp)
        dic["schedule"] = schedule
        bond_dims, dav_thrds, noises = dic["schedule"]
        fints = dic["orbitals"]
        occ = dic["hf_occ"]
        scartch = dic.get("prefix", "./tmp")

        zd = ZMPODMRG(fints, bdims=bond_dims, dav_tols=dav_thrds, noises=noises, scartch=scartch)
        zd.prepare([int(x) for x in occ.split()])
        zd.run()
        zd.convert()
        zd.clean()

    else:
        raise ValueError("""
            Usage:
                (A) python main.py test
                (A) python main.py test2
                (B) python main.py zmpo.conf
            
            zmpo.conf:
                orbitals: FCIDUMP
                hf_occ: integral occ
                schedule: schedule
                maxiter: number of sweeps
                prefix: scartch
        """)

    
