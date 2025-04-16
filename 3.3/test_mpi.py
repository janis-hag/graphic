#!/dios/shared/apps/python/2.7.8/bin/python
# -*- coding: utf-8 -*-

from mpi4py import MPI

nprocs = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
procnm = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

print(rank, nprocs)
