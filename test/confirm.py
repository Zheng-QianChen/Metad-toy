import pyscal.core as pc
sys = pc.System()
sys.read_inputfile('dump.zero_force_test.lammpstrj')
sys.find_neighbors(method='cutoff', cutoff=4)
sys.calculate_q(4)
#sys.calculate_disorder(averaged=True, q=6)
