# import tsplib95 as t95
import time
# import lkh
import numpy as np
from scipy.spatial import distance_matrix, distance
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
from solver_opt import solver_gurobi, solver_concorde
import elkai
import collections
from util import *
# solver_path = '../Instances/LKH-3.0.6/LKH'

# def solve(problemPath):
# 	start_time = time.time()
# 	problem = t95.load(problemPath)
# 	result = lkh.solve(solver="LKH", problem=problem, max_trials=10000, runs=10)
# 	end_time = time.time()
# 	print(end_time-start_time, get_objective(result[0], problem.node_coords))

def solver_lkh(centers):
	return elkai.solve_int_matrix(getDM(centers))
def solver_sa(centers):
	return solve_tsp_simulated_annealing(getDM(centers))[0]
def solver_ls(centers):
	return solve_tsp_local_search(getDM(centers))[0]
def solver_dp(centers):
	return solve_tsp_dynamic_programming(getDM(centers))[0]
def solver_opt(centers):
	return solver_gurobi(getDM(centers))
def solver_cc(centers):
	return solver_concorde(centers)

solvers = {
	"lkh": solver_lkh,
	"sa": solver_sa,
	"ls": solver_ls,
	"dp": solver_dp,
	"opt": solver_opt,
	"cc": solver_cc
}

def get_distance(instancePath, permutation):
	centers = np.load(instancePath)
	centers = centers[permutation, 1:]
	start = centers[:-1,:]
	end = centers[1:,:]
	# print(np.linalg.norm(start - end))
	# print(np.sum(np.sqrt(np.sum(np.square(end - start), axis=1))))
	# dist = distance.cdist(centers, centers, "euclidean")
	# print(np.sum(np.diag(dist, 1)))
	return np.sum(np.sqrt(np.sum(np.square(end - start), axis=1)))
	#return np.linalg.norm(start - end)

def solve(instancePath, instanceTrace, solverType, mode): # -1: flat; 1: hierarchical
	start_time = time.time()
	permutation = solver_helper(instancePath, instanceTrace, mode, solverType)
	solution_time = time.time() - start_time
	if permutation:
		centers = np.load(instancePath)
		dm = distance_matrix(centers[:,1:], centers[:,1:])
		distance = getdist(permutation, dm)
	else:
		distance = -1
	# print(distance, mode, solverType, permutation)
	return solution_time, distance

def getdist(permutation, dm):
	dists = [dm[permutation[i], permutation[i+1]] for i in range(len(permutation) - 1)]

	dists.append(dm[permutation[-1], permutation[0]])
	return np.sum(dists)

def getDM(centers):
	return distance_matrix(centers, centers)


def quick_solve(centers, solverType):
	if len(centers) == 1:
		return [0]
	elif len(centers) == 2:
		return [0, 1]
	elif len(centers) == 3:
		dm = getDM(centers)
		maxloc = np.argmax([dm[0,1], dm[1,2], dm[2,0]])
		if maxloc == 0:
			return [1, 2, 0]
		elif maxloc == 1:
			return [2, 0, 1]
		else:
			return [0, 1, 2]
	else:
		return solvers[solverType](centers)

def solver_helper(instancePath, instanceTrace, mode, solverType):
	centers = np.load(instancePath)
	trace = loadJson(instanceTrace)

	if mode < 0:
		aggCenters = np.fromiter(trace.keys(), dtype=int)
		centers_adj = np.delete(centers, aggCenters, axis=0)
		# dm = distance_matrix(centers_adj[:,1:], centers_adj[:,1:])
		permutation = quick_solve(centers_adj[:,1:], solverType)
		permutation = [centers_adj[:,0][permutation[i]] for i in range(len(permutation))]

	else:
		permutationDict = {}
		for key in trace:
			if len(trace[key]) > 1:
				# dm = distance_matrix(centers[trace[key],1:], centers[trace[key], 1:])
				permutation = quick_solve(centers[trace[key],1:], solverType)
				if len(permutation) == len(trace[key]):
					permutationDict[key] = [trace[key][permutation[i]] for i in range(len(permutation))]
				else:
					return permutation
			else:
				permutationDict[key] = trace[key]
		permutation = mergeTrace(permutationDict)
		# print(permutation)
		# if solverType == "lkh":
		# 	print(centers)
		# dm = distance_matrix(centers[:,1:], centers[:,1:])
		# print(getdist(permutation, dm), "sub")
	return permutation

def mergeTrace(permutationDict):
	permutation = []
	def dfs(curr):
		if curr in permutationDict:
			for child in permutationDict[curr]:
				isTerminal = dfs(str(child))
				if isTerminal:
					permutation.append(child)
			return False
		else:
			return True

	dfs("0")
	return permutation




if __name__ == "__main__":
# 	#solve("Instances/A-n32-k5.vrp.txt")
# 	# solve("Instances/nodes.vrp.txt")
# 	# solve("Instances/nodes1.vrp.txt")
	base = "Instances/nodes1_3_10_0"
	
	solve(base+".npy", base+".json", "cc", 1)
	# solve(base+".npy", base+".json", "opt", -1)
	# solve(base+".npy", base+".json", "ls", -1)
	# solve(base+".npy", base+".json", "lkh", -1)
	# solve(base+".npy", base+".json", "ls", 1)
	# solve(base+".npy", base+".json", "lkh", 1)
	
	
