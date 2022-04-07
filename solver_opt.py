import numpy as np
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from concorde.tsp import TSPSolver

def getdist(permutation, dm):
	dists = [dm[permutation[i], permutation[i+1]] for i in range(len(permutation) - 1)]
	return np.sum(dists)

def solver_gurobi(dm):
	model, vars = construct_model(dm)
	model.optimize(subtourelim)
	if model.status == GRB.OPTIMAL:
		vals = model.getAttr("X", vars)
		tour = subtour(vals, dm.shape[0])
		# print(model.ObjVal, "GUROBI", getdist(tour, dm))
		return tour#, model.ObjVal
	else:
		return []

def construct_model(dm):
	model = gp.Model()

	dist = {(i,j): dm[i,j] for i in range(dm.shape[0]) for j in range(i)}

	vars = model.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name="e")
	for i,j in vars.keys():
		vars[j,i] = vars[i,j]
	model.addConstrs(vars.sum(i, "*") == 2 for i in range(dm.shape[0]))

	model._vars = vars
	model._n = dm.shape[0]
	model.Params.LazyConstraints = 1
	model.Params.LogToConsole = 0
	model.Params.OutputFlag = 0
	return model, vars

def subtourelim(model, where):
	if where == GRB.Callback.MIPSOL:
		vals = model.cbGetSolution(model._vars)
		tour = subtour(vals, model._n)
		if len(tour) < model._n:
			model.cbLazy(gp.quicksum(model._vars[i,j] for i,j in combinations(tour, 2)) <= len(tour) - 1)

def subtour(vals, n):
	edges = gp.tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)
	unvisited = list(range(n))
	cycle = range(n+1)
	while unvisited:
		thiscycle = []
		neighbors = unvisited
		while neighbors:
			current = neighbors[0]
			thiscycle.append(current)
			unvisited.remove(current)
			neighbors = [j for i,j in edges.select(current, "*") if j in unvisited]
		if len(cycle) > len(thiscycle):
			cycle = thiscycle
	return cycle

def solver_concorde(centers):
	print(centers, "****")
	solver = TSPSolver.from_data(centers[:,0], centers[:,1], norm="EUC_2D")
	tour = solver.solve()
	print(tour, "&&&&")
	return tour.tour
