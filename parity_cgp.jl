using CartesianGeneticProgramming
using Cambrian
import Cambrian.mutate
using StatsBase
using Base.Iterators: repeated
include("gen_training_data.jl")
"""
A simple example of using Cartesian Genetic Programming to 
generate a circuit to calculate parity
"""

 NUMCLASSES = 10
 WIDTH = 8 #60
 
 train = create_parity_testcases(WIDTH)
 trainX = [ x[1] for x in train ]
 trainX = hcat(trainX...)
 trainY = [ x[2] for x in train ]
 

X, Y = trainX, trainY

function evaluate(ind::CGPInd, X::AbstractArray, Y::AbstractArray)
    accuracy = 0.0
    for i in 1:size(X, 2)
        out = process(ind, float(X[:, i]))
        if out[1] == Int(Y[i])
            accuracy += 1
        end
    end
    [accuracy / size(X, 1)]
end

function error_count(e, X::AbstractArray, Y::AbstractArray)
   failed = 0
   for i in 1:size(X,2)
      y_hat = process(e.population[5], float(X[:,i]))
      if y_hat[1] != Int(Y[i]) 
           failed += 1
      end
   end
   failed
end

cfg = get_config("./parity.yaml")
fit(i::CGPInd) = evaluate(i, X, Y)
mutate(i::CGPInd) = goldman_mutate(cfg, i)
e = CGPEvolution(cfg, fit)
println("run!")
run!(e)

@show e.population[5].nodes
errors = error_count(e,X,Y )
println("Errors: $errors out of $(size(X,2)) entries")
