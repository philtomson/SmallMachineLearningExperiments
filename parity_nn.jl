using StatsBase
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using BSON: @save

NUMCLASSES = 1
WIDTH = 8


if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

include("gen_training_data.jl")

train = create_parity_testcases(WIDTH)
trainX = [ x[1] for x in train ]
trainX = hcat(trainX...)
trainY = float.([ x[2] for x in train ])

 # create unique training and test cases
function create_testcases(num_tcs, sz, classes=10)
   train_cases = []
   test_cases = []
   test_dict  = Dict( ) #throw in 1 ( maps num to number of 1s)
   function make_uniq_cases(cases, num)
      for i in 1:floor(num)
         num_ones = rand(1:classes)
         ra = make_rand_array(num_ones, sz)
         #make sure it's not already in the dict
         while haskey(test_dict, bitarray_2_num(ra))
            println("collision - retry")
            @show i
            @show ra
            @show num_ones
            num_ones = rand(1:classes)
            ra = make_rand_array(num_ones, sz)
         end
         push!(cases, (Array(ra), num_ones) )
         test_dict[bitarray_2_num(ra)] = num_ones
      end
   end
   make_uniq_cases(train_cases, num_tcs)
   make_uniq_cases(test_cases,  num_tcs/2)
   return train_cases, test_cases
end

@with_kw mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 1    # batch size
    epochs::Int = 170       # number of epochs
    device::Function = cpu  # set as gpu, if gpu available
end


testX = trainX
testY = trainY

function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = trainX, trainY
    xtest, ytest   = testX,  testY
	
    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 1:NUMCLASSES), onehotbatch(ytest, 1:NUMCLASSES)

    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest, batchsize=args.batchsize)

    return train_data, test_data
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += Flux.mse(model(x), y)
    end
    l/length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += cpu(model(x) .== y)*(1 / size(x,2))
    end
    acc/length(data_loader)
end

 #rec_sqrt (x) = max(zero(x), sqrt(abs(x)))
function rec_sqrt(x) 
 x > 0 ? sqrt(x) : x
end



 # ativation func train accuracy  test accuracy:  final loss: 
 # relu:                0.688           0.7          1.768
 # relu: wid*3 in mid   0.775           0.775        1.679
 # relu6:               0.994           1.0          1.493         
 # celu:                0.993           1.0          1.467
 # gelu:                0.774           0.7606       1.679
 # selu:                0.882           0.883        1.575
 # mish:       crash         
 # rrelu:      crash
 # sigmoid:             0.345           0.351        2.21
#activation_fns = [ celu, elu, gelu, hardsigmoid, hardtanh, leakyrelu,
#                   lisht, logcosh, logsigmoid, mish, relu, relu6,
#                  rrelu, selu, sigmoid, softplus, softshrink, 
#                  softsign, swish, tanhshrink, trelu ]
 # Note: mish, rrelu do not work on GPU; trelu is redundant
activation_fns = [ rec_sqrt, celu, elu, gelu, hardsigmoid, hardtanh, leakyrelu,
                   lisht, logcosh, logsigmoid, relu, relu6,
                   selu, sigmoid, softplus, softshrink, 
                   softsign, tanhshrink ]

function build_model(act_fn)
   return Chain(
      Dense(WIDTH,   WIDTH*2,    act_fn ),
      Dense(WIDTH*2, WIDTH,      act_fn ),
      Dense(WIDTH,   1, act_fn )
      )
end

function train_it(af; kws...)
    # Initializing Model parameters 
    args = Args(; kws...)

    # Load Data
    #train_data,test_data = getdata(args)

    # Construct model
    m = build_model(af)
    #train_data = args.device.(train_data)
    #test_data = args.device.(test_data)
    m = args.device(m)
    loss(x,y) = Flux.mse(m(x), y)
    
    ## Training
    #evalcb = () -> @show(loss_all(train_data, m))
    evalcb = () -> @show(loss_all([(trainX,trainY)], m))
    opt = ADAM(args.η)
		
    #trainy = onehotbatch(trainY, 1:2)
    trainy = [[i==false, i==true] for i in trainY]
    trainy = vcat(trainy...)
    @show size(trainy)

    #@epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)
    @epochs args.epochs Flux.train!(loss, params(m), [(trainX,trainY)], opt, cb = evalcb)

    #@show accuracy(train_data, m)
    #@show accuracy([trainX,trainY], m)
    #@show accuracy(test_data, m)
    return m, [(trainX,trainY)]
end

examine(x, y, ypred) = (onecold(y) != onecold(ypred) && @show(x, sum(x), onecold(y), onecold(ypred), ypred))

examine_n(n, x, y, ypred) = (sum(x) == n && @show(x, sum(x), onecold(y), onecold(ypred), ypred))

examine_pred(pred, x, y, yp) = (pred && @show(x, sum(x), onecold(y), onecold(yp), yp))

function find_all_mismatching(dataset)
   for (x, y) in cpu.(dataset)
      ypred = cpu(model)(x)
      examine_pred.((onecold(y) != onecold(ypred)), eachcol(x), eachcol(y), eachcol(ypred))
   end
end

@with_kw mutable struct Entry
   passing::Int = 0
   failing::Int = 0
   #failvals::Dict = Dict()
end

function stat_entry(stats)
   function aux( y, ypred)
      ground_truth = onecold(y)
      if(ground_truth != onecold(ypred))
          if !haskey(stats, ground_truth )
             stats[ground_truth] = Entry()
          end
          stats[ground_truth].failing += 1
      else
          if !haskey(stats, ground_truth )
             stats[ground_truth] = Entry()
          end
          stats[ground_truth].passing += 1
      end
   end
   return aux
end

function stats(model, dataset)
   stats = Dict()
   stat_ent = stat_entry(stats)
   for (x,y) in cpu.(dataset)
      ypred = cpu(model)(x)
      stat_ent.(eachcol(y), eachcol(ypred))
   end
   total_passes = 0
   total_fails  = 0
   for (k,v) in sort(stats)
      println("$k =>\t  passes: $(v.passing)\t  fails: $(v.failing)")
      total_passes += v.passing
      total_fails  += v.failing
   end
   pass_percent = 100*(total_passes)/(total_passes + total_fails)
   @show pass_percent
   return stats
end

function run_fns(afns=activation_fns)
   for af in afns
      @show af
      model, train_data = train_it(af)
      println("Training Data stats: for $af")
      stats(model,train_data)
      println("\nTest Data stats: for $af")
      #stats(model,test_data)
      println("-----------------------------------------------------------\n")
   end
end

model, train_data = train_it(relu)

#run the best two activation fns 4x
#for i in 1:4
#   run_fns([rec_sqrt, tanhshrink])
#end

for i in 1:100
   @show (i, model(train_data[1][1][:,i]), train_data[1][1][:,i])
   end


