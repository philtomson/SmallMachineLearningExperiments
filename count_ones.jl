using StatsBase
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using BSON: @save

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function rand_bitarray(len)
   rand(len) .< 0.5
end

function make_rand_array(num_ones, len)
   ary = zeros(len)
   rnd_idxs = sample(1:len, num_ones, replace=false)
   for idx in rnd_idxs
      ary[idx] = 1
   end
   BitArray(ary)
end

function bitarray_2_num(arr)
   arr = reverse(arr)
   sum(((i, x),) -> Int(x) << ((i-1) * sizeof(x)), enumerate(arr.chunks))
end

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
    batchsize::Int = 100    # batch size
    epochs::Int = 70       # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end

NUMCLASSES = 10
WIDTH = 60

train, test = create_testcases(20000, WIDTH, NUMCLASSES)
trainX = [ x[1] for x in train ]
trainX = hcat(trainX...)
trainY = [ x[2] for x in train ]
testX =  [ x[1] for x in test ]
testX =  hcat(testX...)
testY =  [ x[2] for x in test ]

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
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
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
function build_model()
   return Chain(
      Dense(WIDTH,   WIDTH*2,    relu6 ),
      Dense(WIDTH*2, WIDTH,      relu6 ),
      Dense(WIDTH,   NUMCLASSES, relu6 ),
      softmax)
end

function train_it(; kws...)
    # Initializing Model parameters 
    args = Args(; kws...)

    # Load Data
    train_data,test_data = getdata(args)

    # Construct model
    m = build_model()
    train_data = args.device.(train_data)
    test_data = args.device.(test_data)
    m = args.device(m)
    loss(x,y) = logitcrossentropy(m(x), y)
    
    ## Training
    evalcb = () -> @show(loss_all(train_data, m))
    opt = ADAM(args.η)
		
    @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

    @show accuracy(train_data, m)
    @show accuracy(test_data, m)
    return m, train_data, test_data
end

examine(x, y, ypred) = (onecold(y) != onecold(ypred) && @show(x, sum(x), onecold(y), onecold(ypred), ypred))

examine_n(n, x, y, ypred) = (sum(x) == n && @show(x, sum(x), onecold(y), onecold(ypred), ypred))

function find_all_mismatching(dataset)
   for (x, y) in cpu.(dataset)
      ypred = cpu(model)(x)
      examine.(eachcol(x), eachcol(y), eachcol(ypred))
   end
end

model, train_data, test_data = train_it()

find_all_mismatching(test_data)


