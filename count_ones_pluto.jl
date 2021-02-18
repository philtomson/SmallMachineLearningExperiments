### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ cc63d738-70b2-11eb-0557-1b9d42215962
begin
	using StatsBase
    using Flux, Statistics
    using Flux.Data: DataLoader
    using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
    using Base.Iterators: repeated
    using Parameters: @with_kw
    using CUDA
    using BSON: @save
	using Plots
	using PlutoUI
end
	

# ╔═╡ 3a7c7e5a-70b3-11eb-124e-77f320b54698
#Check if CUDA is available

# ╔═╡ 4f104e14-70b3-11eb-0cf8-794686c238cc
if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

# ╔═╡ 62a3e940-70b3-11eb-1844-9d66a7d3bcdc


# ╔═╡ 6d7e80dc-70b3-11eb-26c2-e5d3ce20f4ad

function rand_bitarray(len)
   rand(len) .< 0.5
end



# ╔═╡ 79f7f866-70b3-11eb-17ae-8dbc8c26e3a9
 
function make_rand_array(num_ones, len)
   ary = zeros(len)
   rnd_idxs = sample(1:len, num_ones, replace=false)
   for idx in rnd_idxs
      ary[idx] = 1
   end
   BitArray(ary)
end

# ╔═╡ acfeb984-70b3-11eb-06ad-2bb0cb6c692b
function bitarray_2_num(arr)
   arr = reverse(arr)
   sum(((i, x),) -> Int(x) << ((i-1) * sizeof(x)), enumerate(arr.chunks))
end

# ╔═╡ c08c9534-70b3-11eb-1b39-f1319b326095
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



# ╔═╡ cc9bc214-70b3-11eb-10cf-51dea4ce61dc
@with_kw mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 200    # batch size
    epochs::Int = 70       # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end

# ╔═╡ d8bc44e2-70b3-11eb-13f3-b1839d05dddf
begin
	NUMCLASSES = 10
    WIDTH = 60

    train, test = create_testcases(20000, WIDTH, NUMCLASSES)
    trainX = [ x[1] for x in train ]
    trainX = hcat(trainX...)
    trainY = [ x[2] for x in train ]
    testX =  [ x[1] for x in test ]
    testX =  hcat(testX...)
    testY =  [ x[2] for x in test ]

end

# ╔═╡ f8fc9842-70b3-11eb-068d-a31cf806a2b2
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


# ╔═╡ 081fc4de-70b4-11eb-1b5a-230eb58ba240
function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

# ╔═╡ 143cb6f0-70b4-11eb-1189-f3d541869fc0
function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end


# ╔═╡ 200b58a6-70b4-11eb-00b0-e50d3386e99a
function rec_sqrt(x) 
 x > 0 ? sqrt(x) : x
end

# ╔═╡ 29f31c6e-70b4-11eb-12d0-23362327c02b
 activation_fns = [ rec_sqrt, celu, elu, gelu, hardsigmoid, hardtanh, leakyrelu,
                   lisht, logcosh, logsigmoid, relu, relu6,
                   selu, sigmoid, softplus, softshrink, 
                   softsign, tanhshrink ]


# ╔═╡ 43bbf76a-70b4-11eb-14d2-7bc8ba582c9f
function build_model(act_fn)
   return Chain(
      Dense(WIDTH,   WIDTH*2,    act_fn ),
      Dense(WIDTH*2, WIDTH,      act_fn ),
      Dense(WIDTH,   NUMCLASSES, act_fn ),
      softmax)
end

# ╔═╡ 50ef934c-70b4-11eb-3e97-6fd6ea146302
function train_it(af; kws...)
    # Initializing Model parameters 
    args = Args(; kws...)

    # Load Data
    train_data,test_data = getdata(args)

    # Construct model
    m = build_model(af)
    train_data = args.device.(train_data)
    test_data = args.device.(test_data)
    m = args.device(m)
    loss(x,y) = logitcrossentropy(m(x), y)
    
    ## Training
    evalcb = function ()
		display!(plot!(float(loss_all(train_data, m)), ylim=(0,7)))
	end
    opt = ADAM(args.η)
		
    @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

    @show accuracy(train_data, m)
    @show accuracy(test_data, m)
    return m, train_data, test_data
end


# ╔═╡ 606f1c7a-70b4-11eb-3186-df64439ecd5f
@with_kw mutable struct Entry
   passing::Int = 0
   failing::Int = 0
   #failvals::Dict = Dict()
end


# ╔═╡ 744e760a-70b4-11eb-0b68-41d3a4b164c6
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



# ╔═╡ 7e5ebd76-70b4-11eb-0d80-23573651204e
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


# ╔═╡ 88cce31e-70b4-11eb-3bec-7d78de9bdb93
function run_fns(afns=activation_fns)
   for af in afns
      @show af
      model, train_data, test_data = train_it(af)
      println("Training Data stats: for $af")
      stats(model,train_data)
      println("\nTest Data stats: for $af")
      stats(model,test_data)
      println("-----------------------------------------------------------\n")
   end
end


# ╔═╡ 947bea48-70b4-11eb-2cbc-ff0fd15e7dfe
for i in 1:4
   run_fns([rec_sqrt, tanhshrink])
end

# ╔═╡ Cell order:
# ╠═cc63d738-70b2-11eb-0557-1b9d42215962
# ╠═3a7c7e5a-70b3-11eb-124e-77f320b54698
# ╠═4f104e14-70b3-11eb-0cf8-794686c238cc
# ╠═62a3e940-70b3-11eb-1844-9d66a7d3bcdc
# ╠═6d7e80dc-70b3-11eb-26c2-e5d3ce20f4ad
# ╠═79f7f866-70b3-11eb-17ae-8dbc8c26e3a9
# ╠═acfeb984-70b3-11eb-06ad-2bb0cb6c692b
# ╠═c08c9534-70b3-11eb-1b39-f1319b326095
# ╠═cc9bc214-70b3-11eb-10cf-51dea4ce61dc
# ╠═d8bc44e2-70b3-11eb-13f3-b1839d05dddf
# ╠═f8fc9842-70b3-11eb-068d-a31cf806a2b2
# ╠═081fc4de-70b4-11eb-1b5a-230eb58ba240
# ╠═143cb6f0-70b4-11eb-1189-f3d541869fc0
# ╠═200b58a6-70b4-11eb-00b0-e50d3386e99a
# ╠═29f31c6e-70b4-11eb-12d0-23362327c02b
# ╠═43bbf76a-70b4-11eb-14d2-7bc8ba582c9f
# ╠═50ef934c-70b4-11eb-3e97-6fd6ea146302
# ╠═606f1c7a-70b4-11eb-3186-df64439ecd5f
# ╠═744e760a-70b4-11eb-0b68-41d3a4b164c6
# ╠═7e5ebd76-70b4-11eb-0d80-23573651204e
# ╠═88cce31e-70b4-11eb-3bec-7d78de9bdb93
# ╠═947bea48-70b4-11eb-2cbc-ff0fd15e7dfe
