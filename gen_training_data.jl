#bitArray parity
parity(x) = isodd(sum(x))

function create_parity_testcases(sz)
   train_cases = []
   for i in 0:(2^sz-1)
     bin = digits(i, base=2, pad=sz)
     push!(train_cases, (bin, parity(bin)))
     @show (bin, parity(bin))
   end
   return train_cases
end

