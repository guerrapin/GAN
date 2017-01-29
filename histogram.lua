function histogram(data, n_interval)
   mean = torch.mean(data)
   var = torch.var(data)
   data_size = data:size()[1]
   min = torch.min(data)
   max = torch.max(data)

   hist = torch.zeros(n_interval,2)

   for i = 1,n_interval do
      hist[i][1] = min + (max-min)/n_interval*(i-1)
   end

   for i = 1 , data_size do
      not_found = true
      iterator =n_interval
      while not_found do
         if data[i][1] >= hist[iterator][1] then
            not_found = false
            hist[iterator][2] = hist[iterator][2]+1/data_size
         else
            iterator = iterator - 1
         end
      end
   end
   return hist
end

