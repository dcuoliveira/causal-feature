rm(list=ls())
gc()
library('SparseTSCGM')
library('data.table')
library('here')

seed = 221994
sparse_gaussian_var = sim.data(model=c("ar1"),
                               time=1,
                               n.obs=5479,
                               n.var=200,
                               seed=seed,
                               prob0=0.1,
                               network=c("random"))
df1_list = list()
names = c()
for (i in 1:dim(sparse_gaussian_var$data1)[2]){
  names = append(names, paste0('var', toString(i)))
  df1_list[[paste0('var', toString(i))]] = as.data.frame(sparse_gaussian_var$data1[,i])
}
df1 = do.call('cbind', df1_list) %>% as.data.table()
colnames(df1) = names
df1$date = seq(as.Date("2006-01-01"), as.Date("2020-12-31"), by=1)
df1 = df1 %>% dplyr::select(date, everything())
write.csv(df1, here("src/data/simulation/gaussian_sparse_var_df.csv"))


