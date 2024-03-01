require(tidyverse)

# pred scores
ddata = read.csv('/Users/xdchen/Downloads/eval2/data/data-out/selected_score6.csv', sep='|')
ddata = subset(ddata, ddata$rndnum == 'rnd1')
ddata$pl = ifelse(ddata$tl == ddata$rl, 1, 0)
# ddata = subset(ddata, ddata$tl == ddata$rl)
ddata = subset(ddata, select = c(inum, dataset, pl))
ddata = subset(ddata, ddata$dataset != c('boolq'))

ddata$inum = as.character(ddata$inum)


dpath = '/Users/xdchen/Downloads/eval2/data/data-out/'
opath = '/Users/xdchen/Downloads/eval2/data/plots/'


ot = data.frame()
for (TASK in c('sst', 'qqp', 'copa', 'winogrande', 'mrpc', 'rte')) {
    mpath = paste(dpath, 'rnd1-', TASK, '-score.csv', sep='')
    dfrmd = read.csv(mpath, sep='|')
    dfrmd = subset(dfrmd, select=c(inum, org, rnd, gnt, cp, mp, slen))
    dfrmd$dataset = TASK
    dfrmd$cp = dfrmd$cp / dfrmd$slen
    dfrmd$mp = dfrmd$mp / dfrmd$slen
    dfrmd$pmi = dfrmd$cp - dfrmd$mp
    dfrmd = subset(dfrmd, dfrmd$slen < 21)
    if (length(ot) == 0) {
        ot = dfrmd
    } else {
        ot = rbind(ot, dfrmd)
    }
}

ot$inum = as.character(ot$inum)

df = full_join(ot, ddata, by=c("dataset", "inum"))
df = na.omit(df)

df = df %>% mutate_if(is.numeric, round, digits = 4)

write.csv(df, "/Users/xdchen/Downloads/eval2/data/data-out/sample.csv", row.names=FALSE)
