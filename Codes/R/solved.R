# Load packs
require(brms)
require(arm)
require(bayestestR)
require(tidyverse)
require(tidybayes)
require(cowplot)

# probe acc

proper_noun <- c(0.458, 0.510, 0.479, 0.583, 0.437, 0.572)
common_noun <- c(0.946, 0.980, 0.909, 0.955, 0.961, 0.935)

noun.df = data.frame(proper_noun, common_noun)
noun.df = noun.df %>% pivot_longer(cols=proper_noun:common_noun, names_to = 'noun', values_to = 'score')

# 0.948; 0.506
noun.mn = noun.df %>% group_by(noun) %>%  summarise(score=mean(score))

ggplot(noun.df, aes(noun, score)) +
    stat_summary(fun.data = mean_se, 
                 geom = "errorbar") + 
    geom_bar(data=noun.mn, aes(x=noun,y=score), stat = 'identity', alpha=0.3) +
    ylim(0, 1) +
    labs(x = "", y = "Probe Accuracy") +
    theme_bw()

ggsave(paste(opath, 'probe-acc.png', sep=''), dpi=300, width = 4, height = 4)

# example plot
data(mtcars)
dat <- subset(mtcars, select=c(mpg, am, vs))

ggplot(dat, aes(x=mpg, y=vs)) + 
    geom_line(stat = "smooth", method = "glm", method.args=list(family="binomial"), aes(x = mpg, y = vs), color = "blue", size = 1.8, alpha = 0.6) +
    geom_smooth(method = 'glm', method.args=list(family="binomial"), aes(x = mpg, y = vs), color = NA, size = 2, alpha = 0.15) +
    labs(x = "I(S;T)", y = "Acceptance Rate") +
    ylim(0, 1) +
    theme(text = element_text(size = 11),
          axis.text.x=element_blank(),
          ) +
    scale_x_continuous(
        breaks = c(10, 20, 30),
        label = c("1", "2", "3")
    ) +
    theme_bw()
ggsave(paste(opath, 'slen-mi2.png', sep=''), dpi=300, width = 4.5, height = 3)

# Load Model
ipath = '/Users/xdchen/OneDrive - McGill University/McGill/Eval2/Codes/R/'
opath = '/Users/xdchen/Downloads/eval2/data/plots/'

mpath1 = paste(ipath, 'model-mi-slen.rds', sep='')
model1 <- readRDS(mpath1)

mpath2 = paste(ipath, 'model-mi.rds', sep='')
model2 <- readRDS(mpath)

# Compare model 1
m1 <- add_criterion(model, c('loo'), cores=4)
m2 <- add_criterion(model2, c('loo'), cores=4)
loo_compare(m1, m2)

# Compare model 2
bayes_factor(m1, m2)


# Load data
ddata = read.csv('/Users/xdchen/Downloads/eval2/data/data-out/selected_score3.csv', 
                 sep='|')
ddata$pl = ifelse(ddata$tl == ddata$rl, 1, 0)
ggplot(ddata, aes(as.factor(pl), mi)) +
    geom_boxplot() +
    theme_bw() +
    labs(x = "Acceptance Rate", y = "PMI") +
    scale_x_discrete(
        breaks = c('0', '1'),
        label = c("wrong", "correct")
    ) +
    facet_wrap(~dataset)
ggsave(paste(opath, 'box.png', sep=''), dpi=300, width = 7, height = 7)


ggplot(ddata, aes(mi, da)) +
    geom_point(alpha=0.2) +
    geom_smooth(method = 'lm') +
    theme_bw() +
    labs(x = "PMI", y = "DA")
    

# EDA 1 Cor between MI and Slen
# Get pred acc
dsub = ddata %>% group_by(dataset) %>% slice_sample(n = 200)

main = ggplot(dsub, aes(slen, mi)) +
    geom_point(alpha=0.2) +
    geom_smooth(method = 'lm') +
    theme_bw() +
    labs(x = "Sentence Length", y = "PMI")

xden = axis_canvas(main, axis = 'x') +
    geom_density(data=dsub, aes(slen), alpha=0.7)
yden = axis_canvas(main, axis = 'y', coord_flip = TRUE) +
    geom_density(data=dsub, aes(mi), alpha=0.7)+
    coord_flip()

p1 = insert_xaxis_grob(main, xden, grid::unit(.2, 'null'), position='top')
p2 = insert_yaxis_grob(p1, yden, grid::unit(.2, 'null'), position='right')

ggdraw(p2)
ggsave(paste(opath, 'slen-mi.png', sep=''), dpi=300, width =5, height = 5)
    
# EDA 2 Cor between PMI and ACC
ggplot(ddata, aes(mi, pl)) +
    geom_smooth(data=ddata, aes(mi, pl, color=dataset), 
                method = "glm", method.args = list(family=binomial), 
                se = F, size=0.8, linetype='dashed') +
    geom_smooth(data=ddata, aes(mi, pl),
                method = "glm", method.args = list(family=binomial), size=1.5) +
    labs(x = "PMI", y = "Acceptance Rate") +
    ylim(0, 1) +
    theme_bw() +
    theme(legend.position = c(0.8, 0.25),
          text = element_text(size = 11),
          legend.background=element_blank(),
          legend.title = element_blank())
ggsave(paste(opath, 'glm-mi.png', sep=''), dpi=300, width = 5, height = 5)

# Model anala
# Get variables
# get_variables(model)
model = model1
summary(model)
# a = prepare_predictions(model,re_formula = NULL,ndraws = 500, resp = 'pl')

# draw samples
draws <- model %>% spread_draws(
    b_Intercept,
    r_dataset[term,Intercept],
    b_mi,
    b_slen,
)

# rename
draws = draws %>% rename(
    randomv = Intercept,
    dataset = term
)

# draws %>% add_epred_draws(model)

draws = draws %>% pivot_wider(names_from = c('randomv'), values_from = c('r_dataset'))

draws$r.mi = draws$mi + draws$b_mi
draws$r.slen = draws$slen + draws$b_slen
draws$r.inter = draws$Intercept + draws$b_Intercept

# Get END2 but in posterior
draws.mean = draws %>% group_by(dataset) %>% summarize(r.slen=mean(r.slen),
                                                       r.inter=mean(r.inter),
                                                       r.mi=mean(r.mi),
                                                       )
x = seq(-2, 2.5, .01)
l = length(x)
x = rep(x, times = 7)
dataset = c('boolq', 'copa', 'mrpc', 'qqp', 'rte', 'sst', 'winogrande')
dataset = rep(dataset, each = l)

d = data.frame(x,dataset)
d.m = full_join(d, draws.mean, by="dataset")

d.m = d.m %>% mutate(y=invlogit(r.inter + r.mi*x + r.slen*x))

ggplot(d.m, aes(x, y, color=dataset)) +
    geom_line(size=1.3, linetype='dashed') +
    ylim(0, 1) +
    labs(x = "Normalized PMI", y = "Acceptance Rate") +
    # ggtitle(expression(paste("y = ", logit^-1,"(-6.13 + 5.46*z.balance)"))) + 
    theme_bw() +
    theme(legend.position = c(0.8, 0.25),
          text = element_text(size = 11),
          legend.background=element_blank(),
          legend.title = element_blank())
ggsave(paste(opath, 'pos-mi-acc.png', sep=''), dpi=300, width = 4.5, height = 4.5)

# plot draws from posterior

# Conditional effect
conditional_effects(model, effects = 'mi', re_formula=NULL, ndraws = 500)$mi -> freqPartial_df
head(freqPartial_df)
miu.eff = conditional_effects(model, effects = "mi", points=T, spaghetti = T, ndraws = 500, method="fitted")
plot(miu.eff, plot = FALSE)[[1]] +
    scale_color_grey() +
    scale_fill_grey() +
    theme(text = element_text(size = 11)) +
    theme_bw() +
    xlab('Normalized PMI') +
    ylab('Acceptance Rate')
ggsave(paste(opath, 'pos-reg.png', sep=''), dpi=300, width =5, height = 5)


draws.mean = draws %>% summarize(dataset = dataset,
                              m.inter = mean(r.inter),
                              m.slen  = mean(r.slen),
                              m.mi    = mean(r.mi)
                              )

draws.mean = as.data.frame(draws.mean)
draws.mean = draws.mean %>% distinct()

ggplot(draws.mean, aes(m.inter, m.mi)) +
    geom_point(data=draws.mean, aes(m.inter, m.mi)) +
    geom_text(data=draws.mean, aes(m.inter+0.15, m.mi+0.15, label=dataset)) +
    geom_density_2d(data=draws.mean, aes(m.inter, m.mi), alpha = 0.4) +
    xlab("Estimated Random Intercepts of Tasks") +
    ylab("Estimated Random Slopes of Tasks") +
    geom_vline(xintercept = c(.8), linetype = "dashed") +
    geom_hline(yintercept = c(.8), linetype = "dashed") +
    annotate("rect", xmin = 0, xmax = 0.8, ymin = 0.8, ymax = 5,
             alpha = .1, fill = "orange") +
    annotate("rect", xmin = 0, xmax = 4, ymin = 0, ymax = 0.8,
             alpha = .1, fill = "orange") +
    xlim(0, 4)+
    ylim(0, 5)+
    theme(legend.text = element_text(size = 11), legend.position=c(.15,.85)) +
    # guides(fill=guide_legend(title="ROPE"))+
    theme_bw()
ggsave(paste(opath, 'cor-in-mi.png', sep=''), dpi=300, width = 7, height = 7)


ggplot(data=draws, aes(y = dataset, x = r.mi)) +
    stat_halfeye() +
    geom_vline(xintercept = c(0, .8), linetype = "dashed") +
    scale_fill_manual(values = c("gray80", "skyblue")) +
    xlab("Random Slope of Tasks") +
    ylab("Task Name") +
    annotate("rect", xmin = 0, xmax = 0.8, ymin = 0, ymax = 8,
             alpha = .1, fill = "orange") +
    # guides(fill=guide_legend(title="ROPE"))+
    theme(legend.text = element_text(size = 11),
          legend.position=c(.1,.75)) +
    theme_bw()
ggsave(paste(opath, 'rnd-slp.png', sep=''), dpi=300, width = 4.5, height = 4.5)


ggplot(data=draws, aes(y = dataset, x = r.inter)) +
    stat_halfeye() +
    geom_vline(xintercept = c(0, .8), linetype = "dashed") +
    scale_fill_manual(values = c("gray80", "skyblue")) +
    xlab("Random Intercept of Tasks") +
    ylab("Task Name") +
    annotate("rect", xmin = 0, xmax = 0.8, ymin = 0, ymax = 8,
             alpha = .1, fill = "orange") +
    theme(legend.text = element_text(size = 11)) +
    theme_bw()
ggsave(paste(opath, 'rnd-int.png', sep=''), dpi=300, width = 4.5, height = 4.5)


ggplot(data=draws, aes(y = dataset, x = r.slen, fill = stat(abs(x) < .8))) +
    stat_halfeye() +
    geom_vline(xintercept = c(0, .8), linetype = "dashed") +
    scale_fill_manual(values = c("gray80", "skyblue")) +
    xlab("Random Intercept of Tasks") +
    ylab("Task Name") +
    annotate("rect", xmin = -.8, xmax = 0.8, ymin = 0, ymax = 8,
             alpha = .1, fill = "orange") +
    theme(legend.text = element_text(size = 11)) +
    guides(fill=guide_legend(title="ROPE"))+
    theme_bw()
ggsave(paste(opath, 'rnd-sln.png', sep=''), dpi=300, width = 4.5, height = 4.5)


