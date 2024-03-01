require(brms)
require(arm)
require(bayestestR)
require(tidyverse)
require(tidybayes)
require(cowplot)

#################################
#           EDA                 #
#################################



dpath = '/Users/xdchen/Downloads/eval2/data/data-out/'
opath = '/Users/xdchen/Downloads/eval2/data/plots/'
ddata = read.csv('/Users/xdchen/Downloads/eval2/data/data-out/selected_score6.csv', sep='|')
ddata$pl = ifelse(ddata$tl == ddata$rl, 1, 0)


# EDA 0 PL - PMI distribution
ggplot(ddata, aes(as.factor(pl), lmi)) +
    geom_boxplot() +
    theme_bw() +
    labs(x = "Acceptance Rate", y = "PMI") +
    scale_x_discrete(
        breaks = c('0', '1'),
        label = c("wrong", "correct")
    ) +
    facet_wrap(~dataset)
ggsave(paste(opath, 'facet-lmi-pl.png', sep=''), dpi=300, width = 7, height = 7)


# EDA 1 Cor between MI and Slen
set.seed(4)
dsub = ddata %>% group_by(dataset) %>% slice_sample(n = 100)

ggplot(dsub, aes(da, lmi)) +
    geom_smooth(method = 'lm') +
    theme_bw() +
    labs(x = "DA", y = "PMI")

main = ggplot(dsub, aes(slen, lmi)) +
            geom_point(alpha=0.2) +
            geom_smooth(method = 'lm') +
            theme_bw() +
            labs(x = "Sentence Length", y = "PMI")

xden = axis_canvas(main, axis = 'x') +
    geom_density(data=dsub, aes(slen), alpha=0.7)
yden = axis_canvas(main, axis = 'y', coord_flip = TRUE) +
    geom_density(data=dsub, aes(lmi), alpha=0.7)+
    coord_flip()
    
p1 = insert_xaxis_grob(main, xden, grid::unit(.2, 'null'), position='top')
p2 = insert_yaxis_grob(p1, yden, grid::unit(.2, 'null'), position='right')

ggdraw(p2)
ggsave(paste(opath, 'len-lmi.png', sep=''), dpi=300, width = 7, height = 7)

#################################
#           Model               #
#################################

# normalize data
sdata = ddata
sdata$slen = rescale(sdata$slen)
sdata$lmi = rescale(sdata$lmi)

mpath = paste(dpath, 'model4-lmi-len.rds', sep='')
mpath = paste(dpath, 'model9-n-lmi.rds', sep='')
model = readRDS(mpath)

summary(model)

# draw samples
draws <- model %>% spread_draws(
    b_Intercept,
    r_dataset[term,Intercept],
    b_lmi,
    b_slen,
)

# rename
draws = draws %>% rename(randomv = Intercept, dataset = term)
draws = draws %>% mutate(dataset)
draws = draws %>% pivot_wider(names_from = c('randomv'), values_from = c('r_dataset'))

# draws$r.lmi = draws$mi + draws$b_mi
draws$r.lmi = draws$lmi + draws$b_lmi
draws$r.slen = draws$slen + draws$b_slen
draws$r.inter = draws$Intercept + draws$b_Intercept

draws.mean = draws %>% group_by(dataset) %>% summarize(r.slen=mean(r.slen),
                                                       r.inter=mean(r.inter),
                                                       r.lmi=mean(r.lmi))

draws.mean = distinct(draws.mean)


minmax = ddata %>% group_by(dataset) %>% summarise(lmax = max(lmi), lmin = min(lmi))

m = mean(ddata$lmi)
d = sd(ddata$lmi)

minmax = minmax %>% mutate(norm_min = (lmin-m)/(2*d), norm_max = (lmax-m)/(2*d))

ot = data.frame()
for (val in c('boolq', 'copa', 'mrpc', 'qqp', 'rte', 'sst', 'winogrande'))
{
    lmax = subset(minmax, (dataset == val))['lmax'][[1]]
    lmin = subset(minmax, (dataset == val))['lmin'][[1]]
    nmax = subset(minmax, (dataset == val))['norm_max'][[1]]
    nmin = subset(minmax, (dataset == val))['norm_min'][[1]]
    xlin = seq(nmin, nmax, .1)
    llen = length(xlin)
    name = rep(val, each = llen)
    dfrm = data.frame(xlin,name)
    if (length(ot) == 0) {
        ot = dfrm
    } else {
        ot = rbind(ot, dfrm)
    }
}

ot = ot %>% rename(x = xlin, dataset = name)
d.m = full_join(ot, draws.mean, by="dataset")
d.m = d.m %>% mutate(y=invlogit(r.inter + r.lmi*x + r.slen*x),
                     o=x*(2*d)+m,
                     )

custom.col <- c("#FFDB6D", "#C4961A", "#D16103", "#C3D7A4", "#52854C", "#4E84C4", "#293352")

a = ggdraw() + draw_image(magick::image_read_pdf("/Users/xdchen/Downloads/eval2/data/plots/scheme.pdf", density = 600))

b = ggplot(subset(ddata, rndnum=='rnd1'), aes(lmi, pl)) +
    geom_smooth(data=ddata, aes(lmi, pl, color=dataset), 
                method = "glm", method.args = list(family=binomial), 
                se = F, size=1.2, linetype='solid') +
    labs(x = "PMI", y = "Acceptance Rate") +
    ylim(0, 1) +
    theme_bw() +
    scale_colour_manual(labels = c("BoolQ", "COPA", "MRPC", "QQP", "RTE", "SST", "WinoGrande"), values = custom.col) +
    theme(
        text = element_text(size = 13),
        # legend.position = c(0.8, 0.25),
        legend.background=element_blank(),
        legend.title = element_blank())

b = ggplot(d.m, aes(o, y, color=dataset)) +
    geom_line(size=1.3, linetype='solid') +
    ylim(0, 1) +
    labs(x = "PMI", y = "Acceptance Rate") +
    theme_bw() +
    scale_colour_manual(labels = c("BoolQ", "COPA", "MRPC", "QQP", "RTE", "SST", "WinoGrande"), values = custom.col) +
    theme(
          text = element_text(size = 13),
          legend.background=element_blank(),
          legend.title = element_blank())

plot_grid(a, b, labels = c('A', 'B'), ncol = 1, align = "v", rel_widths = c(1, 1.3))
ggsave(paste(opath, 'lmi-ar.png', sep=''), dpi=300, width = 6, height = 6.5)


# Model conditional effect
conditional_effects(model, effects = 'lmi', re_formula=NULL, ndraws = 500)$lmi -> freqPartial_df
head(freqPartial_df)
miu.eff = conditional_effects(model, effects = "lmi", points=T, spaghetti = T, ndraws = 500, method="fitted")
plot(miu.eff, plot = FALSE)[[1]] +
    scale_color_grey() +
    scale_fill_grey() +
    theme(text = element_text(size = 11)) +
    theme_bw() +
    xlab('Normalized PMI') +
    ylab('Acceptance Rate')
ggsave(paste(opath, 'con-lmi.png', sep=''), dpi=300, width = 4.5, height = 4.5)

# draws.mean = draws %>% summarize(dataset = dataset,
#                                  m.inter = mean(r.inter),
#                                  m.slen  = mean(r.slen),
#                                  m.lmi   = mean(r.lmi)
# )
# draws.mean = as.data.frame(draws.mean)
# draws.mean = draws.mean %>% distinct()

ggplot(data=draws, aes(y = dataset, x = r.slen)) +
    stat_halfeye() +
    geom_vline(xintercept = c(-.16, .16), linetype = "dashed") +
    scale_fill_manual(values = c("gray80", "skyblue")) +
    xlab("Estimated Slope of Sentence Length") +
    ylab("Task Name") +
    xlim(-2, 2) +
    scale_y_discrete(breaks=c("boolq","copa","mrpc","qqp","rte","sst","winogrande"),
                     labels=c("BoolQ","COPA","MRPC","QQP","RTE","SST","WG")) +
    annotate("rect", xmin = -.16, xmax = 0.16, ymin = 0, ymax = 8,
             alpha = .1, fill = "orange") +
    # guides(fill=guide_legend(title="ROPE"))+
    theme(legend.text = element_text(size = 11),
          legend.position=c(.1,.75)) +
    theme_bw()
ggsave(paste(opath, 'rnd-slen.png', sep=''), dpi=300, width = 4.5, height = 4.5)

ggplot(data=draws, aes(y = dataset, x = r.lmi)) +
    stat_halfeye() +
    geom_vline(xintercept = c(-.16, .16), linetype = "dashed") +
    scale_fill_manual(values = c("gray80", "skyblue")) +
    xlab("Estimated Slope") +
    ylab("Task Name") +
    xlim(-.5, 4) +
    scale_y_discrete(breaks=c("boolq","copa","mrpc","qqp","rte","sst","winogrande"),
                     labels=c("BoolQ","COPA","MRPC","QQP","RTE","SST","WG")) +
    annotate("rect", xmin = -.16, xmax = 0.16, ymin = 0, ymax = 8,
             alpha = .1, fill = "orange") +
    # guides(fill=guide_legend(title="ROPE"))+
    theme(legend.text = element_text(size = 11),
          legend.position=c(.1,.75)) +
    theme_bw()
ggsave(paste(opath, 'rnd-lmi.png', sep=''), dpi=300, width = 4.5, height = 4.5)

ggplot(data=draws, aes(y = dataset, x = r.inter)) +
    stat_halfeye() +
    geom_vline(xintercept = c(-.16, .16), linetype = "dashed") +
    scale_fill_manual(values = c("gray80", "skyblue")) +
    xlab("Estimated Intercept") +
    ylab("Task Name") +
    xlim(-.5, 4) +
    scale_y_discrete(breaks=c("boolq","copa","mrpc","qqp","rte","sst","winogrande"),
                     labels=c("BoolQ","COPA","MRPC","QQP","RTE","SST","WG")) +
    annotate("rect", xmin = -.16, xmax = 0.16, ymin = 0, ymax = 8,
             alpha = .1, fill = "orange") +
    theme(legend.text = element_text(size = 11),
          legend.position=c(.1,.75)) +
    theme_bw()
ggsave(paste(opath, 'rnd-int.png', sep=''), dpi=300, width = 4.5, height = 4.5)
