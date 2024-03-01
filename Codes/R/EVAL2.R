require(tidyverse)
require(lme4)
require(lmerTest)
library(arm)
library(parameters)

#################################
#           EDA                 #
#################################

dpath = '/Users/xdchen/Downloads/eval2/data/data-out/'
opath = '/Users/xdchen/Downloads/eval2/data/plots/'

ddata = read.csv('/Users/xdchen/Downloads/eval2/data/data-out/selected_score6.csv', sep='|')
ddata$pl = ifelse(ddata$tl == ddata$rl, 1, 0)


ddata %>% group_by(dataset) %>% summarise(score = mean(mi))

ggplot(ddata, aes(lmi, da)) +
    geom_smooth(method = 'lm')

ggplot(ddata, aes(mi, pl)) +
    geom_point()

ggplot(ddata, aes(lmi, pl)) +
    geom_point()

ggplot(ddata, aes(lmi, pl)) +
    geom_smooth(data=ddata, aes(lmi, pl, color=dataset),
                method = "glm", method.args = list(family=binomial),
                se = F, size=1.2, linetype='dashed') +
    geom_smooth(data=ddata, aes(lmi, pl),
                method = "glm", method.args = list(family=binomial), size=1.2) +
    labs(x = "PMI", y = "Acceptance Rate") +
    ylim(0, 1) +
    # scale_y_continuous(
    #     breaks = c(0, 1),
    #     label = c("0", "1")) +
    theme_bw() +
    theme(legend.position = c(0.8, 0.25),
          text = element_text(size = 11),
          legend.background=element_blank(),
          legend.title = element_blank())

ggsave(paste(opath, 'glm-mi.png', sep=''), dpi=300, width = 5, height = 5)

ggplot(ddata, aes(as.factor(pl), mi)) +
    geom_boxplot() +
    theme_bw() +
    labs(x = "Acceptance Rate", y = "PMI") +
    scale_x_discrete(
        breaks = c('0', '1'),
        label = c("wrong", "correct")
    ) +
    facet_wrap(~dataset)


#################################
#           Model               #
#################################

# normalize data
sdata = ddata
sdata$slen = rescale(sdata$slen)
sdata$da = rescale(sdata$da)
sdata$mi = rescale(sdata$mi)
# Get pred acc
sdata$pl = ifelse(sdata$tl == sdata$rl, 1, 0)


sdata = ddata

model1 = glmer(pl ~ da + slen + (1 + da | dataset), sdata, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
summary(model1)

se <- sqrt(diag(vcov(model1)))
# table of estimates with 95% CI
tab <- cbind(Est = fixef(model1), LL = fixef(model1) - 1.96 * se, UL = fixef(model1) + 1.96 * se)

exp(tab)

model2 = glmer(pl ~ da + (1 + da | dataset), sdata, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
summary(model2)
ranef(model2)$dataset + fixef(model2)[['da']]

model2 = glmer(pl ~ mi + (1 + mi | dataset), sdata, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
summary(model2)
summary(model2)
ranef(model2)$dataset + fixef(model2)[['mi']]

beta1 <- fixef(model1)[['da']]
offset1 <- ranef(model1)$dataset
effect1 <- (beta1 + offset1)
effect.diff1 = data.frame(Effects=effect1)
colnames(effect.diff1) = c('effect-da', 'diff.eff')

beta2 <- fixef(model2)[['mi']]
offset2 <- ranef(model2)$dataset
effect2 <- (beta2 + offset2)
effect.diff2 = data.frame(Effects=effect2)
colnames(effect.diff2) = c('effect-mi', 'diff.eff')

effect.diff1$dataset = rownames(effect.diff1)
effect.diff2$dataset = rownames(effect.diff2)

imi_join = full_join(effect.diff1, effect.diff2, by="dataset")

ggplot(aes(effect-mi, effect-da), data=imi_join) +
    geom_point() +
    geom_text(data=imi_join, aes(effect-mi+0.2, effect-da+0.2, label=dataset)) +
    geom_density(data=imi_join, aes(effect-mi, effect-da), alpha = 0.4) 
    # geom_vline(aes(xintercept=0),lty=2) +
    # geom_vline(aes(xintercept=beta1),color='red') +
    xlab("Estimated MI Effect") +
    ylab("Estimated DA Effect") +
    theme_bw()

