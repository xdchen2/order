# Load packs
require(tidyverse)
require(lme4)
require(lmerTest)
require(arm)

dpath = '/Users/xdchen/Downloads/eval2/data/data-out/'
ddata = read.csv('/Users/xdchen/Downloads/eval2/data/data-out/selected_score6.csv', sep='|')
ddata$pl = ifelse(ddata$tl == ddata$rl, 1, 0)

sdata = ddata
sdata$slen = rescale(sdata$slen)
sdata$lmi = rescale(sdata$lmi)

ggplot(subset(ddata, rndnum=='rnd1'), aes(lmi, pl)) +
    geom_smooth(data=ddata, aes(lmi, pl, color=dataset), 
                method = "glm", method.args = list(family=binomial), 
                se = F, size=1.2, linetype='dashed') +
    labs(x = "log PMI", y = "Acceptance Rate") +
    ylim(0, 1) +
    theme_bw() +
    theme(legend.position = c(0.8, 0.25),
          text = element_text(size = 11),
          legend.background=element_blank(),
          legend.title = element_blank())

set.seed(42)
trainsplit = sdata$pl %>%
  createDataPartition(p = 0.8, list = FALSE)
trainset = sdata[trainsplit, ]
testiset = sdata[-trainsplit, ]


model1 = glmer(pl ~ lmi + slen + (1 + slen + lmi | dataset), sdata, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))


predi <- model1 %>% predict(test.testiset)

summary(model1)
require(MuMIn)
r.squaredGLMM(model1)

# R2m       R2c
# theoretical 0.03995649 0.3556992
# delta       0.01913049 0.1703027
# R2c is interpreted as a variance explained by the entire model, including both fixed and random effects
# R2m represents the variance explained by the fixed effects

RMSE(predi, testiset$pl)
R2(predi, testiset$pl)

# VIF between 1 and 5: variables are moderately correlated 
car::vif(model1)
# VIF: 3.715924
