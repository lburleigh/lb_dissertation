library(dplyr)
library(purrr)
library(readr)
library(tidyr)
library(ggplot2)

d <- read_csv("phase-B_exp-task_roi-amyg_bi_thr50_dv-csp_csm.csv")

d_subj <- d %>%
  group_by(target_subject, single, model_type) %>%
  summarize(across(c(starts_with("acc_"), starts_with("loss_")), list(avg = mean, std = sd))) %>%
  unite(cond, model_type, single) %>%
  pivot_longer(
    c(starts_with("acc_"), starts_with("loss_")),
    names_to = c("metric", "subset", "stat"),
    names_sep = "_",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  )
  
ggplot(d_subj, aes(x = target_subject, y = avg, color = cond)) +
  geom_point() +
  facet_grid(metric ~ subset, scales = "free_y")

se <- function(x) {
  return(sd(x) / sqrt(length(x)))
}
d_avg <- d_subj %>%
  group_by(cond, metric, subset) %>%
  summarize(se = se(avg), avg = mean(avg))

ggplot(d_avg, aes(x = cond, y = avg, fill = cond)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = avg - se, ymax = avg + se), position = position_dodge()) +
  facet_grid(metric ~ subset, scales = "free_y")
  
gkeys <- d_subj %>%
  group_by(metric, subset) %>%
  group_keys() %>%
  unite(label, metric, subset)

ttests <- d_subj %>%
  group_by(metric, subset) %>%
  group_split() %>%
  map(~{
    list(
      "cf_v_rf" = t.test(avg ~ cond, paired = TRUE, data = .x %>% filter(cond %in% c("coirls_FALSE", "ridgels_FALSE"))),
      "cf_v_rt" = t.test(avg ~ cond, paired = TRUE, data = .x %>% filter(cond %in% c("coirls_FALSE", "ridgels_TRUE"))),
      "rf_v_rt" = t.test(avg ~ cond, paired = TRUE, data = .x %>% filter(cond %in% c("ridgels_FALSE", "ridgels_TRUE")))
    )
  })

names(ttests) <- gkeys$label