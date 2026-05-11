library(data.table)
library(knitr)

# ---------------------------------------------------------
# Core Estimators (Mapping estimating equations to sequential GLMs)
# ---------------------------------------------------------
est_case1 <- function(df) {
  m <- glm(Y ~ A * W, data = df[S == 1], family = binomial)
  
  df1 <- copy(df)[, A := 1]
  df0 <- copy(df)[, A := 0]
  
  y1_hat <- predict(m, newdata = df1, type = "response")
  y0_hat <- predict(m, newdata = df0, type = "response")
  
  rd_std <- mean(y1_hat) - mean(y0_hat)
  rd_mod <- mean(y1_hat[df$A == 1]) - mean(y0_hat[df$A == 0])
  
  return(c("Standard" = rd_std, "Modified" = rd_mod))
}

est_case2 <- function(df) {
  df_s1 <- df[S == 1]
  df1 <- copy(df)[, A := 1]
  df0 <- copy(df)[, A := 0]
  
  # 1-3. Standard G-computation variants
  m_x <- glm(Y ~ A + X, data = df_s1, family = binomial)
  m_z <- glm(Y ~ A + Z, data = df_s1, family = binomial)
  m_xz <- glm(Y ~ A + X + Z, data = df_s1, family = binomial)
  
  rd_x <- mean(predict(m_x, df1, type="response")) - mean(predict(m_x, df0, type="response"))
  rd_z <- mean(predict(m_z, df1, type="response")) - mean(predict(m_z, df0, type="response"))
  
  y1_inner <- predict(m_xz, df1, type="response")
  y0_inner <- predict(m_xz, df0, type="response")
  rd_xz <- mean(y1_inner) - mean(y0_inner)
  
  # 4. Nested G-computation
  df_temp <- copy(df)
  df_temp[, `:=`(y1_inner = y1_inner, y0_inner = y0_inner)]
  
  # Suppress fractional binomial warnings for outer models
  suppressWarnings({
    m_out1 <- glm(y1_inner ~ A + Z, data = df_temp, family = binomial)
    m_out0 <- glm(y0_inner ~ A + Z, data = df_temp, family = binomial)
  })
  
  rd_nested <- mean(predict(m_out1, df1, type="response")) - mean(predict(m_out0, df0, type="response"))
  
  return(c("X-only" = rd_x, "Z-only" = rd_z, "X+Z" = rd_xz, "Nested" = rd_nested))
}

# ---------------------------------------------------------
# Bootstrap Wrapper & Table Generation
# ---------------------------------------------------------
run_comparison_table <- function(df, est_func, table_name, B=500) {
  n <- nrow(df)
  m <- floor(n * 0.9)
  d <- floor(n * 0.1)
  
  pt_ests <- est_func(df)
  k_names <- names(pt_ests)
  
  np_boots <- matrix(NA, nrow=B, ncol=length(k_names))
  m_boots <- matrix(NA, nrow=B, ncol=length(k_names))
  jk_boots <- matrix(NA, nrow=B, ncol=length(k_names))
  
  set.seed(42)
  cat(sprintf("\nRunning Bootstraps for %s (B=%d)...\n", table_name, B))
  
  for (i in 1:B) {
    np_boots[i, ] <- est_func(df[sample(1:n, n, replace = TRUE)])
    m_boots[i, ]  <- est_func(df[sample(1:n, m, replace = TRUE)])
    jk_boots[i, ] <- est_func(df[sample(1:n, (n-d), replace = FALSE)])
  }
  
  # Compile metrics
  results <- data.frame(Estimator = k_names, Point_Estimate = pt_ests, row.names = NULL)
  
  # Column calculations leveraging scaled variances for accurate CIs
  results$`NP-Boot SE` <- apply(np_boots, 2, sd)
  results$`NP 95% CI` <- sprintf("[%gre, %gre]", 
                                 pt_ests - 1.96*results$`NP-Boot SE`, 
                                 pt_ests + 1.96*results$`NP-Boot SE`)
  
  results$`m-Boot SE` <- apply(m_boots, 2, sd) * sqrt(m/n)
  results$`m-Boot CI` <- sprintf("[%gre, %gre]", 
                                 pt_ests - 1.96*results$`m-Boot SE`, 
                                 pt_ests + 1.96*results$`m-Boot SE`)
  
  results$`Jackknife SE` <- apply(jk_boots, 2, sd) * sqrt((n-d)/d)
  results$`Jackknife CI` <- sprintf("[%gre, %gre]", 
                                    pt_ests - 1.96*results$`Jackknife SE`, 
                                    pt_ests + 1.96*results$`Jackknife SE`)
  
  cat(paste0("\n=== ", table_name, " ===\n"))
  print(kable(results, digits = 3, align = 'l'))
}

# Execution
d1 <- fread("data/example1.csv")
d2 <- fread("data/example2.csv")

run_comparison_table(d1, est_case1, "Table 1: Case 1 Results", B=500)
run_comparison_table(d2, est_case2, "Table 2: Case 2 Results", B=500)