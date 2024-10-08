
library(readxl)
library(ggplot2)
library(emmeans)
    
#\"R_week1.xlsx\
#gigastats1_adj.xlsx
baseball <- read_xlsx("gigastats1_adj.xlsx")
pitch <- baseball$PITCH
A <- baseball$A
hand_v <- baseball$Hand_v
efficiency <- baseball$Efficiency
K_I <- baseball$K_I
a <- baseball$damp
bend_v2 <- baseball$bend_v2
stride <- baseball$Stride_length
w_dim = baseball$omega_dim

  #ANOVA OF BREAKING BALLS TO FASTBALLS
model1 <- aov(a ~ pitch, data = baseball)
summary(model1)
TukeyHSD(model1)
means <- emmeans(model1, ~factor1*factor2)

# Print the means
print(means, infer = FALSE)


#MODEL FOR ALPHA VS HAND VELOCITY
model_hand <- lm(a~hand_v, data = baseball)
# Create scatter plot with regression line and residuals
plot(model_hand, which = 1,main = "Linear Regression Plot with Residuals")
legend("bottomright", legend = c("Residuals"), col = c("red"), lty = 1)
# Add regression line
abline(model_hand, col = "blue")
# Create scatter plot with regression line and residuals

intercept_subset <- summary(model_hand)$coefficients[1]
slope_subset <- summary(model_hand)$coefficients[2]
# Create equation string
eqn_subset <- paste0("a = ", round(intercept_subset, 2), " + ", round(slope_subset, 2), "V_H")
print(eqn_subset)
summary(model_hand)



  #MODEL FOR EFFICIENCY VS ALPHA
model_eff <- lm(efficiency~a, data = baseball)
# Create scatter plot with regression line and residuals
plot(model_eff, which = 1, main = "Linear Regression Plot with Residuals")
# Add regression line
abline(model_eff, col = "blue")
# Add legend
intercept_subset <- summary(model_eff)$coefficients[1]
slope_subset <- summary(model_eff)$coefficients[2]
# Create equation string
eqn_subset <- paste0("a = ", round(intercept_subset, 2), " - ", round(slope_subset, 2), "EFF")
print(eqn_subset)
legend("bottomleft", legend = c("Residuals", "Regression Line"), col = c("red", "blue"), lty = 1)
summary(model_eff)





  #MODEL FOR ALPHA VS BENDING VELOCITY FINAL OF THORAX (SPINAL FLEXION VELOCITY) 
model_bend <- lm(a~bend_v2, data = baseball)
# Create scatter plot with regression line and residuals
plot(model_bend, which = 1, main = "Linear Regression Plot with Residuals")
abline(model_bend, col = "blue")
# Add regression line
# Add legend
intercept_subset <- summary(model_bend)$coefficients[1]
slope_subset <- summary(model_bend)$coefficients[2]
# Create equation string
eqn_subset <- paste0("a = ", round(intercept_subset, 2), "+", round(slope_subset, 2), "V_H")
print(eqn_subset)
legend("topleft", legend = c("Residuals"), col = c("red"), lty = 1)
summary(model_bend)

model_stride <- lm(a~stride, data = baseball)
# Create scatter plot with regression line and residuals
plot(model_stride, which = 1, main = "Linear Regression Plot with Residuals")
# Add regression line
abline(model_stride, col = "blue")
intercept_subset <- summary(model_bend)$coefficients[1]
slope_subset <- summary(model_bend)$coefficients[2]
# Create equation string
eqn_subset <- paste0("a=",round(intercept_subset, 2), "+", round(slope_subset, 2), "(stride)")
print(eqn_subset)
legend("topright", legend = c("Residuals", "Regression Line"), col = c("red","blue"), lty = 1)
summary(model_stride)

model_w <- lm(a~w_dim, data = baseball)
# Create scatter plot with regression line and residuals
plot(model_w, which = 1, main = "predicted model with Residuals")
# Add regression line
abline(model_stride, col = "blue")
# Add legend
legend("topright", legend = c("Regression Line"), col = c("blue"), lty = 1)
# Add legend
summary(model_w)



'
model2 <- aov(SDF ~ ball, data = baseball)
summary(model2)
TukeyHSD(model2)
model3 <- aov(V_IR ~ ball, data = baseball) 
summary(model3)

model4 <- aov(V_IR ~ VT, data = baseball),
summary(model4)
TukeyHSD(model4)

#lm(formula = K_I ~ hand_v, data = baseball)
 '

