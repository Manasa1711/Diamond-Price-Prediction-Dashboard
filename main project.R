library(caret)
library(ggplot2)
library(shiny)
library(dplyr)
library(class)     # For KNN
library(rpart)     # For Decision Tree
library(e1071)     # For Naive Bayes
library(shinydashboard)  # For box layout

# Load diamonds dataset
data("diamonds")

# Create a binary target variable based on price (above or below median)
diamonds$PriceAboveMedian <- as.factor(ifelse(diamonds$price > median(diamonds$price), 1, 0))

# Split dataset into training and test sets
set.seed(123)
train_size <- round(0.7 * nrow(diamonds))  # Using round instead of floor
trainData <- diamonds[1:train_size, ]
testData <- diamonds[(train_size+1):nrow(diamonds), ]

# K-Nearest Neighbors (KNN) Model
train_scaled <- trainData[, c("carat", "depth")]
test_scaled <- testData[, c("carat", "depth")]
knn_pred <- knn(train_scaled, test_scaled, trainData$PriceAboveMedian, k = 5)
testData$KNN_Prediction <- knn_pred

# Linear Regression Model
lm_model <- lm(price ~ carat + depth + table + x + y + z, data = trainData)
lm_pred <- predict(lm_model, testData)
# Convert continuous predictions into binary (PriceAboveMedian)
lm_pred_bin <- as.factor(ifelse(lm_pred > median(diamonds$price), 1, 0))
testData$Regression_Prediction <- lm_pred_bin

# Decision Tree Model
dt_model <- rpart(PriceAboveMedian ~ carat + depth + table + x + y + z, data = trainData, method = "class")
dt_pred <- predict(dt_model, testData, type = "class")
testData$DT_Prediction <- dt_pred

# Naive Bayes Model
nb_model <- naiveBayes(PriceAboveMedian ~ carat + depth + table + x + y + z, data = trainData)
nb_pred <- predict(nb_model, testData)
testData$NB_Prediction <- nb_pred

# Accuracy Calculation for each model
knn_accuracy <- sum(testData$KNN_Prediction == testData$PriceAboveMedian) / nrow(testData)
lm_accuracy <- sum(lm_pred_bin == testData$PriceAboveMedian) / nrow(testData)
dt_accuracy <- sum(testData$DT_Prediction == testData$PriceAboveMedian) / nrow(testData)
nb_accuracy <- sum(testData$NB_Prediction == testData$PriceAboveMedian) / nrow(testData)

# Create a data frame to store the accuracies
accuracy_data <- data.frame(
  Model = c("KNN", "Linear Regression", "Decision Tree", "Naive Bayes"),
  Accuracy = c(knn_accuracy, lm_accuracy, dt_accuracy, nb_accuracy)
)

# Visualization: Accuracy Comparison with Decreased Bar Size and Reduced Gap
accuracy_plot <- ggplot(accuracy_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.3, show.legend = FALSE) +  # Decreased width for smaller bars
  labs(title = "Accuracy Comparison of Models", x = "Model", y = "Accuracy") +
  theme_minimal() +
  scale_fill_manual(values = c("KNN" = "lightblue", "Linear Regression" = "lightgreen", 
                               "Decision Tree" = "lightcoral", "Naive Bayes" = "orange"))  # Color for Naive Bayes

# Visualization 1: Carat vs. Depth (KNN)
caratDepthPlot <- ggplot(testData, aes(x = carat, y = depth, color = KNN_Prediction)) +
  geom_point(alpha = 0.7) +
  labs(title = "Carat vs. Depth by KNN Prediction", x = "Carat", y = "Depth") +
  theme_minimal() +
  scale_color_manual(values = c("0" = "red", "1" = "green"))

# Visualization 2: Carat vs. Predicted Price (Linear Regression)
caratPricePlot <- ggplot(testData, aes(x = carat, y = lm_pred)) +
  geom_line(color = "green") +
  labs(title = "Carat vs. Predicted Price (Linear Regression)", x = "Carat", y = "Predicted Price") +
  theme_minimal()

# Visualization 3: Price Distribution by Decision Tree Prediction
pricePlot <- ggplot(testData, aes(x = price, fill = DT_Prediction)) +
  geom_density(alpha = 0.5) +
  labs(title = "Price Distribution by Decision Tree Prediction", x = "Price", y = "Density") +
  theme_minimal() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "orange"))

# Visualization 4: Cut and Price Above Median Rate (Naive Bayes) with different colors
cutPlot <- ggplot(testData, aes(x = cut, fill = NB_Prediction)) +
  geom_bar(position = "fill") +
  labs(title = "Price Above Median Rate by Cut (Naive Bayes)", x = "Cut", y = "Proportion") +
  scale_fill_manual(values = c("0" = "purple", "1" = "orange"), labels = c("Below Median", "Above Median")) +
  theme_minimal()

# Define UI for the Shiny dashboard
ui <- fluidPage(
  titlePanel("Diamond Price Prediction Dashboard"),
  sidebarLayout(
    sidebarPanel(
      h4("Explore Diamond Price Indicators"),
      box(
        title = "Model Accuracy", status = "primary", solidHeader = TRUE, width = 12,
        # Display the accuracy of each model in a simple box format
        p(paste("KNN Accuracy: ", round(knn_accuracy, 3))),
        p(paste("Linear Regression Accuracy: ", round(lm_accuracy, 3))),
        p(paste("Decision Tree Accuracy: ", round(dt_accuracy, 3))),
        p(paste("Naive Bayes Accuracy: ", round(nb_accuracy, 3)))
      )
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Carat vs Depth (KNN)", plotOutput("caratDepthPlot")),
        tabPanel("Carat vs Predicted Price (Linear Regression)", plotOutput("caratPricePlot")),
        tabPanel("Price Distribution (Decision Tree)", plotOutput("pricePlot")),
        tabPanel("Cut and Price Above Median (Naive Bayes)", plotOutput("cutPlot")),
        tabPanel("Accuracy Comparison", plotOutput("accuracyPlot"))  # New tab for accuracy plot
      )
    )
  )
)

# Define server logic for the Shiny dashboard
server <- function(input, output) {
  output$caratDepthPlot <- renderPlot({ caratDepthPlot })
  output$caratPricePlot <- renderPlot({ caratPricePlot })
  output$pricePlot <- renderPlot({ pricePlot })
  output$cutPlot <- renderPlot({ cutPlot })
  output$accuracyPlot <- renderPlot({ accuracy_plot })  # Render the accuracy plot
}

# Run the Shiny application
shinyApp(ui = ui, server = server) 
