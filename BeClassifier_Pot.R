library(shiny)
library(randomForest)
library(ggplot2)
library(DT)
library(caret)
library(dplyr)
library(circular)
library(e1071)
library(plotly)
library(pROC)

ui <- fluidPage(
  titlePanel("Assessment Tool for Interest Behavior in Pots"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload CSV File",
                accept = c("text/csv",
                           "text/comma-separated-values,text/plain",
                           ".csv")),
      selectInput("target", "Select Species Column (Target):", choices = NULL),
      actionButton("train", "Train Random Forest"),
      downloadButton("downloadModel", "Download Trained RF Model")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Feature Importance", plotOutput("importancePlot")),
        tabPanel("Model Accuracy", 
                 verbatimTextOutput("modelAccuracy"), 
                 plotOutput("confMatrixPlot"),
                 plotOutput("rocPlot")),
        tabPanel("Data Preview", DTOutput("dataTable")),
        tabPanel("Behavior Tracks", plotOutput("behaviorTracks")),
        tabPanel("Behavior Averages",
                 plotOutput("speedBoxPlot"),
                 plotOutput("circularTurningPlot"),
                 plotOutput("freqBoxPlot"),
                 plotOutput("distBoxPlot"),
                 plotOutput("speciesPropPlot"),
                 plotOutput("heatMapPlot"),
                 plotOutput("sizeBoxPlot")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  dataset <- reactive({
    req(input$datafile)
    read.csv(input$datafile$datapath)
  })
  
  observeEvent(dataset(), {
    updateSelectInput(session, "target",
                      choices = names(dataset())[!names(dataset()) %in% c("FishID", "x", "y", "time")])
  })
  
  modelResults <- eventReactive(input$train, {
    req(input$target)
    df <- dataset()
    target <- as.factor(df[[input$target]])
    predictors <- df[, !names(df) %in% c("FishID", "x", "y", "time", input$target)]
    
    set.seed(123)
    train_index <- createDataPartition(target, p = 0.7, list = FALSE)
    train_data <- predictors[train_index, ]
    train_label <- target[train_index]
    test_data <- predictors[-train_index, ]
    test_label <- target[-train_index]
    
    rf <- randomForest(train_data, train_label, importance = TRUE, probability = TRUE)
    preds <- predict(rf, test_data)
    probs <- predict(rf, test_data, type = "prob")
    
    cm <- confusionMatrix(preds, test_label)
    
    list(model = rf, confusion = cm, probs = probs, truth = test_label)
  })
  
  output$importancePlot <- renderPlot({
    req(modelResults())
    importance_df <- data.frame(Feature = rownames(importance(modelResults()$model)),
                                Importance = importance(modelResults()$model)[, 1])
    
    ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_bar(stat = "identity") +
      coord_flip() +
      labs(title = "Feature Importance", x = "Features", y = "MeanDecreaseGini") +
      theme_minimal()
  })
  
  output$modelAccuracy <- renderPrint({
    req(modelResults())
    cat("Random Forest Evaluation Metrics:\n")
    print(modelResults()$confusion)
    cat("\nRandom Forest Hyperparameters:\n")
    print(modelResults()$model$call)
  })
  
  output$confMatrixPlot <- renderPlot({
    req(modelResults())
    cm <- modelResults()$confusion$table
    cm_df <- as.data.frame(cm)
    ggplot(cm_df, aes(Prediction, Reference, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "white", size = 6) +
      scale_fill_gradient(low = "grey", high = "steelblue") +
      labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
      theme_minimal()
  })
  
  output$rocPlot <- renderPlot({
    req(modelResults())
    if (ncol(modelResults()$probs) == 2) {
      roc_obj <- roc(modelResults()$truth, modelResults()$probs[, 2], levels = rev(levels(modelResults()$truth)))
      plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)
      abline(a = 0, b = 1, lty = 2, col = "gray")
    }
  })
  
  output$downloadModel <- downloadHandler(
    filename = function() {
      paste("trained_rf_model", Sys.Date(), ".rds", sep = "")
    },
    content = function(file) {
      saveRDS(modelResults()$model, file)
    }
  )
  
  output$dataTable <- renderDT({
    dataset()
  })
  
  output$behaviorTracks <- renderPlot({
    req(input$target)
    df <- dataset()
    
    ggplot(df, aes(x = x, y = y, group = FishID, color = .data[[input$target]])) +
      geom_path(alpha = 0.6) +
      labs(title = "Fish Tracks Colored by Target", x = "X Position", y = "Y Position") +
      theme_minimal()
  })
  
  output$speedBoxPlot <- renderPlot({
    req(input$target)
    df <- dataset()
    ggplot(df, aes(x = .data[[input$target]], y = speed, fill = .data[[input$target]])) +
      geom_boxplot() +
      labs(title = "Speed Distribution by Target", x = "Target", y = "Speed (units/s)") +
      theme_minimal()
  })
  
  output$freqBoxPlot <- renderPlot({
    req(input$target)
    df <- dataset()
    ggplot(df, aes(x = .data[[input$target]], y = frequency_near_bait, fill = .data[[input$target]])) +
      geom_boxplot() +
      labs(title = "Frequency Near Bait by Target", x = "Target", y = "Frequency (events/min)") +
      theme_minimal()
  })
  
  output$distBoxPlot <- renderPlot({
    req(input$target)
    df <- dataset()
    ggplot(df, aes(x = .data[[input$target]], y = distance_to_bait, fill = .data[[input$target]])) +
      geom_boxplot() +
      labs(title = "Distance to Bait by Target", x = "Target", y = "Distance (pixels or cm)") +
      theme_minimal()
  })
  
  output$speciesPropPlot <- renderPlot({
    req(input$target)
    df <- dataset()
    if (!"species" %in% names(df)) return(NULL)
    df %>% 
      group_by(.data[[input$target]], species) %>% 
      summarise(count = n(), .groups = 'drop') %>% 
      group_by(.data[[input$target]]) %>% 
      mutate(prop = count / sum(count)) %>% 
      ggplot(aes(x = .data[[input$target]], y = prop, fill = species)) +
      geom_bar(stat = "identity", position = "fill") +
      labs(title = "Species Proportion per Target", x = "Target", y = "Proportion") +
      theme_minimal()
  })
  
  output$heatMapPlot <- renderPlot({
    df <- dataset()
    ggplot(df, aes(x = x, y = y)) +
      stat_density_2d(aes(fill = after_stat(density)), geom = "raster", contour = FALSE) +
      scale_fill_viridis_c() +
      labs(title = "Fish Density Heatmap", x = "X Position", y = "Y Position") +
      theme_minimal()
  })
  
  output$sizeBoxPlot <- renderPlot({
    df <- dataset()
    if (!"species" %in% names(df) || !"size" %in% names(df)) return(NULL)
    ggplot(df, aes(x = species, y = size, fill = species)) +
      geom_boxplot() +
      labs(title = "Size Distribution by Species", x = "Species", y = "Size (cm)") +
      theme_minimal()
  })
  
  output$circularTurningPlot <- renderPlot({
    req(input$target)
    df <- dataset()
    df <- df[!is.na(df$turning_angle), ]
    df$angle_rad <- circular::rad(df$turning_angle)
    
    ggplot(df, aes(x = angle_rad, fill = .data[[input$target]])) +
      geom_histogram(binwidth = pi / 12, position = "identity", alpha = 0.6, color = "black") +
      coord_polar(start = 0) +
      scale_x_continuous(limits = c(0, 2 * pi), breaks = seq(0, 2 * pi, by = pi / 2),
                         labels = c("0°", "90°", "180°", "270°", "360°")) +
      labs(title = "Turning Angle Distribution by Target", x = "Turning Angle (degrees)", y = "Frequency") +
      theme_minimal()
  })
  
}

shinyApp(ui = ui, server = server)
