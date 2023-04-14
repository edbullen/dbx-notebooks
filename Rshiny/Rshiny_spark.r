# Databricks notebook source
# MAGIC %md
# MAGIC ## Simple Shiny Example with Databricks Spark
# MAGIC 
# MAGIC ref: https://docs.databricks.com/sparkr/shiny.html#how-can-i-save-the-shiny-applications-that-i-developed-on-hosted-rstudio-server

# COMMAND ----------

# DBTITLE 1,Imports and Spark Data Connect
library(dplyr)
library(ggplot2)
library(shiny)
library(sparklyr)

sc <- spark_connect(method = "databricks")
diamonds_tbl <- spark_read_csv(sc, path = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv")

# COMMAND ----------

# DBTITLE 1,UI Definition
# Define the UI
ui <- fluidPage(
  sliderInput("carat", "Select Carat Range:",
              min = 0, max = 5, value = c(0, 5), step = 0.01),
  plotOutput('plot')
)

# COMMAND ----------

# DBTITLE 1,Shiny Server Definition
# Define the server code
server <- function(input, output) {
  output$plot <- renderPlot({
    # Select diamonds in carat range
    df <- diamonds_tbl %>%
      dplyr::select("carat", "price") %>%
      dplyr::filter(carat >= !!input$carat[[1]], carat <= !!input$carat[[2]])

    # Scatter plot with smoothed means
    ggplot(df, aes(carat, price)) +
      geom_point(alpha = 1/2) +
      geom_smooth() +
      scale_size_area(max_size = 2) +
      ggtitle("Price vs. Carat")
  })
}

# COMMAND ----------

# DBTITLE 1,Start the Shiny Server
# Return a Shiny app object
shinyApp(ui = ui, server = server)

# COMMAND ----------

head(diamonds_tbl)

# COMMAND ----------


