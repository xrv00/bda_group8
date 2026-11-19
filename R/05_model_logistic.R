# 05_model_logistic.R
# Owner: Ong Chia Yin (Data Scientist)
# Purpose: Logistic regression on the 3 feature sets

# Input   : data/processed/{audio,lyrics,combined}_{train,test}.parquet
#           (from 02_features: features already scaled / TF-IDF,
#            label 0-5 and a class-weight column)
# Output  : outputs/predictions_logistic_regression.csv
             
# AI-assisted (Claude), 19 Nov 2026


# -----------------------------------------------------------------------------
# SECTION 0 - SETTINGS
# -----------------------------------------------------------------------------

# Folder that contains the 6 parquet folders from 02_features.R
parquet_dir <- "data/processed"

# Folder where the predictions are saved
out_dir <- "outputs"

# Windows only: folder that contains bin/winutils.exe and bin/hadoop.dll
# (ignored on Mac / Linux)
hadoop_home <- "C:/hadoop"

# Spark settings (same as the Data Engineer's working laptop settings)
spark_version_used <- "3.5.1"
spark_memory       <- "4G"
spark_master       <- "local[2]"   # 2 CPU cores

# The three feature sets
feature_sets <- c("audio", "lyrics", "combined")

# Label numbers used by the Feature Engineer (0-5, alphabetical order)
emotion_names <- c("anger", "fear", "joy", "love", "sadness", "surprise")


# -----------------------------------------------------------------------------
# SECTION 1 - LOAD THE PACKAGES
# -----------------------------------------------------------------------------

required_pkgs <- c("sparklyr", "dplyr")
missing_pkgs  <- required_pkgs[!vapply(required_pkgs, requireNamespace,
                                       logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing packages: ", paste(missing_pkgs, collapse = ", "),
       "\nInstall them with: install.packages(c(\"",
       paste(missing_pkgs, collapse = "\", \""), "\"))")
}

suppressPackageStartupMessages({
  library(sparklyr)
  library(dplyr)
})


# -----------------------------------------------------------------------------
# SECTION 2 - OUTPUT FOLDER, LOG AND HELPER FUNCTION
# -----------------------------------------------------------------------------

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(out_dir, "lr_log.txt")
cat("Logistic regression log -", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n",
    file = log_file)

# Print a message to the console AND append it to the log file
log_msg <- function(...) {
  msg <- paste0(...)
  message(msg)
  cat(msg, "\n", file = log_file, append = TRUE)
}

# Load one parquet folder into Spark.
# The label and weight columns are saved as decimals, which Spark's models
# can't use, so changed them into normal numbers first.
load_set <- function(name) {
  path <- file.path(parquet_dir, paste0(name, ".parquet"))
  if (!dir.exists(path)) stop("Parquet folder not found: ", path)
  df <- spark_read_parquet(sc, name = name, path = path, memory = FALSE) |>
    mutate(label = as.numeric(label))
  if ("weight" %in% colnames(df)) df <- df |> mutate(weight = as.numeric(weight))
  df
}


# -----------------------------------------------------------------------------
# SECTION 3 - START SPARK
# -----------------------------------------------------------------------------

# Windows only: tell Spark where winutils.exe and hadoop.dll are
if (.Platform$OS.type == "windows") {
  Sys.setenv(HADOOP_HOME = hadoop_home)
  Sys.setenv(PATH = paste(file.path(hadoop_home, "bin"), Sys.getenv("PATH"), sep = ";"))
}

conf <- spark_config()
conf$`sparklyr.shell.driver-memory` <- spark_memory
conf$spark.sql.shuffle.partitions   <- 8

sc <- spark_connect(master = spark_master, version = spark_version_used,
                    config = conf)
log_msg("Spark version: ", spark_version(sc))


# -----------------------------------------------------------------------------
# SECTION 4 - CHECK THE LABELS
# Make sure label numbers match the emotion names (0 = anger ... 5 = surprise),
# otherwise the predicted emotion names would be wrong.
# -----------------------------------------------------------------------------

label_map <- load_set("audio_train") |> distinct(label, emotion) |>
  arrange(label) |> collect()
if (!identical(label_map$emotion, emotion_names)) {
  stop("Label numbers do not match emotion_names. Check with the Feature Engineer.")
}
log_msg("Label mapping OK: ", paste(label_map$label, label_map$emotion,
                                    sep = "=", collapse = ", "))


# -----------------------------------------------------------------------------
# SECTION 5 - TRAIN ONE MODEL PER FEATURE SET AND PREDICT THE TEST SET
# Settings:
#  - family = "multinomial": one model for all 6 emotions at once
#  - weight_col = "weight": rare emotions (example: surprise) count more in training
#  - all other settings are Spark's defaults (no regularisation, 100 iterations)
# -----------------------------------------------------------------------------

all_predictions <- list()

for (fs in feature_sets) {
  train_df <- load_set(paste0(fs, "_train"))
  test_df  <- load_set(paste0(fs, "_test"))
  log_msg(fs, ": ", sdf_nrow(train_df), " training songs, ",
          sdf_nrow(test_df), " test songs")

  # Step 1: train
  start <- Sys.time()
  model <- ml_logistic_regression(train_df,
                                  features_col = "features",
                                  label_col    = "label",
                                  weight_col   = "weight",
                                  family       = "multinomial")
  secs <- round(as.numeric(difftime(Sys.time(), start, units = "secs")))
  log_msg(fs, ": model trained in ", secs, " seconds")

  # Step 2: predict the test set and keep the 4 columns the Integration Lead
  # needs. row_id is turned into text inside Spark so large IDs are written exactly
  all_predictions[[fs]] <- ml_transform(model, test_df) |>
    mutate(row_id = as.character(row_id)) |>
    select(row_id, emotion, prediction) |>
    collect() |>
    mutate(prediction  = emotion_names[prediction + 1],   # 0-5 -> emotion name
           feature_set = fs)
}


# -----------------------------------------------------------------------------
# SECTION 6 - SAVE THE PREDICTIONS
# -----------------------------------------------------------------------------

predictions <- bind_rows(all_predictions)
# quote = FALSE: same plain format as predictions_naive_bayes.csv
write.csv(predictions, file.path(out_dir, "predictions_logistic_regression.csv"),
          row.names = FALSE, quote = FALSE)
log_msg("Saved predictions: ", nrow(predictions), " rows (",
        paste(names(all_predictions), sapply(all_predictions, nrow),
              sep = " = ", collapse = ", "), ")")


# -----------------------------------------------------------------------------
# SECTION 7 - FINISH
# -----------------------------------------------------------------------------

spark_disconnect(sc)
cat("\n--- sessionInfo() ---\n", file = log_file, append = TRUE)
capture.output(sessionInfo(), file = log_file, append = TRUE)
log_msg("Logistic regression finished. Outputs are in: ", normalizePath(out_dir))
