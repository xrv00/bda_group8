# ==============================================================================
# 06_model_random_forest.R
#
# Purpose:  Train Spark MLlib Random Forest on audio, lyric and combined feature
#           sets, predict on the TEST sets only, and export one tidy CSV.
#
# Inputs:   data/processed/audio_{train,test}.parquet
#           data/processed/lyrics_{train,test}.parquet   (feature_set = "lyrics")
#           data/processed/combined_{train,test}.parquet
#
# Outputs:  output/predictions_random_forest.csv         (all three sets)
#           output/predictions_random_forest_partial.csv (smoke test / failures)
#
# Notes:    sparklyr 1.9.6 ml_random_forest() has no weight_col argument, so the
#           underlying RandomForestClassifier estimator is used and its Java
#           weightCol parameter is set explicitly.
#
# AI-assisted (claude), 2026-11-24
# ==============================================================================

suppressPackageStartupMessages({
  library(sparklyr)
  library(dplyr)
  library(tibble)
  library(readr)
})

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Smoke test first with `feature_sets_to_run <- "audio"`; once its checks
# pass, use all three sets for the final export.
feature_sets_to_run <- c("audio", "lyric", "combined")

MODEL_CFG <- list(
  num_trees = 50,
  max_depth = 10,
  seed = 42,
  impurity = "gini",
  feature_subset_strategy = "auto"
)

SPARK_VERSION <- "3.5.1"
DATA_DIR <- file.path("data", "processed")
OUT_DIR <- "output"
FILE_MAP <- c(audio = "audio", lyric = "lyrics", combined = "combined")
OUTPUT_LABEL <- c(audio = "audio", lyric = "lyrics", combined = "combined")
ALL_SETS <- unname(OUTPUT_LABEL)
WEIGHT_COL <- "weight"

set.seed(MODEL_CFG$seed)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

# Inverse-frequency class weights computed from TRAIN only.
# w_c = N / (K * n_c), so the mean weight over all training rows is exactly 1.
# Shared formula so all modelling scripts produce identical weights.
make_class_weights <- function(train_sdf, label_col = "label", weight_col = "class_weight") {
  counts <- train_sdf %>%
    group_by(.data[[label_col]]) %>%
    summarise(n = n(), .groups = "drop") %>%
    collect()
  counts[[weight_col]] <- sum(counts$n) / (nrow(counts) * counts$n)
  counts
}

# Check the weights already stored in the parquet against the shared formula.
check_existing_weights <- function(train_sdf, expected) {
  observed <- train_sdf %>%
    group_by(label) %>%
    summarise(n = n(), existing_weight = mean(as.numeric(.data[[WEIGHT_COL]])), .groups = "drop") %>%
    collect()
  out <- left_join(observed, expected, by = "label")
  out$abs_diff <- abs(out$existing_weight - out$class_weight)
  out
}

# ------------------------------------------------------------------------------
# Spark session (local, relative paths only)
# ------------------------------------------------------------------------------

Sys.setenv(SPARK_LOCAL_IP = "127.0.0.1")
hadoop_home <- file.path(
  spark_install_dir(),
  paste0("spark-", SPARK_VERSION, "-bin-hadoop3"), "tmp", "hadoop"
)
if (dir.exists(hadoop_home)) {
  Sys.setenv(HADOOP_HOME = hadoop_home)
  Sys.setenv(PATH = paste(file.path(hadoop_home, "bin"), Sys.getenv("PATH"), sep = ";"))
  java_opts <- paste0(
    "-Dhadoop.home.dir=", hadoop_home,
    " -Djava.library.path=", file.path(hadoop_home, "bin")
  )
} else {
  java_opts <- ""
}

sc <- spark_connect(
  master = "local",
  version = SPARK_VERSION,
  config = list(
    spark.sql.shuffle.partitions = "8",
    spark.sql.warehouse.dir = file.path(tempdir(), "spark-warehouse"),
    spark.ui.enabled = "false",
    `sparklyr.shell.driver-memory` = "2g",
    `spark.driver.extraJavaOptions` = java_opts,
    `spark.executor.extraJavaOptions` = java_opts
  )
)

cat("SPARK:", SPARK_VERSION, "| master: local\n")
cat("MODEL SETTINGS:", paste(names(MODEL_CFG), unlist(MODEL_CFG), sep = "=", collapse = ", "), "\n")
cat("WEIGHT COLUMN:", WEIGHT_COL, "\n")
cat("SETS TO RUN:", paste(feature_sets_to_run, collapse = ", "), "\n")

# ------------------------------------------------------------------------------
# Label -> emotion mapping, derived from the data (never invented)
# ------------------------------------------------------------------------------

label_map <- spark_read_parquet(
  sc, "label_map_source", file.path(DATA_DIR, paste0(FILE_MAP[["audio"]], "_train.parquet"))
) %>%
  select(label, emotion) %>%
  distinct() %>%
  collect() %>%
  mutate(label = as.numeric(label))

stopifnot(
  nrow(label_map) == 6L,
  !any(duplicated(label_map$label)),
  !any(duplicated(label_map$emotion))
)
cat("LABEL MAP:", paste(paste0(label_map$label, "=", label_map$emotion), collapse = ", "), "\n")

# ------------------------------------------------------------------------------
# Train / predict loop
# ------------------------------------------------------------------------------

predictions <- list()
elapsed <- list()
accuracy <- list()
completed <- character()

for (fs in feature_sets_to_run) {
  t0 <- Sys.time()
  cat("\n========== ", fs, " ==========\n", sep = "")

  result <- tryCatch({
    stem <- FILE_MAP[[fs]]
    train_path <- file.path(DATA_DIR, paste0(stem, "_train.parquet"))
    test_path <- file.path(DATA_DIR, paste0(stem, "_test.parquet"))

    train <- spark_read_parquet(sc, paste0(fs, "_train"), train_path) %>%
      mutate(label = as.numeric(label), weight = as.numeric(weight))
    test <- spark_read_parquet(sc, paste0(fs, "_test"), test_path) %>%
      mutate(label = as.numeric(label))

    n_train <- sdf_nrow(train)
    n_test <- sdf_nrow(test)
    cat("ROWS: train =", n_train, "| test =", n_test, "\n")

    nulls <- train %>%
      summarise(
        label = sum(as.numeric(is.na(label))),
        features = sum(as.numeric(is.na(features))),
        weight = sum(as.numeric(is.na(.data[[WEIGHT_COL]])))
      ) %>%
      collect()
    if (any(nulls > 0)) stop("nulls found in train label/features/weight")
    cat("NULL CHECK: PASS\n")

    dist <- train %>%
      group_by(label) %>%
      tally() %>%
      collect() %>%
      mutate(pct = round(100 * n / sum(n), 2)) %>%
      arrange(label)
    cat("TRAIN CLASS DISTRIBUTION:\n")
    print(as.data.frame(dist))

    # Reuse YQ's weight; verify it matches the shared inverse-frequency formula.
    expected <- make_class_weights(train)
    wcheck <- check_existing_weights(train, expected)
    cat("WEIGHT CHECK (max abs diff vs shared formula):", max(wcheck$abs_diff), "\n")
    if (max(wcheck$abs_diff) > 1e-6) {
      warning("Existing weight column differs from make_class_weights(); using existing as configured")
    }

    # sparklyr's ml_random_forest() wrapper has no weight_col argument.
    estimator <- ml_random_forest_classifier(
      sc,
      features_col = "features",
      label_col = "label",
      num_trees = MODEL_CFG$num_trees,
      max_depth = MODEL_CFG$max_depth,
      seed = MODEL_CFG$seed,
      impurity = MODEL_CFG$impurity,
      feature_subset_strategy = MODEL_CFG$feature_subset_strategy
    )
    invoke(estimator$.jobj, "setWeightCol", WEIGHT_COL)
    model <- ml_fit(estimator, train)

    scored <- ml_predict(model, test) %>%
      select(row_id, actual = emotion, pred_index = prediction) %>%
      collect()

    pred_index <- as.integer(scored$pred_index)
    predicted <- label_map$emotion[match(pred_index, label_map$label)]

    # row_id is a 64-bit Long (up to ~4e11); keep numeric, never as.integer().
    out <- tibble(
      row_id = as.numeric(scored$row_id),
      feature_set = OUTPUT_LABEL[[fs]],
      model = "random_forest",
      actual = scored$actual,
      predicted = predicted
    )

    # Smoke-test gates / export checks.
    if (nrow(out) != n_test) stop("predicted rows != test rows")
    if (any(is.na(out$predicted))) stop("NA in predicted")
    if (!setequal(unique(out$actual), label_map$emotion)) stop("not all emotion classes present in actual")
    if (ncol(out) != 5L || !identical(names(out), c("row_id", "feature_set", "model", "actual", "predicted"))) {
      stop("unexpected output schema")
    }

    acc <- mean(out$actual == out$predicted)
    predictions[[fs]] <- out
    accuracy[[fs]] <- acc
    completed <- c(completed, fs)
    cat("CHECKS: PASS | rows =", nrow(out), "| accuracy =", round(acc, 4), "\n")
    NULL
  }, error = function(e) {
    cat("ERROR in", fs, ":", conditionMessage(e), "\n")
    NULL
  })

  elapsed[[fs]] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("ELAPSED: %.1f seconds\n", elapsed[[fs]]))
  invisible(result)
}

# ------------------------------------------------------------------------------
# Export and summary
# ------------------------------------------------------------------------------

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
all_done <- setequal(completed, ALL_SETS)
out_file <- if (all_done) {
  file.path(OUT_DIR, "predictions_random_forest.csv")
} else {
  file.path(OUT_DIR, "predictions_random_forest_partial.csv")
}

if (length(predictions) > 0) {
  final <- bind_rows(predictions) %>%
    arrange(factor(feature_set, levels = ALL_SETS), row_id)
  write_csv(final, out_file)
  cat("\nSAVED:", out_file, "| rows =", nrow(final), "\n")
} else {
  cat("\nNO PREDICTIONS SAVED (all sets failed)\n")
}

cat("\n========== SUMMARY ==========\n")
cat("ROWS PER FEATURE SET:\n")
print(table(factor(OUTPUT_LABEL[names(predictions)], levels = ALL_SETS)))
cat("ACCURACY PER FEATURE SET:\n")
if (length(accuracy)) print(round(unlist(accuracy), 4)) else cat("(none)\n")
cat("RUNTIME PER FEATURE SET (seconds):\n")
if (length(elapsed)) print(round(unlist(elapsed), 1)) else cat("(none)\n")

spark_disconnect(sc)
cat("DONE\n")
