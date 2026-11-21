# 05_model_logistic.R
# Owner: Ong Chia Yin (Data Scientist)
# Purpose: Logistic regression on the 3 feature sets
# AI-assisted (Claude), 21 Nov 2026


# ---------------------------------------------------------------------
# section 0 - SETTINGS
# ---------------------------------------------------------------------

PILOT           <- FALSE     # must match 02_features.R
PREFERRED_SPARK <- "3.5.1"   # used if installed, else the newest installed version
DRIVER_MEM      <- "8G"      # memory for Spark (lower it if your laptop struggles)
CORES           <- 4         # CPU cores for Spark

# Model settings: Spark's defaults are used (no tuning).
# The actual values are printed in the log after training.

# Project root = working directory 
PROJECT_ROOT <- normalizePath(getwd(), winslash = "/")
if (basename(PROJECT_ROOT) == "R") PROJECT_ROOT <- dirname(PROJECT_ROOT)

PROC_DIR <- file.path(PROJECT_ROOT, "data", "processed")
IN_DIR   <- if (PILOT) file.path(PROC_DIR, "pilot") else PROC_DIR
PRED_DIR <- if (PILOT) file.path(PROJECT_ROOT, "outputs", "pilot") else file.path(PROJECT_ROOT, "outputs")
RES_DIR  <- file.path(PRED_DIR, "logistic_regression")   # summary and log

# Spark temp folder (same place as the other model scripts)
WORK_ROOT <- Sys.getenv("BDA_WORK_ROOT",
                        unset = if (.Platform$OS.type == "windows") "C:/BDA_work"
                        else file.path(path.expand("~"), "BDA_work"))

CLASSES <- c("anger", "fear", "joy", "love", "sadness", "surprise")   # labels 0..5, set in 02_features.R
SETS    <- c("audio", "lyrics", "combined")   # trained in this order, one after another

# ---------------------------------------------------------------------
# SECTION 1 - PACKAGES, LOGGING, SPARK CONNECTION
# ---------------------------------------------------------------------

pk <- c("sparklyr", "dplyr", "readr")
miss <- pk[!pk %in% rownames(installed.packages())]
if (length(miss)) stop("Please install first: install.packages(c(\"",
                       paste(miss, collapse = "\", \""), "\"))")
suppressPackageStartupMessages({ library(sparklyr); library(dplyr) })

for (d in c(PRED_DIR, RES_DIR)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
LOG <- file.path(RES_DIR, "lr_log.txt")
if (file.exists(LOG)) file.remove(LOG)
say <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = LOG, append = TRUE) }
hdr <- function(t) say("\n==== ", t, " ====")
say_df <- function(d) for (l in capture.output(print(as.data.frame(d), row.names = FALSE))) say(l)
T0 <- Sys.time()

hdr("STEP 1: Spark connection")
say("Run mode: ", if (PILOT) "PILOT (test run)" else "FULL")
say("Started : ", format(T0))
say("Project root: ", PROJECT_ROOT)

# Windows: point Spark to the Hadoop files if they exist
if (.Platform$OS.type == "windows") {
  hh <- Sys.getenv("HADOOP_HOME")
  if (hh == "" && dir.exists("C:/hadoop/bin")) { hh <- "C:/hadoop"; Sys.setenv(HADOOP_HOME = hh) }
  if (hh != "") Sys.setenv(PATH = paste(normalizePath(file.path(hh, "bin"), winslash = "\\"), Sys.getenv("PATH"), sep = ";"))
}

need_files <- paste0(as.vector(outer(SETS, c("_train", "_test"), paste0)), ".parquet")
gone <- need_files[!file.exists(file.path(IN_DIR, need_files))]
if (length(gone)) stop("Missing in ", IN_DIR, ": ", paste(gone, collapse = ", "),
                       ". Run R/02_features.R in the same mode first (or get the files from the Drive).")
say("All input files found in ", IN_DIR)

invisible(tryCatch(spark_disconnect_all(), error = function(e) NULL))   # close old connections
cfg <- spark_config()
cfg$`sparklyr.shell.driver-memory` <- DRIVER_MEM
cfg$sparklyr.cores.local           <- CORES
cfg$spark.sql.shuffle.partitions   <- 32
cfg$spark.local.dir                <- file.path(WORK_ROOT, "spark_tmp")
dir.create(cfg$spark.local.dir, recursive = TRUE, showWarnings = FALSE)

# Use SPARK_HOME if it is set; otherwise pick an installed version like the other scripts
if (Sys.getenv("SPARK_HOME") != "") {
  sc <- spark_connect(master = "local", spark_home = Sys.getenv("SPARK_HOME"), config = cfg)
} else {
  inst <- tryCatch(spark_installed_versions()$spark, error = function(e) character(0))
  spark_ver <- if (PREFERRED_SPARK %in% inst) PREFERRED_SPARK else
    if (length(inst)) tail(sort(inst), 1) else PREFERRED_SPARK
  sc <- spark_connect(master = "local", version = spark_ver, config = cfg)
}
say("Spark ", as.character(spark_version(sc)), " connected. Driver memory ", DRIVER_MEM, ", cores ", CORES)

# ---------------------------------------------------------------------
# SECTION 2 - READ FILES AND CHECK FORMAT
# ---------------------------------------------------------------------

hdr("STEP 2: read the input files")
for (nm in sub("\\.parquet$", "", need_files)) {
  p <- normalizePath(file.path(IN_DIR, paste0(nm, ".parquet")), winslash = "/")
  t <- spark_read_parquet(sc, paste0("in_", nm), p, memory = FALSE)
  cc <- colnames(t)
  if (!all(c("row_id", "emotion", "label", "features") %in% cc))
    stop(nm, " lacks columns: ", paste(setdiff(c("row_id", "emotion", "label", "features"), cc), collapse = ", "))
  if (grepl("_train$", nm) && !"weight" %in% cc) stop(nm, " has no weight column (class weights).")
  say(sprintf("%-15s rows = %s", nm, format(sdf_nrow(t), big.mark = ",")))
}
say("Format check: PASS (row_id, emotion, label, features; weight in train files)")

# ---------------------------------------------------------------------
# SECTION 3 - TRAIN AND PREDICT, ONE FEATURE SET AT A TIME
# The same code runs for every set, so all three get identical settings.
# ---------------------------------------------------------------------

hdr("STEP 3: logistic regression (multinomial, Spark default hyperparameters, class weights from train)")

# Helpers to read a setting or result from the trained model.
# Return NA instead of crashing if a Spark version names it differently.
get_param <- function(model, getter) {
  tryCatch(invoke(spark_jobj(model), getter), error = function(e) NA)
}
get_iterations <- function(model) {
  tryCatch(as.integer(invoke(spark_jobj(model), "summary") %>% invoke("totalIterations")),
           error = function(e) NA_integer_)
}

summary_rows <- list(); preds <- list()
for (nm in SETS) {
  say("\n--- ", nm, " ---")
  # label and weight are stored as decimals; Spark ML needs doubles
  tr <- sdf_sql(sc, sprintf("SELECT CAST(label AS DOUBLE) AS label, CAST(weight AS DOUBLE) AS weight, features
                             FROM in_%s_train", nm))
  te <- sdf_sql(sc, sprintf("SELECT CAST(row_id AS STRING) AS row_id, emotion, CAST(label AS DOUBLE) AS label, features
                             FROM in_%s_test", nm))
  n_train <- sdf_nrow(tr); n_test <- sdf_nrow(te)
  
  # Multinomial logistic regression (6 emotions) with the class weights from 02_features.R. 
  # Everything else is Spark's default.
  t0 <- Sys.time()
  model <- tryCatch(
    ml_logistic_regression(tr, family = "multinomial", weight_col = "weight",
                           features_col = "features", label_col = "label"),
    error = function(e) stop("Logistic regression failed on '", nm, "': ", conditionMessage(e)))
  train_sec <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  
  # The settings Spark actually used (printed once, they are the same for every set)
  max_iter <- as.integer(get_param(model, "getMaxIter"))
  if (nm == SETS[1]) {
    say("Spark default settings used: reg_param = ", get_param(model, "getRegParam"),
        " | elastic_net = ", get_param(model, "getElasticNetParam"),
        " | max_iter = ", max_iter,
        " | tol = ", get_param(model, "getTol"),
        " | fit_intercept = ", get_param(model, "getFitIntercept"),
        " | standardization = ", get_param(model, "getStandardization"))
  }
  
  # Spark reports exactly max_iter when training stopped at the limit,
  # and fewer when it converged earlier.
  iters     <- get_iterations(model)
  converged <- if (is.na(iters) || is.na(max_iter)) NA else iters < max_iter
  n_feat    <- tryCatch(length(model$coefficient_matrix[1, ]), error = function(e) NA_integer_)
  say("Training rows: ", format(n_train, big.mark = ","), " | features: ", format(n_feat, big.mark = ","))
  say("Training time: ", train_sec, " seconds | iterations: ", iters,
      if (isTRUE(converged)) " (converged before the limit)" else if (isFALSE(converged))
        paste0(" (reached the ", max_iter, "-iteration limit, may not be fully converged)") else "")
  
  # Predict the test songs and turn 0-5 back into emotion names
  pr <- ml_predict(model, te)
  invisible(sdf_register(pr, "lr_pred_tmp"))
  p <- sdf_sql(sc, "SELECT row_id, emotion, CAST(prediction AS INT) AS prediction FROM lr_pred_tmp") %>% collect()
  if (nrow(p) != n_test) stop(nm, ": predicted ", nrow(p), " rows but the test file has ", n_test)
  if (anyNA(p$prediction)) stop("Some predictions are NA for ", nm)
  if (anyDuplicated(p$row_id)) stop("Duplicate row_id in the predictions for ", nm)
  
  out <- data.frame(row_id = p$row_id, emotion = p$emotion,
                    prediction = CLASSES[p$prediction + 1], feature_set = nm,
                    stringsAsFactors = FALSE)
  preds[[nm]] <- out
  say("Test rows predicted: ", format(nrow(out), big.mark = ","))
  
  # Count predictions per emotion, as a quick check that the model isn't predicting one emotion for everything.
  say("Predicted class counts: ",
      paste(sprintf("%s=%s", CLASSES, formatC(as.integer(table(factor(out$prediction, levels = CLASSES))),
                                              big.mark = ",", format = "d")), collapse = ", "))
  
  summary_rows[[nm]] <- data.frame(feature_set = nm, train_rows = n_train, test_rows = n_test,
                                   n_features = n_feat, train_seconds = train_sec,
                                   iterations = iters, converged = converged)
}

# ---------------------------------------------------------------------
# SECTION 4 - COMBINE AND SAVE
# ---------------------------------------------------------------------

hdr("STEP 4: save outputs")
train_summary <- do.call(rbind, summary_rows)
readr::write_csv(train_summary, file.path(RES_DIR, "lr_training_summary.csv"))
say("Saved: lr_training_summary.csv")

# Save all three sets in one file
pc <- do.call(rbind, preds[SETS])
n_dup <- sum(duplicated(pc[, c("row_id", "feature_set")]))
if (n_dup > 0) stop("Duplicate (row_id, feature_set) pairs in the combined predictions: ", n_dup)
readr::write_csv(pc, file.path(PRED_DIR, "predictions_logistic_regression.csv"))
say("Saved: ", file.path(PRED_DIR, "predictions_logistic_regression.csv"),
    " (", format(nrow(pc), big.mark = ","), " rows = 3 feature sets x test rows)")

# ---------------------------------------------------------------------
# SECTION 5 - SUMMARY
# ---------------------------------------------------------------------

hdr(paste0("SUMMARY", if (PILOT) " (PILOT: test run only)" else ""))
say_df(train_summary)
say("\nElapsed: ", round(difftime(Sys.time(), T0, units = "mins"), 1), " minutes")

spark_disconnect(sc)
