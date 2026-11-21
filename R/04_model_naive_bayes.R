# 04_model_naive_bayes.R
# Owner: Tam Yi Qing (Spark Feature Engineer)
# Purpose: Naive Bayes on the 3 feature sets
# AI-assisted (Claude), 20 Nov 2026
# =====================================================================
# Reads : data/processed/{audio,lyrics,combined}_{train,test}.parquet
# Model : multinomial Naive Bayes, smoothing 1.0, class weights
# Writes: predictions_naive_bayes.csv, nb_log.txt
# =====================================================================

# ---------------------------------------------------------------------
# 0. SETTINGS
# ---------------------------------------------------------------------
PILOT         <- FALSE     # must match 02_features.R
PREFERRED_SPARK <- "3.5.1" # used if installed, else the newest installed version
DRIVER_MEM    <- "8G"      # memory for Spark
CORES         <- 4         # CPU cores for Spark
SMOOTHING     <- 1.0       # Naive Bayes smoothing

# Project root (fixed folder, all outputs are saved inside it)
PROJECT_ROOT <- "C:/Users/USER/OneDrive - ums.edu.my/Desktop/BDA/BDA"
if (!dir.exists(PROJECT_ROOT)) stop("PROJECT_ROOT does not exist: ", PROJECT_ROOT)
setwd(PROJECT_ROOT)

PROC_DIR  <- file.path(PROJECT_ROOT, "data", "processed")
IN_DIR    <- if (PILOT) file.path(PROC_DIR, "pilot") else PROC_DIR
PRED_DIR  <- if (PILOT) file.path(PROJECT_ROOT, "outputs", "pilot") else file.path(PROJECT_ROOT, "outputs")
RES_DIR   <- file.path(PRED_DIR, "naive_bayes")       # log

# Spark temp folder
WORK_ROOT <- Sys.getenv("BDA_WORK_ROOT",
                        unset = if (.Platform$OS.type == "windows") "C:/BDA_work"
                        else file.path(path.expand("~"), "BDA_work"))
CLASSES   <- c("anger", "fear", "joy", "love", "sadness", "surprise")  # labels 0..5

# ---------------------------------------------------------------------
# 1. PACKAGES, LOGGING, SPARK CONNECTION
# ---------------------------------------------------------------------
pk <- c("sparklyr", "dplyr", "readr")
miss <- pk[!pk %in% rownames(installed.packages())]
if (length(miss)) install.packages(miss)
suppressPackageStartupMessages({ library(sparklyr); library(dplyr) })

for (d in c(PRED_DIR, RES_DIR)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
LOG <- file.path(RES_DIR, "nb_log.txt")
if (file.exists(LOG)) file.remove(LOG)
say  <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = LOG, append = TRUE) }
hdr  <- function(t) say("\n==== ", t, " ====")
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

need_files <- paste0(c("audio_train", "audio_test", "lyrics_train", "lyrics_test",
                       "combined_train", "combined_test"), ".parquet")
gone <- need_files[!file.exists(file.path(IN_DIR, need_files))]
if (length(gone)) stop("Missing in ", IN_DIR, ": ", paste(gone, collapse = ", "),
                       ". Run R/02_features.R in the same mode first.")
say("All six input files found in ", IN_DIR)

inst <- tryCatch(spark_installed_versions()$spark, error = function(e) character(0))
SPARK_VERSION <- if (PREFERRED_SPARK %in% inst) PREFERRED_SPARK else
  if (length(inst)) tail(sort(inst), 1) else PREFERRED_SPARK

tryCatch(spark_disconnect_all(), error = function(e) NULL)   # close old connections
cfg <- spark_config()
cfg$`sparklyr.shell.driver-memory` <- DRIVER_MEM
cfg$sparklyr.cores.local           <- CORES
cfg$spark.sql.shuffle.partitions   <- 32
cfg$spark.local.dir                <- file.path(WORK_ROOT, "spark_tmp")
dir.create(cfg$spark.local.dir, recursive = TRUE, showWarnings = FALSE)
sc <- spark_connect(master = "local", version = SPARK_VERSION, config = cfg)
say("Spark ", SPARK_VERSION, " connected. Driver memory ", DRIVER_MEM, ", cores ", CORES)

# ---------------------------------------------------------------------
# 2. READ FILES AND CHECK FORMAT
# ---------------------------------------------------------------------
hdr("STEP 2: read the six files")
for (nm in sub("\\.parquet$", "", need_files)) {
  p <- normalizePath(file.path(IN_DIR, paste0(nm, ".parquet")), winslash = "/")
  t <- spark_read_parquet(sc, paste0("in_", nm), p, memory = FALSE)
  say(sprintf("%-15s rows = %s | columns: %s", nm, format(sdf_nrow(t), big.mark = ","),
              paste(colnames(t), collapse = ", ")))
}
need_cols <- c("row_id", "emotion", "label", "features")
for (nm in c("audio_train", "audio_test", "lyrics_train", "lyrics_test", "combined_train", "combined_test")) {
  cc <- colnames(tbl(sc, paste0("in_", nm)))
  if (!all(need_cols %in% cc)) stop(nm, " lacks columns: ", paste(setdiff(need_cols, cc), collapse = ", "))
  if (grepl("_train$", nm) && !"weight" %in% cc) stop(nm, " has no weight column.")
}
say("Format check: PASS (row_id, emotion, label, features; weight in train files)")

# ---------------------------------------------------------------------
# 3. TRAIN AND PREDICT
# ---------------------------------------------------------------------
hdr(paste0("STEP 3: Naive Bayes (multinomial, smoothing = ", SMOOTHING, ", class weights from train)"))
sets <- list(
  audio    = list(tr = "in_audio_train",    te = "in_audio_test"),
  lyrics   = list(tr = "in_lyrics_train",   te = "in_lyrics_test"),
  combined = list(tr = "in_combined_train", te = "in_combined_test"))

preds <- list()
for (nm in names(sets)) {
  s <- sets[[nm]]
  say("\n--- ", nm, " ---")
  tr <- sdf_sql(sc, sprintf("SELECT label, weight, features FROM %s", s$tr))
  te <- sdf_sql(sc, sprintf("SELECT CAST(row_id AS STRING) AS row_id, emotion, label, features FROM %s", s$te))
  
  t0 <- Sys.time()
  model <- tryCatch(
    ml_naive_bayes(tr, model_type = "multinomial", smoothing = SMOOTHING, weight_col = "weight",
                   features_col = "features", label_col = "label"),
    error = function(e) stop("Naive Bayes failed on '", nm, "': ", conditionMessage(e)))
  train_sec <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  say("Training time: ", train_sec, " seconds")
  
  pr <- ml_predict(model, te)
  invisible(sdf_register(pr, "nb_pred_tmp"))
  p <- sdf_sql(sc, "SELECT row_id, emotion, CAST(label AS INT) AS label, CAST(prediction AS INT) AS prediction FROM nb_pred_tmp") %>% collect()
  say("Test rows predicted: ", format(nrow(p), big.mark = ","))
  if (anyNA(p$prediction)) stop("Some predictions are NA for ", nm)
  
  preds[[nm]] <- data.frame(row_id = p$row_id, emotion = p$emotion,
                            prediction = CLASSES[p$prediction + 1], feature_set = nm,
                            stringsAsFactors = FALSE)
}

# Row check: train and test rows used by each feature set
hdr("ROW CHECK")
row_check <- do.call(rbind, lapply(names(sets), function(nm) {
  data.frame(
    feature_set = nm,
    train_rows  = sdf_nrow(tbl(sc, sets[[nm]]$tr)),
    test_rows   = sdf_nrow(tbl(sc, sets[[nm]]$te)),
    predicted   = nrow(preds[[nm]])
  )
}))
say_df(row_check)
say("Total predictions: ", format(sum(row_check$predicted), big.mark = ","))
say("All feature sets use same rows: ",
    if (length(unique(row_check$train_rows)) == 1 &&
        length(unique(row_check$test_rows)) == 1 &&
        all(row_check$test_rows == row_check$predicted)) "PASS" else "FAIL")

# ---------------------------------------------------------------------
# 4. SAVE OUTPUTS
# ---------------------------------------------------------------------
hdr("STEP 4: save outputs")
main_sets <- c("audio", "lyrics", "combined")
pc <- do.call(rbind, preds[main_sets])
readr::write_csv(pc, file.path(PRED_DIR, "predictions_naive_bayes.csv"))
say("Saved: ", file.path(PRED_DIR, "predictions_naive_bayes.csv"))
say("Saved: nb_log.txt in ", RES_DIR)
say("predictions_naive_bayes.csv rows: ", format(nrow(pc), big.mark = ","),
    " (", length(main_sets), " feature sets x test rows); duplicate (row_id, feature_set) pairs: ",
    sum(duplicated(pc[, c("row_id", "feature_set")])))

# ---------------------------------------------------------------------
# 5. SUMMARY
# ---------------------------------------------------------------------
hdr(paste0("SUMMARY", if (PILOT) " (PILOT: test run only)" else ""))
say("Elapsed: ", round(difftime(Sys.time(), T0, units = "mins"), 1), " minutes")
say("Note: row_id is saved as text.")

spark_disconnect(sc)
