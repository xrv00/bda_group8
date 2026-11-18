# 04_model_naive_bayes.R
# Owner: Tam Yi Qing (Spark Feature Engineer)
# Purpose: Naive Bayes on the 3 feature sets

# ---------------------------------------------------------------------
# 0. SWITCHES AND SETTINGS (edit only this block)
# ---------------------------------------------------------------------
PILOT         <- FALSE      # must match the mode you ran 05_features_split.R in
SPARK_VERSION <- "3.5.1"
DRIVER_MEM    <- "8G"      # if Spark stalls, use "6G"
CORES         <- 4         # if Spark stalls, use 2
SMOOTHING     <- 1.0       # fixed in advance; do not tune on the test set
RUN_ABLATION  <- FALSE     # 11-feature audio set dropped; the files only hold the 9-feature vector

WORK_ROOT <- "C:/BDA_work"
IN_DIR    <- if (PILOT) file.path(WORK_ROOT, "pilot") else WORK_ROOT
OUT_DIR   <- IN_DIR
CLASSES   <- c("anger", "fear", "joy", "love", "sadness", "surprise")  # label 0..5

# ---------------------------------------------------------------------
# 1. PACKAGES, LOGGING, ENVIRONMENT CHECK
# ---------------------------------------------------------------------
pk <- c("sparklyr", "dplyr", "readr")
miss <- pk[!pk %in% rownames(installed.packages())]
if (length(miss)) install.packages(miss)
suppressPackageStartupMessages({ library(sparklyr); library(dplyr) })

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
LOG <- file.path(OUT_DIR, "nb_log.txt")
if (file.exists(LOG)) file.remove(LOG)
say  <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = LOG, append = TRUE) }
hdr  <- function(t) say("\n==== ", t, " ====")
say_df <- function(d) for (l in capture.output(print(as.data.frame(d), row.names = FALSE))) say(l)
T0 <- Sys.time()

hdr("STEP 1: environment")
say("Run mode: ", if (PILOT) "PILOT (scores are not reportable)" else "FULL")
say("Started : ", format(T0))

jv <- tryCatch(system2("java", "-version", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
if (is.null(jv)) stop("Java was not found on PATH. Install Temurin 17 and restart R.")
m <- regmatches(jv[1], regexec('version "([0-9]+)\\.?([0-9]*)', jv[1]))[[1]]
java_major <- as.integer(m[2]); if (java_major == 1L) java_major <- as.integer(m[3])
say("Java major version: ", java_major)
if (!java_major %in% c(8L, 11L, 17L))
  stop("Spark 3.5 needs Java 8, 11 or 17. Found Java ", java_major, ".")

if (Sys.getenv("JAVA_HOME") == "") {
  jp <- Sys.which("java")
  if (nzchar(jp)) {
    cand <- dirname(dirname(normalizePath(jp, winslash = "/")))
    if (file.exists(file.path(cand, "release"))) { Sys.setenv(JAVA_HOME = cand); say("JAVA_HOME set to ", cand) }
  }
}

inst <- spark_installed_versions()$spark
if (!SPARK_VERSION %in% inst)
  stop("Spark ", SPARK_VERSION, " is not installed. Installed: ", paste(inst, collapse = ", "))
say("Spark ", SPARK_VERSION, " found.")

if (.Platform$OS.type == "windows") {
  hh <- Sys.getenv("HADOOP_HOME")
  if (hh == "" && file.exists("C:/hadoop/bin/winutils.exe")) { hh <- "C:/hadoop"; Sys.setenv(HADOOP_HOME = hh) }
  has_files <- hh != "" && file.exists(file.path(hh, "bin", "winutils.exe")) &&
    file.exists(file.path(hh, "bin", "hadoop.dll"))
  if (!has_files)
    stop("winutils.exe and/or hadoop.dll not found (HADOOP_HOME = '", hh, "'). Put both in C:/hadoop/bin, restart R, rerun.")
  Sys.setenv(PATH = paste(normalizePath(file.path(hh, "bin"), winslash = "\\"), Sys.getenv("PATH"), sep = ";"))
  say("HADOOP_HOME = ", hh)
}

need_files <- paste0(c("audio_train", "audio_test", "lyrics_train", "lyrics_test",
                       "combined_train", "combined_test"), ".parquet")
gone <- need_files[!file.exists(file.path(IN_DIR, need_files))]
if (length(gone)) stop("Missing in ", IN_DIR, ": ", paste(gone, collapse = ", "),
                       ". Run 05_features_split.R in the same mode first.")
say("All six input files found in ", IN_DIR)
say("Check: these files must come from the version 3 run of 05 (look for 'Script version 3' in stage2_log.txt).")

cfg <- spark_config()
cfg$`sparklyr.shell.driver-memory` <- DRIVER_MEM
cfg$sparklyr.cores.local           <- CORES
cfg$spark.sql.shuffle.partitions   <- 32
cfg$spark.local.dir                <- file.path(WORK_ROOT, "spark_tmp")
dir.create(cfg$spark.local.dir, recursive = TRUE, showWarnings = FALSE)
sc <- spark_connect(master = "local", version = SPARK_VERSION, config = cfg)
say("Spark connected. Driver memory ", DRIVER_MEM, ", cores ", CORES)

# ---------------------------------------------------------------------
# 2. READ THE FILES AND CHECK THEIR FORMAT
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
if (RUN_ABLATION && !"features_11" %in% colnames(tbl(sc, "in_audio_train"))) {
  say("features_11 not found in audio files, ablation skipped."); RUN_ABLATION <- FALSE
}
say("Format check: PASS (row_id, emotion, label, features; weight in train files)")

maj <- sdf_sql(sc, "SELECT label, COUNT(*) AS n FROM in_audio_train GROUP BY label ORDER BY n DESC LIMIT 1") %>% collect()
MAJ_LABEL <- as.integer(maj$label[1])
say("Majority class in train: ", CLASSES[MAJ_LABEL + 1], " (label ", MAJ_LABEL, ")")

# ---------------------------------------------------------------------
# 3. METRIC HELPERS (Macro F1 gives every class equal weight)
# ---------------------------------------------------------------------
class_metrics <- function(actual, pred) {
  cm <- table(factor(actual, levels = 0:5), factor(pred, levels = 0:5))
  tp <- diag(cm); fn <- rowSums(cm) - tp; fp <- colSums(cm) - tp
  prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  rec  <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1   <- ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
  list(cm = cm, precision = unname(prec), recall = unname(rec), f1 = unname(f1),
       macro_f1 = mean(f1), accuracy = sum(tp) / sum(cm))
}
metric_row <- function(set_name, mt, train_sec) {
  out <- data.frame(feature_set = set_name, macro_f1 = mt$macro_f1, accuracy = mt$accuracy,
                    train_seconds = train_sec, stringsAsFactors = FALSE)
  for (i in seq_along(CLASSES)) out[[paste0("recall_", CLASSES[i])]]    <- mt$recall[i]
  for (i in seq_along(CLASSES)) out[[paste0("precision_", CLASSES[i])]] <- mt$precision[i]
  for (i in seq_along(CLASSES)) out[[paste0("f1_", CLASSES[i])]]        <- mt$f1[i]
  out
}

# ---------------------------------------------------------------------
# 4. TRAIN AND PREDICT (identical settings for every feature set)
# ---------------------------------------------------------------------
hdr(paste0("STEP 3: Naive Bayes (multinomial, smoothing = ", SMOOTHING, ", class weights from train)"))
sets <- list(
  audio    = list(tr = "in_audio_train",    te = "in_audio_test",    col = "features"),
  lyrics   = list(tr = "in_lyrics_train",   te = "in_lyrics_test",   col = "features"),
  combined = list(tr = "in_combined_train", te = "in_combined_test", col = "features"))
if (RUN_ABLATION) sets$audio_11 <- list(tr = "in_audio_train", te = "in_audio_test", col = "features_11")

res <- list(); preds <- list(); cms <- list()
for (nm in names(sets)) {
  s <- sets[[nm]]
  say("\n--- ", nm, " ---")
  tr <- sdf_sql(sc, sprintf("SELECT label, weight, %s AS features FROM %s", s$col, s$tr))
  te <- sdf_sql(sc, sprintf("SELECT CAST(row_id AS STRING) AS row_id, emotion, label, %s AS features FROM %s", s$col, s$te))
  
  t0 <- Sys.time()
  model <- tryCatch(
    ml_naive_bayes(tr, model_type = "multinomial", smoothing = SMOOTHING, weight_col = "weight",
                   features_col = "features", label_col = "label"),
    error = function(e) stop("Naive Bayes failed on '", nm, "'. If the message mentions negative values, ",
                             "a feature is below 0 (Naive Bayes needs non-negative input). Original error: ",
                             conditionMessage(e)))
  train_sec <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  say("Training time: ", train_sec, " seconds")
  
  pr <- ml_predict(model, te)
  invisible(sdf_register(pr, "nb_pred_tmp"))
  p <- sdf_sql(sc, "SELECT row_id, emotion, CAST(label AS INT) AS label, CAST(prediction AS INT) AS prediction FROM nb_pred_tmp") %>% collect()
  say("Test rows predicted: ", format(nrow(p), big.mark = ","))
  if (anyNA(p$prediction)) stop("Some predictions are NA for ", nm)
  
  mt <- class_metrics(p$label, p$prediction)
  say("Macro F1 = ", round(mt$macro_f1, 4), " | accuracy = ", round(mt$accuracy, 4))
  say("Predicted class counts: ",
      paste(sprintf("%s=%d", CLASSES, as.integer(colSums(mt$cm))), collapse = ", "))
  say("Actual class counts   : ",
      paste(sprintf("%s=%d", CLASSES, as.integer(rowSums(mt$cm))), collapse = ", "))
  
  res[[nm]] <- metric_row(nm, mt, train_sec)
  cm_long <- as.data.frame(mt$cm, stringsAsFactors = FALSE)
  names(cm_long) <- c("actual", "predicted", "n")
  cm_long$actual    <- CLASSES[as.integer(cm_long$actual) + 1]
  cm_long$predicted <- CLASSES[as.integer(cm_long$predicted) + 1]
  cms[[nm]] <- cbind(feature_set = nm, cm_long)
  preds[[nm]] <- data.frame(row_id = p$row_id, emotion = p$emotion,
                            prediction = CLASSES[p$prediction + 1], feature_set = nm,
                            stringsAsFactors = FALSE)
  if (nm == "audio") base_actual <- p$label     # same test rows in every set
}

# Majority-class baseline (predict the train majority class for every test row)
mb <- class_metrics(base_actual, rep(MAJ_LABEL, length(base_actual)))
res[["baseline_majority"]] <- metric_row("baseline_majority", mb, NA_real_)
say("\nMajority-class baseline: Macro F1 = ", round(mb$macro_f1, 4), " | accuracy = ", round(mb$accuracy, 4))

# ---------------------------------------------------------------------
# 5. SAVE THE OUTPUTS
# ---------------------------------------------------------------------
hdr("STEP 4: save outputs")
metrics <- do.call(rbind, res)
readr::write_csv(metrics, file.path(OUT_DIR, "nb_metrics.csv"))
readr::write_csv(do.call(rbind, cms), file.path(OUT_DIR, "nb_confusion.csv"))

main_sets <- intersect(c("audio", "lyrics", "combined"), names(preds))
readr::write_csv(do.call(rbind, preds[main_sets]), file.path(OUT_DIR, "predictions_naive_bayes.csv"))
if (RUN_ABLATION) readr::write_csv(preds[["audio_11"]], file.path(OUT_DIR, "predictions_naive_bayes_audio11.csv"))
say("Saved: predictions_naive_bayes.csv, ",
    if (RUN_ABLATION) "predictions_naive_bayes_audio11.csv, " else "",
    "nb_metrics.csv, nb_confusion.csv in ", OUT_DIR)

pc <- do.call(rbind, preds[main_sets])
say("predictions_naive_bayes.csv rows: ", format(nrow(pc), big.mark = ","),
    " (", length(main_sets), " feature sets x test rows); duplicate (row_id, feature_set) pairs: ",
    sum(duplicated(pc[, c("row_id", "feature_set")])))

# ---------------------------------------------------------------------
# 6. SUMMARY
# ---------------------------------------------------------------------
hdr(paste0("SUMMARY", if (PILOT) " (PILOT: do not report these scores)" else ""))
say_df(metrics %>% select(feature_set, macro_f1, accuracy, train_seconds) %>%
         mutate(across(c(macro_f1, accuracy), ~ round(.x, 4))))
say("\nPer-class recall (which emotions are missed):")
rc <- metrics %>% select(feature_set, starts_with("recall_"))
names(rc) <- sub("^recall_", "", names(rc))
say_df(rc %>% mutate(across(-feature_set, ~ round(.x, 3))))
say("\nElapsed: ", round(difftime(Sys.time(), T0, units = "mins"), 1), " minutes")
say("Hand-off: send predictions_naive_bayes.csv to Xin Rou (Integration Lead).")
say("Tell her to read row_id as TEXT (readr::cols(row_id = col_character())), because ids are large numbers.")
spark_disconnect(sc)
