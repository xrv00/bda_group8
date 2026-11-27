# ==============================================================================
# eval_utils.R
#
# Purpose:  Shared, side-effect-free evaluation functions for the emotion
#           prediction project.
#
# Inputs:   None on source. Functions take vectors / a parquet path as arguments.
#
# Outputs:  None on source. Functions return metrics and data frames.
#
# AI-assisted (claude), 2026-11-24
# ==============================================================================

# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------

#' Confusion matrix counts with a fixed class order
#'
#' @param actual Character or factor vector of true class labels.
#' @param predicted Character or factor vector of predicted class labels.
#' @param classes Character vector defining the fixed row and column order.
#' @return Data frame with columns `actual`, `predicted`, `n`, containing every
#'   combination of `classes`, including classes that are never predicted.
confusion_matrix <- function(actual, predicted, classes) {
  if (length(actual) != length(predicted)) stop("`actual` and `predicted` must have the same length")
  if (!length(classes) || anyDuplicated(classes)) stop("`classes` must be non-empty and unique")

  actual <- as.character(actual)
  predicted <- as.character(predicted)
  unknown <- setdiff(unique(c(actual, predicted)), classes)
  if (length(unknown)) stop("labels outside `classes`: ", paste(unknown, collapse = ", "))

  counts <- table(factor(actual, levels = classes), factor(predicted, levels = classes))
  grid <- expand.grid(actual = classes, predicted = classes, stringsAsFactors = FALSE)
  idx <- cbind(match(grid$actual, classes), match(grid$predicted, classes))
  grid$n <- as.integer(counts[idx])
  grid$actual <- factor(grid$actual, levels = classes)
  grid$predicted <- factor(grid$predicted, levels = classes)
  grid
}

#' Per-class precision, recall and F1
#'
#' @param actual Character or factor vector of true class labels.
#' @param predicted Character or factor vector of predicted class labels.
#' @param classes Character vector defining the fixed class order.
#' @return Data frame with one row per class: `class`, `precision`, `recall`,
#'   `f1`, `support`. A class with no predictions or no support scores 0 rather
#'   than NaN for precision / F1.
per_class_metrics <- function(actual, predicted, classes) {
  cm <- confusion_matrix(actual, predicted, classes)
  tp <- vapply(classes, function(cls) {
    cm$n[cm$actual == cls & cm$predicted == cls]
  }, numeric(1))
  fp <- vapply(classes, function(cls) {
    sum(cm$n[cm$actual != cls & cm$predicted == cls])
  }, numeric(1))
  fn <- vapply(classes, function(cls) {
    sum(cm$n[cm$actual == cls & cm$predicted != cls])
  }, numeric(1))

  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)

  data.frame(
    class = factor(classes, levels = classes),
    precision = precision,
    recall = recall,
    f1 = f1,
    support = tp + fn,
    stringsAsFactors = FALSE
  )
}

#' Macro F1 over a fixed set of classes
#'
#' @param actual Character or factor vector of true class labels.
#' @param predicted Character or factor vector of predicted class labels.
#' @param classes Optional character vector of classes. Defaults to all labels
#'   present in `actual` or `predicted`, sorted.
#' @return Unweighted mean of the per-class F1 scores.
macro_f1 <- function(actual, predicted, classes = NULL) {
  if (is.null(classes)) classes <- sort(unique(c(as.character(actual), as.character(predicted))))
  mean(per_class_metrics(actual, predicted, classes)$f1)
}

#' Accuracy
#'
#' @param actual Character or factor vector of true class labels.
#' @param predicted Character or factor vector of predicted class labels.
#' @return Fraction of rows where actual equals predicted, or NA for empty input.
accuracy <- function(actual, predicted) {
  if (length(actual) != length(predicted)) stop("`actual` and `predicted` must have the same length")
  if (!length(actual)) return(NA_real_)
  if (anyNA(actual) || anyNA(predicted)) stop("NA values in `actual` or `predicted`")
  mean(as.character(actual) == as.character(predicted))
}

# ------------------------------------------------------------------------------
# Baseline
# ------------------------------------------------------------------------------

#' Majority-class baseline
#'
#' @param train_labels Character or factor vector of TRAIN class labels. Must be
#'   emotion names, not numeric indexes.
#' @param test_actual Character or factor vector of true TEST labels.
#' @return List with `class` (the most frequent TRAIN class), `accuracy` and
#'   `macro_f1` on the test rows. Ties break alphabetically.
majority_baseline <- function(train_labels, test_actual) {
  if (is.numeric(train_labels)) {
    stop("TRAIN labels are numeric; read the `emotion` column from the parquet file instead of guessing a mapping")
  }
  labs <- as.character(train_labels)
  labs <- labs[!is.na(labs)]
  if (!length(labs)) stop("`train_labels` contains no usable labels")

  tab <- sort(table(labs), decreasing = TRUE)
  cls <- names(tab)[1L]
  predicted <- rep(cls, length(test_actual))
  classes <- sort(unique(c(as.character(test_actual), cls)))

  list(
    class = cls,
    accuracy = accuracy(test_actual, predicted),
    macro_f1 = macro_f1(test_actual, predicted, classes)
  )
}

#' Read TRAIN emotion labels from a Spark parquet dataset
#'
#' @param path Relative path to a train parquet dataset under `data/processed`.
#' @param spark_version Spark version string used by sparklyr.
#' @return Character vector of emotion names from the `emotion` column only.
read_train_labels <- function(path = file.path("data", "processed", "audio_train.parquet"),
                              spark_version = "3.5.1") {
  if (!requireNamespace("sparklyr", quietly = TRUE)) stop("package `sparklyr` is required")

  Sys.setenv(SPARK_LOCAL_IP = "127.0.0.1")
  java_opts <- ""
  hadoop_home <- ""
  if (.Platform$OS.type == "windows") {
    candidates <- unique(c(
      Sys.getenv("HADOOP_HOME", unset = ""),
      file.path(sparklyr::spark_install_dir(),
                paste0("spark-", spark_version, "-bin-hadoop3"), "tmp", "hadoop"),
      "C:/hadoop"
    ))
    candidates <- candidates[nzchar(candidates)]
    for (cand in candidates) {
      if (file.exists(file.path(cand, "bin", "winutils.exe"))) {
        hadoop_home <- normalizePath(cand, winslash = "/", mustWork = FALSE)
        break
      }
    }
    if (nzchar(hadoop_home)) {
      Sys.setenv(HADOOP_HOME = hadoop_home)
      Sys.setenv(PATH = paste(normalizePath(file.path(hadoop_home, "bin"), winslash = "\\"), Sys.getenv("PATH"), sep = ";"))
      java_opts <- paste0(
        "-Dhadoop.home.dir=", hadoop_home,
        " -Djava.library.path=", file.path(hadoop_home, "bin")
      )
    } else {
      warning(
        "Windows Hadoop native files not found. If Spark fails, install winutils.exe in C:/hadoop/bin or set HADOOP_HOME.",
        call. = FALSE
      )
    }
  }

  sc <- sparklyr::spark_connect(
    master = "local",
    version = spark_version,
    config = list(
      spark.sql.shuffle.partitions = "8",
      spark.ui.enabled = "false",
      `spark.driver.extraJavaOptions` = java_opts,
      `spark.executor.extraJavaOptions` = java_opts
    )
  )
  on.exit(sparklyr::spark_disconnect(sc), add = TRUE)

  sdf <- sparklyr::spark_read_parquet(sc, "eval_train_labels", path)
  as.character(dplyr::collect(dplyr::select(sdf, emotion))$emotion)
}

# ------------------------------------------------------------------------------
# Self-test (not run on source)
# Run with: EVAL_UTILS_SELFTEST=TRUE Rscript R/eval_utils.R
# ------------------------------------------------------------------------------

if (identical(Sys.getenv("EVAL_UTILS_SELFTEST"), "TRUE")) {
  cls <- c("a", "b", "c")
  a <- c("a", "a", "b", "b")
  p <- c("a", "b", "b", "b")
  stopifnot(
    nrow(confusion_matrix(a, p, cls)) == 9L,
    sum(confusion_matrix(a, p, cls)$n) == 4L,
    identical(per_class_metrics(a, p, cls)$f1[3], 0),
    isTRUE(all.equal(macro_f1(a, p, cls), (2 / 3 + 4 / 5 + 0) / 3)),
    isTRUE(all.equal(accuracy(a, p), 0.75)),
    identical(majority_baseline(c("a", "a", "b"), a)$class, "a")
  )
  cat("eval_utils self-test: PASS\n")
}
