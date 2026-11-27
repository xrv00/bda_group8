# ==============================================================================
# 07_evaluate.R
#
# Purpose:  Evaluate all 3 models x 3 feature sets with one shared metric
#           implementation, compare against a majority-class baseline, and
#           export metrics tables plus two PNG plots.
#
# Inputs:   outputs/predictions_naive_bayes.csv
#           outputs/predictions_logistic_regression.csv
#           outputs/predictions_random_forest.csv
#           data/processed/audio_train.parquet  (emotion column, baseline only)
#           R/eval_utils.R
#
# Outputs:  outputs/metrics_summary.csv
#           outputs/metrics_per_class.csv
#           outputs/plot4_confusion.png
#           outputs/plot4b_confusion_by_featureset.png
#           outputs/plot4_confusion_all.png
#           outputs/plot5_macro_f1.png
#
# Notes:    The first two prediction files use the columns
#           row_id, emotion, prediction, feature_set. `normalise_predictions()`
#           maps them onto the shared schema without editing the source files.
#           row_id is read as character because it is a 64-bit Spark Long.
#
# AI-assisted (claude), 2026-11-24
# AI-assisted (opencode), 2026-11-27
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
  library(ggplot2)
  library(scales)
})

source("R/eval_utils.R")

# ------------------------------------------------------------------------------
# Same text style as the EDA plots in R/03_eda.R, so all dashboard plots match
# ------------------------------------------------------------------------------
theme_eda <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title          = element_text(face = "bold", size = 14, colour = "#0b0b0b"),
      plot.subtitle       = element_text(colour = "#52514e", margin = margin(b = 10)),
      plot.title.position = "plot",
      axis.title          = element_text(colour = "#52514e", size = 12),
      axis.title.x        = element_text(margin = margin(t = 8)),
      axis.title.y        = element_text(margin = margin(r = 8)),
      axis.text           = element_text(colour = "#0b0b0b", size = 12),
      legend.title        = element_text(colour = "#52514e", size = 12),
      legend.text         = element_text(size = 12),
      panel.grid.minor    = element_blank(),
      panel.grid.major    = element_line(colour = "#e6e5e1", linewidth = 0.3),
      legend.position     = "none",
      plot.margin         = margin(12, 16, 12, 12)
    )
}

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

OUT_DIR <- "outputs"
DATA_DIR <- file.path("data", "processed")

CLASSES <- c("anger", "fear", "joy", "love", "sadness", "surprise")
FEATURE_SETS <- c("audio", "lyrics", "combined")
MODELS <- c("naive_bayes", "logistic_regression", "random_forest")

MODEL_FILES <- c(
  naive_bayes = "predictions_naive_bayes.csv",
  logistic_regression = "predictions_logistic_regression.csv",
  random_forest = "predictions_random_forest.csv"
)

# Plot sizes for easy teammate adjustment
PLOT4_SIZE     <- c(width = 6.2, height = 5.4)
PLOT4B_SIZE    <- c(width = 14, height = 5.5)
PLOT4_ALL_SIZE <- c(width = 16, height = 12)
PLOT5_SIZE     <- c(width = 9, height = 6)

# ------------------------------------------------------------------------------
# Normalisation and input checks
# ------------------------------------------------------------------------------

#' Normalise one prediction file to the shared schema
#'
#' @param df Data frame read from one of the prediction CSVs.
#' @param model_name Model name from the `MODEL_FILES` lookup.
#' @return Tibble with columns row_id, feature_set, model, actual, predicted.
#'   `feature_set` is a factor with levels audio, lyrics, combined.
normalise_predictions <- function(df, model_name) {
  missing <- setdiff(c("row_id", "feature_set"), names(df))
  if (length(missing)) stop("missing columns in ", model_name, ": ", paste(missing, collapse = ", "))

  actual_col <- intersect(names(df), c("actual", "emotion"))
  pred_col <- intersect(names(df), c("predicted", "prediction"))
  if (length(actual_col) != 1L || length(pred_col) != 1L) {
    stop("cannot find actual/predicted columns in ", model_name)
  }

  predicted_values <- as.character(df[[pred_col]])
  numeric_like <- !is.na(suppressWarnings(as.numeric(predicted_values)))
  if (all(numeric_like)) {
    stop("prediction values in ", model_name, " are numeric indexes; a label mapping is required")
  }
  cat(sprintf(
    "  %s: prediction values are emotion names (%s)\n",
    model_name, paste(sort(unique(predicted_values)), collapse = ", ")
  ))

  out <- tibble(
    row_id = as.character(df$row_id),
    feature_set = as.character(df$feature_set),
    model = model_name,
    actual = as.character(df[[actual_col]]),
    predicted = predicted_values
  )

  out$feature_set[out$feature_set == "lyric"] <- "lyrics"
  bad_sets <- setdiff(unique(out$feature_set), FEATURE_SETS)
  if (length(bad_sets)) stop("unknown feature_set: ", paste(bad_sets, collapse = ", "))
  bad_labels <- setdiff(unique(c(out$actual, out$predicted)), CLASSES)
  if (length(bad_labels)) stop("unknown emotion label: ", paste(bad_labels, collapse = ", "))

  out$feature_set <- factor(out$feature_set, levels = FEATURE_SETS)
  out
}

cat("Reading and normalising prediction files...\n")
predictions <- bind_rows(lapply(names(MODEL_FILES), function(nm) {
  path <- file.path(OUT_DIR, MODEL_FILES[[nm]])
  df <- read_csv(path, col_types = cols(row_id = col_character()), show_col_types = FALSE)
  normalise_predictions(df, nm)
}))

check <- function(label, ok, detail = "") {
  cat(sprintf("[%s] %s%s\n", if (ok) "PASS" else "FAIL", label,
              if (nzchar(detail)) paste0(" -- ", detail) else ""))
  if (!ok) stop("input check failed: ", label)
}

cat("\nCross-file checks (normalised data):\n")
by_model <- split(predictions, predictions$model)
check("columns and order identical",
      length(unique(lapply(by_model, names))) == 1L,
      paste(names(predictions), collapse = ", "))
check("label spellings and case identical",
      length(unique(lapply(by_model, function(x) sort(unique(c(x$actual, x$predicted)))))) == 1L)
check("feature_set values identical",
      all(vapply(by_model, function(x) identical(levels(x$feature_set), FEATURE_SETS), logical(1))))

for (fs in FEATURE_SETS) {
  ids <- lapply(by_model, function(x) sort(unique(x$row_id[x$feature_set == fs])))
  check(paste0("row_id set identical across models: ", fs),
        identical(ids[[1]], ids[[2]]) && identical(ids[[1]], ids[[3]]),
        paste(lengths(ids), collapse = "/"))
}
for (nm in names(by_model)) {
  check(paste0("no duplicate row_id: ", nm),
        !anyDuplicated(by_model[[nm]][, c("feature_set", "row_id")]))
}
for (nm in names(by_model)[-1]) {
  m <- inner_join(
    by_model[[1]][, c("feature_set", "row_id", "actual")],
    by_model[[nm]][, c("feature_set", "row_id", "actual")],
    by = c("feature_set", "row_id"), suffix = c("_a", "_b")
  )
  check(paste0("actual identical for shared row_id: ", names(by_model)[1], " vs ", nm),
        nrow(m) > 0L && all(m$actual_a == m$actual_b))
}

# ------------------------------------------------------------------------------
# Metrics for all 9 model x feature_set combinations
# ------------------------------------------------------------------------------

metric_row <- function(model_name, fs_name, actual, predicted) {
  pc <- per_class_metrics(actual, predicted, CLASSES)
  tibble(
    model = model_name,
    feature_set = as.character(fs_name),
    macro_f1 = mean(pc$f1),
    macro_precision = mean(pc$precision),
    macro_recall = mean(pc$recall),
    accuracy = accuracy(actual, predicted)
  )
}

summary_rows <- list()
per_class_rows <- list()
cm_rows <- list()

for (nm in MODELS) {
  for (fs in FEATURE_SETS) {
    sub <- predictions[predictions$model == nm & predictions$feature_set == fs, ]
    summary_rows[[paste(nm, fs, sep = "|")]] <- metric_row(nm, fs, sub$actual, sub$predicted)

    per_class_rows[[paste(nm, fs, sep = "|")]] <- per_class_metrics(sub$actual, sub$predicted, CLASSES) %>%
      mutate(model = nm, feature_set = as.character(fs)) %>%
      select(model, feature_set, class, precision, recall, f1, support)

    cm <- confusion_matrix(sub$actual, sub$predicted, CLASSES)
    cm_rows[[paste(nm, fs, sep = "|")]] <- cm %>%
      mutate(model = nm, feature_set = as.character(fs), n = as.numeric(n)) %>%
      group_by(actual) %>%
      mutate(prop = n / sum(n)) %>%
      ungroup() %>%
      select(model, feature_set, actual, predicted, n, prop)
  }
}

# ------------------------------------------------------------------------------
# Majority-class baseline (train labels from audio_train.parquet)
# ------------------------------------------------------------------------------

cat("\nReading train labels for the majority baseline...\n")
train_labels <- read_train_labels(file.path(DATA_DIR, "audio_train.parquet"))
base_actual <- predictions[predictions$model == MODELS[1] & predictions$feature_set == FEATURE_SETS[1], ]$actual
baseline <- majority_baseline(train_labels, base_actual)
base_predicted <- rep(baseline$class, length(base_actual))
base_pc <- per_class_metrics(base_actual, base_predicted, CLASSES)

baseline_row <- tibble(
  model = "majority_baseline",
  feature_set = "all",
  macro_f1 = baseline$macro_f1,
  macro_precision = mean(base_pc$precision),
  macro_recall = mean(base_pc$recall),
  accuracy = baseline$accuracy
)
baseline_pc <- base_pc %>%
  mutate(model = "majority_baseline", feature_set = "all") %>%
  select(model, feature_set, class, precision, recall, f1, support)

metrics_summary <- bind_rows(c(summary_rows, list(baseline = baseline_row)))
metrics_per_class <- bind_rows(c(per_class_rows, list(baseline = baseline_pc)))
cm_all <- bind_rows(cm_rows)
cm_all$model <- factor(cm_all$model, levels = MODELS)
cm_all$feature_set <- factor(cm_all$feature_set, levels = FEATURE_SETS)

write_csv(metrics_summary, file.path(OUT_DIR, "metrics_summary.csv"))
write_csv(metrics_per_class, file.path(OUT_DIR, "metrics_per_class.csv"))

# ------------------------------------------------------------------------------
# Console table, sorted by macro F1
# ------------------------------------------------------------------------------

cat("\n======================================================================\n")
cat("METRICS (sorted by macro F1)\n")
cat("======================================================================\n")
console_table <- metrics_summary %>%
  arrange(desc(macro_f1)) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))
print(as.data.frame(console_table), row.names = FALSE)

# ------------------------------------------------------------------------------
# Sanity warnings
# ------------------------------------------------------------------------------

cat("\nSanity checks:\n")
for (i in seq_len(nrow(metrics_summary))) {
  row <- metrics_summary[i, ]
  if (row$model == "majority_baseline") next
  if (row$macro_f1 <= baseline$macro_f1) {
    warning(sprintf(
      "%s / %s: macro F1 %.4f is at or below baseline %.4f",
      row$model, row$feature_set, row$macro_f1, baseline$macro_f1
    ), call. = FALSE)
  }
}
for (nm in MODELS) {
  for (fs in FEATURE_SETS) {
    sub <- predictions[predictions$model == nm & predictions$feature_set == fs, ]
    counts <- table(factor(sub$predicted, levels = CLASSES))
    missing_classes <- names(counts)[counts == 0L]
    if (length(missing_classes)) {
      warning(sprintf(
        "%s / %s never predicts: %s | predictions per class: %s",
        nm, fs,
        paste(missing_classes, collapse = ", "),
        paste(names(counts), counts, sep = "=", collapse = ", ")
      ), call. = FALSE)
    }
  }
}

# ------------------------------------------------------------------------------
# Best model (used by Plot 4 and findings)
# ------------------------------------------------------------------------------

model_metrics <- metrics_summary[metrics_summary$model != "majority_baseline", ]
best <- model_metrics[which.max(model_metrics$macro_f1), ]
best_model <- as.character(best$model)
best_fs <- as.character(best$feature_set)

# ------------------------------------------------------------------------------
# Plot 4: row-normalised confusion matrices
# AI-assisted (opencode), 2026-11-24
# ------------------------------------------------------------------------------

cm_best <- cm_all[cm_all$model == best_model & cm_all$feature_set == best_fs, ]
cm_best$label_colour <- ifelse(cm_best$prop > 0.5, "white", "grey20")
cm_best$is_diag <- as.character(cm_best$actual) == as.character(cm_best$predicted)

p4 <- ggplot(cm_best, aes(x = predicted, y = actual, fill = prop)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_tile(
    data = cm_best[cm_best$is_diag, ],
    aes(x = predicted, y = actual),
    fill = NA, colour = "grey15", linewidth = 1.1, inherit.aes = FALSE
  ) +
  geom_text(aes(label = sprintf("%.2f", prop), colour = label_colour),
            size = 3.2, fontface = "bold", vjust = -0.2) +
  geom_text(aes(label = paste0("n=", scales::comma(n)), colour = label_colour),
            size = 2.3, vjust = 1.4, alpha = 0.9) +
  scale_colour_identity() +
  scale_fill_gradient(low = "white", high = "#08306B", limits = c(0, 1), name = "Proportion of\nactual class") +
  labs(
    title = paste0("Confusion matrix: ", best_model, " / ", best_fs),
    subtitle = "Rows = actual emotion, normalised by actual class count",
    x = "Predicted emotion", y = "Actual emotion"
  ) +
  theme_eda() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.text = element_text(size = 7.5)
  )

ggsave(file.path(OUT_DIR, "plot4_confusion.png"), p4,
       width = PLOT4_SIZE[["width"]], height = PLOT4_SIZE[["height"]], dpi = 300, bg = "white")

# Plot 4b: best model across audio, lyrics, combined (1 x 3)
cm_best_model <- cm_all[cm_all$model == best_model, ]
cm_best_model$feature_set <- factor(cm_best_model$feature_set, levels = FEATURE_SETS)
cm_best_model$label_colour <- ifelse(cm_best_model$prop > 0.5, "white", "grey20")
cm_best_model$is_diag <- as.character(cm_best_model$actual) == as.character(cm_best_model$predicted)

p4b <- ggplot(cm_best_model, aes(x = predicted, y = actual, fill = prop)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_tile(
    data = cm_best_model[cm_best_model$is_diag, ],
    aes(x = predicted, y = actual),
    fill = NA, colour = "grey15", linewidth = 0.8, inherit.aes = FALSE
  ) +
  geom_text(aes(label = sprintf("%.2f", prop), colour = label_colour), size = 3.2, fontface = "bold") +
  scale_colour_identity() +
  facet_wrap(~ feature_set, nrow = 1) +
  scale_fill_gradient(low = "white", high = "#08306B", limits = c(0, 1), name = "Proportion of actual class") +
  labs(
    title = paste0("Confusion matrices by feature set: ", best_model),
    subtitle = "Rows = actual emotion, normalised by actual class count",
    x = "Predicted emotion", y = "Actual emotion"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    strip.text = element_text(face = "bold", colour = "black", size = 12),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.background = element_rect(fill = "white", colour = NA)
  )

ggsave(file.path(OUT_DIR, "plot4b_confusion_by_featureset.png"), p4b,
       width = PLOT4B_SIZE[["width"]], height = PLOT4B_SIZE[["height"]], dpi = 300, bg = "white")

# Plot 4 (all): fixed 9-panel grid
cm_all$label_colour <- ifelse(cm_all$prop > 0.5, "white", "grey20")
p4_all <- ggplot(cm_all, aes(x = predicted, y = actual, fill = prop)) +
  geom_tile(colour = "white", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%.2f", prop), colour = label_colour), size = 2.1) +
  scale_colour_identity() +
  facet_grid(model ~ feature_set) +
  scale_fill_gradient(low = "white", high = "#08306B", limits = c(0, 1), name = "Proportion of actual class") +
  labs(
    title = "Confusion matrices by model and feature set",
    subtitle = "Rows = actual emotion, normalised by actual class count",
    x = "Predicted emotion", y = "Actual emotion"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", colour = "black", size = 11),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.background = element_rect(fill = "white", colour = NA)
  )

ggsave(file.path(OUT_DIR, "plot4_confusion_all.png"), p4_all,
       width = PLOT4_ALL_SIZE[["width"]], height = PLOT4_ALL_SIZE[["height"]], dpi = 300, bg = "white")

cat("Plot 4 files:", file.path(OUT_DIR, "plot4_confusion.png"), "|",
    file.path(OUT_DIR, "plot4b_confusion_by_featureset.png"), "|",
    file.path(OUT_DIR, "plot4_confusion_all.png"), "\n")
cat("Best model/feature set for plot4:", best_model, "/", best_fs, "\n")

# ------------------------------------------------------------------------------
# Plot 5: grouped macro F1 bars with baseline reference
# AI-assisted (opencode), 2026-11-24
# ------------------------------------------------------------------------------

p5_data <- metrics_summary[metrics_summary$model != "majority_baseline", ]
p5_data$model <- factor(p5_data$model, levels = MODELS)
p5_data$feature_set <- factor(p5_data$feature_set, levels = FEATURE_SETS)

model_labels <- c(
  naive_bayes = "Naive Bayes",
  logistic_regression = "Logistic Regression",
  random_forest = "Random Forest"
)
cb_palette <- c(
  naive_bayes = "#8d99b0",
  logistic_regression = "#2c3e50",
  random_forest = "#5b6b86"
)

baseline_label <- sprintf("Baseline\n(%.3f)", baseline$macro_f1)

p5 <- ggplot(p5_data, aes(x = feature_set, y = macro_f1, fill = model)) +
  geom_col(position = position_dodge(width = 0.9), width = 0.75) +
  geom_text(
    aes(label = sprintf("%.3f", macro_f1)),
    position = position_dodge(width = 0.9),
    vjust = -0.4, size = 4.2, colour = "#222222"
  ) +
  geom_hline(yintercept = baseline$macro_f1, linetype = "dashed", colour = "#444444", linewidth = 0.7) +
  annotate(
    "text",
    x = Inf, y = baseline$macro_f1,
    label = baseline_label,
    hjust = -0.08, lineheight = 0.9,
    colour = "#444444", size = 4.2
  ) +
  coord_cartesian(clip = "off") +
  scale_fill_manual(values = cb_palette, labels = model_labels, name = "Model") +
  scale_y_continuous(limits = c(0, 0.55), breaks = seq(0, 0.5, 0.1), expand = c(0, 0)) +
  labs(
    title = "Macro F1 by model and feature set",
    subtitle = "Test set, 98,950 songs per feature set. Dashed line = majority-class baseline.",
    x = "Feature set",
    y = "Macro F1"
  ) +
  theme_eda() +
  theme(
    panel.grid.major.x = element_blank(),
    legend.position = "bottom",
    plot.margin = margin(12, 85, 12, 12)
  )

ggsave(file.path(OUT_DIR, "plot5_macro_f1.png"), p5,
       width = PLOT5_SIZE[["width"]], height = PLOT5_SIZE[["height"]], dpi = 300, bg = "white")
cat("Plot 5 file:", file.path(OUT_DIR, "plot5_macro_f1.png"), "\n")
cat("Baseline macro F1 used:", sprintf("%.3f", baseline$macro_f1), "\n")

# ------------------------------------------------------------------------------
# Final findings
# ------------------------------------------------------------------------------

cat("\n======================================================================\n")
cat("FINDINGS\n")
cat("======================================================================\n")
cat(sprintf(
  "BEST: %s / %s (macro F1 = %.4f, accuracy = %.4f)\n",
  best$model, best$feature_set, best$macro_f1, best$accuracy
))

cat("\nCombined vs single feature sets (macro F1):\n")
for (nm in MODELS) {
  m <- model_metrics[model_metrics$model == nm, ]
  f <- function(fs) m$macro_f1[m$feature_set == fs]
  cat(sprintf(
    "  %-20s combined %.4f vs audio %.4f (%s) | vs lyrics %.4f (%s)\n",
    nm, f("combined"), f("audio"), ifelse(f("combined") > f("audio"), "BETTER", "NOT BETTER"),
    f("lyrics"), ifelse(f("combined") > f("lyrics"), "BETTER", "NOT BETTER")
  ))
}

weak <- metrics_per_class[
  metrics_per_class$model == best$model & metrics_per_class$feature_set == best$feature_set,
]
weak <- weak[order(weak$recall), ][1:3, ]
cat("\n3 largest per-class weaknesses for the best model (lowest recall):\n")
for (i in seq_len(nrow(weak))) {
  cat(sprintf(
    "  %d. %-9s recall %.4f | precision %.4f | F1 %.4f | support %d\n",
    i, weak$class[i], weak$recall[i], weak$precision[i], weak$f1[i], weak$support[i]
  ))
}
cat("\nBaseline used:", baseline$class, "| macro F1 =", round(baseline$macro_f1, 4),
    "| accuracy =", round(baseline$accuracy, 4), "\n")
