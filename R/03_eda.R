# 03_eda.R
# Owner: Ong Chia Yin (Data Scientist)
# Purpose: SQL queries, plots 1 to 3 and 6

# Input   : data/processed/spotify_structured.sqlite  (from R/01_ingest.R)
#           -> view `model_ready` (493,820 rows: no duplicates, 6 valid emotions)
# Output  : outputs/plot1.png, plot2.png, plot3.png, plot6.png
#           outputs/table1_*.csv ... table6_*.csv (summary tables for the report)
#           outputs/eda_log.txt (row counts + session info)

# AI-assisted (Claude), 18 Nov 2026

# -----------------------------------------------------------------------------
# 0. SETTINGS
# -----------------------------------------------------------------------------

# Path to the SQLite database (relative to the project root, see README "Data")
db_path <- "data/processed/spotify_structured.sqlite"

# Folder where plots and tables are saved (see README "Outputs")
out_dir <- "outputs"

# Audio features shown in Plot 3 (confirmed with the Spark Feature Engineer:
# the same 9 features as the audio feature set used for modelling).
audio_features <- c("energy", "valence", "danceability", "tempo", "loudness",
                    "speechiness", "liveness", "acousticness", "instrumentalness")

# Expected row count of model_ready view (from the Stage 1)
expected_rows <- 493820

# Size and sharpness of the saved pictures.
plot_width  <- 8     # inches
plot_height <- 5     # inches 
plot_dpi    <- 300   # publication quality


# -----------------------------------------------------------------------------
# SECTION 1 - LOAD THE PACKAGES
# -----------------------------------------------------------------------------

# This section first checks they are installed then stops with a clear message 
# if any are missing and prompt for user to install them

required_pkgs <- c("DBI", "RSQLite", "ggplot2", "dplyr", "tidyr", "scales")
missing_pkgs  <- required_pkgs[!vapply(required_pkgs, requireNamespace,
                                       logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing packages: ", paste(missing_pkgs, collapse = ", "),
       "\nInstall them with: install.packages(c(\"",
       paste(missing_pkgs, collapse = "\", \""), "\"))")
}

suppressPackageStartupMessages({
  library(DBI)
  library(RSQLite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
})


# -----------------------------------------------------------------------------
# SECTION 2 - COLOURS AND PLOT STYLE (SHARED BY EVERY PLOT)
# -----------------------------------------------------------------------------

# Fixed order used on every x-axis (largest class to smallest)
emotion_levels <- c("joy", "sadness", "anger", "fear", "love", "surprise")

# One colour per emotion. Emotion names are always shown on the axis as a second cue.
emotion_cols <- c(
  joy      = "#eda100",  # yellow
  sadness  = "#2a78d6",  # blue
  anger    = "#e34948",  # red
  fear     = "#6250d6",  # violet
  love     = "#e87ba4",  # pink
  surprise = "#1baf7a"   # green
)

ink_primary   <- "#0b0b0b"   # main text
ink_secondary <- "#52514e"   # axis text, captions
grid_col      <- "#e6e5e1"   # light gridlines

theme_bda <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title.position = "plot",
      plot.caption.position = "plot",
      plot.title    = element_text(face = "bold", colour = ink_primary,
                                   size = rel(1.15)),
      plot.subtitle = element_text(colour = ink_secondary,
                                   margin = margin(b = 10)),
      plot.caption  = element_text(colour = ink_secondary, size = rel(0.8),
                                   hjust = 0, margin = margin(t = 10)),
      axis.title    = element_text(colour = ink_secondary),
      axis.text     = element_text(colour = ink_secondary),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(colour = grid_col, linewidth = 0.4),
      legend.position = "none",
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(12, 16, 12, 12)
    )
}

# Helper: save a plot with the standard size and report where it went
save_plot <- function(plot, file_name, width = plot_width,
                      height = plot_height) {
  path <- file.path(out_dir, file_name)
  ggsave(path, plot, width = width, height = height, dpi = plot_dpi,
         bg = "white")
  log_msg("Saved ", path)
}

# Helper: save a table as CSV
save_table <- function(df, file_name) {
  path <- file.path(out_dir, file_name)
  write.csv(df, path, row.names = FALSE)
  log_msg("Saved ", path)
}


# -----------------------------------------------------------------------------
# SECTION 3 - SET UP OUTPUT FOLDER AND LOG
# -----------------------------------------------------------------------------

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(out_dir, "eda_log.txt")
cat("EDA log -", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n",
    file = log_file)

# Print a message to the console AND append it to the log file
log_msg <- function(...) {
  msg <- paste0(...)
  message(msg)
  cat(msg, "\n", file = log_file, append = TRUE)
}


# -----------------------------------------------------------------------------
# SECTION 4 - CONNECT TO SQLITE (read-only) AND CHECK THE DATA
# -----------------------------------------------------------------------------

if (!file.exists(db_path)) {
  stop("Database not found at: ", db_path,
       "\nCheck db_path in Section 0 of this script.")
}

con <- tryCatch(
  dbConnect(RSQLite::SQLite(), db_path, flags = SQLITE_RO),
  error = function(e) stop("Could not open the database: ", conditionMessage(e))
)

if (!"model_ready" %in% dbListTables(con)) {
  stop("The view 'model_ready' is missing. Re-run R/01_ingest.R")
}

n_rows <- dbGetQuery(con, "SELECT COUNT(*) AS n FROM model_ready")$n
log_msg("model_ready rows: ", format(n_rows, big.mark = ","))
if (n_rows != expected_rows) {
  warning("model_ready has ", n_rows, " rows; expected ", expected_rows,
          ". The database may have been rebuilt - check with the Data Engineer.")
}

found_emotions <- dbGetQuery(con, "SELECT DISTINCT emotion FROM model_ready")$emotion
if (!setequal(found_emotions, emotion_levels)) {
  stop("Unexpected emotion labels: ",
       paste(setdiff(found_emotions, emotion_levels), collapse = ", "))
}



# -----------------------------------------------------------------------------
# SECTION 5 - PLOT 1: Emotion class distribution
# (To find out whether six emotion classes are balanced)
# -----------------------------------------------------------------------------

# Step 1: SQL counts the songs in each emotion (GROUP BY).
emotion_counts <- dbGetQuery(con, "
  SELECT emotion, COUNT(*) AS n_songs
  FROM model_ready
  GROUP BY emotion
") |>
  # Step 2: R adds the percentage and saves the table.
  mutate(pct     = n_songs / sum(n_songs),
         emotion = factor(emotion, levels = emotion_levels)) |>
  arrange(emotion)


save_table(emotion_counts |> mutate(pct = round(pct * 100, 2)),
           "table1_emotion_counts.csv")

# Step 3: Draw a bar chart with the count and % above each bar.
plot1 <- ggplot(emotion_counts, aes(x = emotion, y = n_songs, fill = emotion)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = paste0(comma(n_songs), "\n(",
                               percent(pct, accuracy = 0.1), ")")),
            vjust = -0.25, size = 3.5, colour = ink_primary, lineheight = 0.9) +
  scale_fill_manual(values = emotion_cols) +
  scale_y_continuous(labels = label_comma(),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(title    = "Plot 1: Joy, sadness and anger dominate the emotion distribution",
       subtitle = "Number of songs per emotion label",
       x = "Emotion", y = "Number of songs") +
  theme_bda()

save_plot(plot1, "plot1.png") # Output: plot1.png, table1_emotion_counts.csv


# -----------------------------------------------------------------------------
# SECTION 6 - PLOT 2: Valence by emotion
# To find out does audio "positiveness" (valence) differ between emotions
# -----------------------------------------------------------------------------

# Step 1: SQL works out the count, mean, min and max for each emotion.
valence_sql <- dbGetQuery(con, "
  SELECT emotion,
         COUNT(*)     AS n_songs,
         AVG(valence) AS mean_valence,
         MIN(valence) AS min_valence,
         MAX(valence) AS max_valence
  FROM model_ready
  GROUP BY emotion
")

# Step 2: R works out the median and quartiles (SQLite has no MEDIAN function).
valence_df <- dbGetQuery(con, "SELECT emotion, valence FROM model_ready") |>
  mutate(emotion = factor(emotion, levels = emotion_levels))

valence_quart <- valence_df |>
  group_by(emotion) |>
  summarise(q1_valence     = quantile(valence, 0.25),
            median_valence = median(valence),
            q3_valence     = quantile(valence, 0.75),
            .groups = "drop") |>
  mutate(emotion = as.character(emotion))

# Step 3: Join both into one table and save it.
valence_summary <- valence_sql |>
  left_join(valence_quart, by = "emotion") |>
  mutate(emotion = factor(emotion, levels = emotion_levels)) |>
  arrange(emotion) |>
  mutate(across(where(is.numeric) & !n_songs, ~ round(.x, 2)))

save_table(valence_summary, "table2_valence_by_emotion.csv")

# Step 4: Draw a boxplot, with the median written beside each box.
plot2 <- ggplot(valence_df, aes(x = emotion, y = valence, fill = emotion)) +
  geom_boxplot(width = 0.6, outlier.shape = NA, colour = ink_secondary,
               linewidth = 0.4) +
  # Median value written beside each box
  geom_text(data = valence_quart |>
              mutate(emotion = factor(emotion, levels = emotion_levels)),
            aes(x = as.numeric(emotion) + 0.34, y = median_valence,
                label = round(median_valence, 1)),
            inherit.aes = FALSE, hjust = 0, size = 3.5, colour = ink_primary) +
  scale_fill_manual(values = emotion_cols) +
  scale_y_continuous(breaks = seq(0, 100, 25)) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(title    = "Plot 2: Musical valence distributions overlap across emotion classes",
       subtitle = "Musical positiveness of the audio, 0 = negative, 100 = positive (number = median)",
       x = "Emotion", y = "Valence (0-100)") +
  theme_bda()

save_plot(plot2, "plot2.png")     # Output: plot2.png, table2_valence_by_emotion.csv
rm(valence_df); invisible(gc())   # free memory


# -----------------------------------------------------------------------------
# SECTION 7 - PLOT 3: Correlation between audio features
# To find out which audio features move together (multicollinearity)
# -----------------------------------------------------------------------------

# Step 1: SQL fetches the 9 audio features for every song.
audio_sql <- paste0("SELECT ", paste(audio_features, collapse = ", "),
                    " FROM model_ready")
audio_df <- dbGetQuery(con, audio_sql)

# Step 2: R calculates the Spearman correlation between every pair.
#         Spearman compares songs by their ranking, so it is not distorted by
#         skewed features (speechiness, liveness, instrumentalness) or outliers.
cor_mat <- cor(audio_df, method = "spearman")
rm(audio_df); invisible(gc())

save_table(data.frame(feature = rownames(cor_mat), round(cor_mat, 3),
                      row.names = NULL),
           "table3_audio_correlations.csv")

# Step 3: keep only the lower triangle (the upper half is a mirror copy).
cor_long <- as.data.frame(as.table(cor_mat)) |>
  setNames(c("feature_x", "feature_y", "r")) |>
  mutate(ix = match(feature_x, audio_features),
         iy = match(feature_y, audio_features)) |>
  filter(ix < iy) |>
  mutate(feature_x = factor(feature_x, levels = audio_features),
         feature_y = factor(feature_y, levels = rev(audio_features)))

# Step 4: draw a heatmap - red = move together, blue = move in opposite directions.
plot3 <- ggplot(cor_long, aes(x = feature_x, y = feature_y, fill = r)) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = sub("^-0\\.00$", "0.00", sprintf("%.2f", r)),
                colour = abs(r) > 0.5),
            size = 3.3, show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = ink_primary)) +
  scale_fill_gradient2(low = "#2a78d6", mid = "#f0efec", high = "#e34948",
                       midpoint = 0, limits = c(-1, 1),
                       name = "Spearman r") +
  labs(title    = "Plot 3: Energy is strongly associated with loudness and lower acousticness",
       subtitle = "Spearman correlation between the 9 audio features",
       x = NULL, y = NULL) +
  theme_bda() +
  theme(legend.position = "right",
        panel.grid.major.y = element_blank(),
        axis.text.x = element_text(angle = 40, hjust = 1))

save_plot(plot3, "plot3.png", width = 8, height = 6.5) # Output: plot3.png, table3_audio_correlations.csv


# -----------------------------------------------------------------------------
# SECTION 8 - PLOT 6: Similarity score by emotion
# To find out whether songs of some emotions are more similar 
# to their listed "similar songs" (semi-structured Similar Songs data)
# -----------------------------------------------------------------------------

zoom_min <- 0.90   # lower edge of the zoomed y-axis

# Step 1: fetch every song's emotion and mean_similarity.
sim_df <- dbGetQuery(con, "SELECT emotion, mean_similarity FROM model_ready") |>
  mutate(emotion = factor(emotion, levels = emotion_levels))

# Step 2: summarise each emotion (median, quartiles, outliers) and save it.
sim_summary <- sim_df |>
  group_by(emotion) |>
  summarise(n_songs         = n(),
            mean_sim        = mean(mean_similarity),
            q1_sim          = quantile(mean_similarity, 0.25),
            median_sim      = median(mean_similarity),
            q3_sim          = quantile(mean_similarity, 0.75),
            min_sim         = min(mean_similarity),
            n_below_zoom    = sum(mean_similarity < zoom_min),
            # Boxplot outliers: beyond 1.5 x IQR from the box edges
            n_outliers      = sum(mean_similarity < q1_sim - 1.5 * (q3_sim - q1_sim) |
                                    mean_similarity > q3_sim + 1.5 * (q3_sim - q1_sim)),
            pct_outliers    = round(100 * n_outliers / n_songs, 2),
            .groups = "drop") |>
  mutate(across(c(mean_sim, q1_sim, median_sim, q3_sim, min_sim),
                ~ round(.x, 4)))

save_table(sim_summary, "table6_similarity_by_emotion.csv")

n_below      <- sum(sim_summary$n_below_zoom)
n_outliers   <- sum(sim_summary$n_outliers)
pct_outliers <- 100 * n_outliers / n_rows
log_msg("Plot 6 outliers: ", comma(n_outliers), " songs (",
        round(pct_outliers, 1), "%), of which ", n_below, " are below ", zoom_min)

# Step 3: draw a boxplot zoomed in to 0.90-1.00, because most songs score 
#   above 0.95. On a 0-1 axis the boxes would be flat lines.
plot6 <- ggplot(sim_df, aes(x = emotion, y = mean_similarity, fill = emotion)) +
  # Outlier dots are not drawn: ~17,800 overlapping dots would form dark
  # columns that draw the eye away from the boxes (stated in the caption)
  geom_boxplot(width = 0.5, outlier.shape = NA, colour = ink_secondary,
               linewidth = 0.4) +
  # Median value written beside each box
  geom_text(data = sim_summary,
            aes(x = as.numeric(emotion) + 0.29, y = median_sim,
                label = sprintf("%.3f", median_sim)),
            inherit.aes = FALSE, hjust = 0, size = 3.3, colour = ink_primary) +
  scale_fill_manual(values = emotion_cols) +
  # Zoom like a camera: every song still counts in the box statistics
  # clip = "off" lets the last median label sit outside the panel edge
  coord_cartesian(ylim = c(zoom_min, 1.00), clip = "off") +
  labs(title    = "Plot 6: Song similarity remains high across all emotion categories",
       subtitle = "Average similarity to each song's 3 listed similar songs (number = median)",
       x = "Emotion", y = "Mean similarity score (0-1)",
       # Needed: the axis does not start at 0, so the reader must be told
       caption = paste0("Y-axis starts at ", sprintf("%.2f", zoom_min),
                        ". Outlier points (", comma(n_outliers), " songs, ",
                        sprintf("%.1f", pct_outliers), "%) are not drawn; ",
                        "all songs are counted in the boxes.")) +
  theme_bda() +
  theme(plot.margin = margin(12, 40, 12, 12))   # room for the last label

save_plot(plot6, "plot6.png") # Output: plot6.png, table6_similarity_by_emotion.csv
rm(sim_df); invisible(gc())


# -----------------------------------------------------------------------------
# SECTION 9 - FINISH: close connection, record session info
# -----------------------------------------------------------------------------

dbDisconnect(con)
cat("\n--- sessionInfo() ---\n", file = log_file, append = TRUE)
capture.output(sessionInfo(), file = log_file, append = TRUE)
log_msg("EDA finished. Outputs are in: ", normalizePath(out_dir))
