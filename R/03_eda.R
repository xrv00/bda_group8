# 03_eda.R
# Owner: Ong Chia Yin (Data Scientist)
# Purpose: SQL queries, plots 1 to 3 and 6
# AI-assisted (Claude), 2026-11-20


# -----------------------------------------------------------------------------
# SECTION 1 - PACKAGES
# -----------------------------------------------------------------------------

# Install once in the Console if missing:
# install.packages(c("DBI", "RSQLite", "ggplot2", "dplyr", "tidyr", "scales"))
library(DBI)
library(RSQLite)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)


# -----------------------------------------------------------------------------
# SECTION 2 - FILE PATHS
# -----------------------------------------------------------------------------

# Path to the SQLite database
db_path    <- "data/processed/spotify_structured.sqlite"

# Get queries from SQL 
sql_path   <- "sql/eda_queries.sql"

# Folder where plots and tables are saved 
output_dir <- "outputs"

# Stop early with a helpful message instead of a confusing error later.
if (!file.exists(db_path)) {
  stop("Database not found at ", db_path,
       ". Download spotify_structured.sqlite from the Google Drive link ",
       "in the README and put it in data/processed/.")
}
if (!file.exists(sql_path)) {
  stop("SQL file not found at ", sql_path, ". Make sure you opened the .Rproj file.")
}
if (!dir.exists(output_dir)) dir.create(output_dir)


# -----------------------------------------------------------------------------
# SECTION 3 - LOAD THE SQL QUERIES
# -----------------------------------------------------------------------------

# Every query in eda_queries.sql starts with a "-- name: something" line.
# This reads the file and keeps each query under its name, so the SQL
# lives in one place only.
read_queries <- function(path) {
  lines  <- readLines(path, warn = FALSE)
  starts <- grep("^--\\s*name:", lines)
  ends   <- c(starts[-1] - 1, length(lines))
  query_names <- trimws(sub("^--\\s*name:", "", lines[starts]))
  query_text  <- mapply(function(s, e) {
    block <- lines[(s + 1):e]
    
    # Drop comment lines, so notes and section dividers never reach SQLite.
    block <- block[!grepl("^\\s*--", block)]
    paste(block, collapse = "\n")
  }, starts, ends)
  setNames(trimws(query_text), query_names)
}

queries <- read_queries(sql_path)

# Run a query by its name and get the result back as a data frame.
run_sql <- function(name) {
  if (!name %in% names(queries)) {
    stop("No query called '", name, "' in ", sql_path)
  }
  dbGetQuery(con, queries[[name]])
}


# -----------------------------------------------------------------------------
# SECTION 4 - CONNECT TO THE DATABASE
# -----------------------------------------------------------------------------

# Read-only, so nothing in this script can change the shared database.
con <- dbConnect(RSQLite::SQLite(), db_path, flags = RSQLite::SQLITE_RO)
cat("Connected. Tables and views:", paste(dbListTables(con), collapse = ", "), "\n")


# -----------------------------------------------------------------------------
# SECTION 5 - CHECK WE HAVE THE RIGHT VERSION OF THE DATA
# -----------------------------------------------------------------------------

# Confirm the database matches the cleaned Stage 1 data described in the
# report before any plot is made: 551,443 songs, of which 492,943 are in
# the model_ready view used for this EDA. Stop if anything differs.
expected_songs       <- 551443
expected_model_ready <- 492943

n_songs       <- run_sql("count_songs")$n
n_model_ready <- run_sql("count_model_ready")$n
n_use_flag    <- run_sql("count_use_for_model")$n

check <- function(ok, message) {
  if (!ok) stop("Data check failed: ", message, call. = FALSE)
  cat("OK  -", message, "\n")
}

check(n_songs == expected_songs,
      paste0("songs has ", format(n_songs, big.mark = ","), " rows"))
check(n_model_ready == expected_model_ready,
      paste0("model_ready has ", format(n_model_ready, big.mark = ","), " rows"))
check(n_use_flag == n_model_ready,
      "use_for_model = 1 marks the same rows as the model_ready view")

# Emotion counts: should be six classes, from joy (largest) to surprise.
emotion_counts <- run_sql("emotion_counts")
check(nrow(emotion_counts) == 6, "model_ready has exactly 6 emotion classes")
print(emotion_counts)


# -----------------------------------------------------------------------------
# SECTION 6 - SHARED PLOT STYLE
# -----------------------------------------------------------------------------

# One look for every EDA plot, so they sit together nicely in the report
# and dashboard.

# Each emotion keeps the same colour in every plot, so a reader can follow
# one emotion across the dashboard. Emotion names always stay on the axis
# too, so the meaning never depends on colour alone (colour-blind safe).

emotion_colours <- c(
  joy      = "#eda100",   # yellow
  sadness  = "#2a78d6",   # blue
  anger    = "#e34948",   # red
  fear     = "#6250d6",   # violet
  love     = "#e87ba4",   # pink
  surprise = "#1baf7a"    # aqua
)

theme_eda <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title          = element_text(face = "bold", size = 14, colour = "#0b0b0b"),
      plot.subtitle       = element_text(colour = "#52514e", margin = margin(b = 10)),
      plot.title.position = "plot",
      axis.title          = element_text(colour = "#52514e"),
      axis.text           = element_text(colour = "#0b0b0b"),
      panel.grid.minor    = element_blank(),
      panel.grid.major    = element_line(colour = "#e6e5e1", linewidth = 0.3),
      legend.position     = "none",   # emotion names are on the axis already
      plot.margin         = margin(12, 16, 12, 12)
    )
}


# -----------------------------------------------------------------------------
# SECTION 7 - PLOT 1: EMOTION DISTRIBUTIONS
# -----------------------------------------------------------------------------

# Check distribution of classes. 
# Decide class weights.
# Prove why Macro F1 is fairest metrics.

write.csv(emotion_counts, file.path(output_dir, "eda_emotion_counts.csv"), row.names = FALSE)

plot1_data <- emotion_counts %>%
  mutate(
    label   = paste0(comma(n), "  (", sprintf("%.1f", pct), "%)"),   # always 1 decimal
    emotion = reorder(emotion, n)   # largest bar at the top
  )

plot1 <- ggplot(plot1_data, aes(x = n, y = emotion, fill = emotion)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = label), hjust = -0.08, size = 3.6, colour = "#0b0b0b") +
  scale_fill_manual(values = emotion_colours) +
  scale_x_continuous(
    labels = comma,
    expand = expansion(mult = c(0, 0.25))   # room on the right for the labels
  ) +
  labs(
    title    = "Emotion labels are highly imbalanced, from 38% joy to 1% surprise",
    subtitle = paste0("Number of songs per emotion label (n = ",
                      comma(sum(emotion_counts$n)), ")"),
    x        = "Number of songs",
    y        = NULL
  ) +
  theme_eda() +
  theme(panel.grid.major.y = element_blank())   # bars don't need horizontal lines

ggsave(file.path(output_dir, "plot1_emotion_distribution.png"),
       plot1, width = 9, height = 5, dpi = 300, bg = "white")
cat("Saved plot1_emotion_distribution.png\n")


# -----------------------------------------------------------------------------
# SECTION 8 - PLOT 2: VALANCE BY EMOTION
# -----------------------------------------------------------------------------

# Valence = how positive a song sounds (0 = sad/tense, 1 = happy).
# Plot 2 checks whether it differs between the emotions.
audio_means <- run_sql("audio_means_by_emotion")
write.csv(audio_means, file.path(output_dir, "eda_audio_means_by_emotion.csv"), row.names = FALSE)
print(audio_means[, c("emotion", "n", "valence", "energy", "danceability")])

valence_data <- run_sql("valence_by_emotion") %>%
  mutate(emotion = factor(emotion, levels = rev(emotion_counts$emotion)))

# Medians are hard to read off the axis when they are this close, so we
# print each one just above its box. SQLite has no MEDIAN(), so it's done in R.
valence_medians <- valence_data %>%
  group_by(emotion) %>%
  summarise(median = median(valence), .groups = "drop")

plot2 <- ggplot(valence_data, aes(x = valence, y = emotion, fill = emotion)) +

  geom_boxplot(width = 0.5, colour = "#52514e", linewidth = 0.4,
               outlier.size = 0.4, outlier.alpha = 0.3) +
  geom_text(data = valence_medians,
            
            # 2 decimals: valence is stored to 2 decimals
            aes(x = median, y = emotion, label = sprintf("%.2f", median)),
            nudge_y = 0.38, size = 3.3, colour = "#0b0b0b", inherit.aes = FALSE) +
  scale_fill_manual(values = emotion_colours) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(
    title    = "Emotions overlap heavily in valence; medians range only from 0.41 (sadness) to 0.49 (joy)",
    subtitle = paste0("Valence per emotion label (n = ",
                      comma(nrow(valence_data)), "); labels show medians"),
    x        = "Valence (0 = negative-sounding, 1 = positive-sounding)",
    y        = NULL
  ) +
  theme_eda() +
  theme(panel.grid.major.y = element_blank())

ggsave(file.path(output_dir, "plot2_valence_by_emotion.png"),
       plot2, width = 9, height = 5, dpi = 300, bg = "white")
cat("Saved plot2_valence_by_emotion.png\n")


# -----------------------------------------------------------------------------
# SECTION 9 - PLOT 3: CORRELATIONS BETWEEN AUDIO FEATURES
# -----------------------------------------------------------------------------

# Plot 3: correlations between the 9 audio features used in the models,
# to see which features overlap.
audio_features <- run_sql("audio_features")

# Pearson correlation (the standard measure of linear relationship).
cor_matrix <- cor(audio_features, method = "pearson")
write.csv(round(cor_matrix, 3), file.path(output_dir, "eda_audio_correlations.csv"))

# Readable names for the axes, in the same order as the SQL query.
feature_labels <- c(
  energy = "Energy", loudness = "Loudness", acousticness = "Acousticness",
  valence = "Valence", danceability = "Danceability", tempo = "Tempo",
  speechiness = "Speechiness", liveness = "Liveness",
  instrumentalness = "Instrumentalness"
)

# Keep only the lower triangle (the upper half repeats it; the diagonal is always 1).
cor_long <- as.data.frame(as.table(cor_matrix)) %>%
  rename(row_feature = Var1, col_feature = Var2, r = Freq) %>%
  filter(match(row_feature, colnames(cor_matrix)) > match(col_feature, colnames(cor_matrix))) %>%
  mutate(
    col_feature  = factor(feature_labels[as.character(col_feature)], levels = feature_labels),
    row_feature  = factor(feature_labels[as.character(row_feature)], levels = rev(feature_labels)),
    label_colour = ifelse(abs(r) >= 0.5, "white", "#0b0b0b")   # keep text readable on dark cells
  )

# Brown = negative, teal = positive, grey = no relationship. 
# Blue/red is avoided on purpose: those colours already mean sadness/anger elsewhere.
plot3 <- ggplot(cor_long, aes(x = col_feature, y = row_feature, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.2f", r), colour = label_colour), size = 3.1) +
  scale_colour_identity() +
  scale_fill_gradient2(low = "#8c510a", mid = "#f0efec", high = "#01665e",
                       midpoint = 0, limits = c(-1, 1),
                       breaks = c(-1, -0.5, 0, 0.5, 1),
                       name = "Correlation (r)") +
  coord_fixed() +
  labs(
    title    = "Energy, loudness and acousticness are the only strongly related features",
    subtitle = paste0("Pearson correlation of the 9 audio-model features (n = ",
                      comma(nrow(audio_features)), ")"),
    x = NULL, y = NULL
  ) +
  theme_eda() +
  theme(
    panel.grid.major = element_blank(),    # no gridlines behind the tiles
    axis.text.x      = element_text(angle = 45, hjust = 1, vjust = 1),
    legend.position  = "right",            # this plot needs its colour key
    legend.title     = element_text(colour = "#52514e", size = 10),
    legend.key.height = unit(1.2, "cm")
  )

ggsave(file.path(output_dir, "plot3_audio_correlations.png"),
       plot3, width = 9, height = 7, dpi = 300, bg = "white")
cat("Saved plot3_audio_correlations.png\n")


# -----------------------------------------------------------------------------
# SECTION 10 - PLOT 6: SIMILAR-SONGS SCORE BY EMOTION
# -----------------------------------------------------------------------------

# Plot 6: mean similarity score by emotion. The score comes from the
# semi-structured "Similar Songs" field (nested JSON, flattened in Stage 1).
similarity_summary <- run_sql("similarity_summary_by_emotion")
write.csv(similarity_summary, file.path(output_dir, "eda_similarity_by_emotion.csv"), row.names = FALSE)
print(similarity_summary)

similarity_data <- run_sql("similarity_by_emotion") %>%
  mutate(emotion = factor(emotion, levels = rev(emotion_counts$emotion)))   # same order as Plots 1-2

# Almost every score is between 0.90 and 1, so the axis is zoomed in.
# coord_cartesian() only zooms the view.
# Every song still counts in the boxes, medians and outlier totals. 
# Songs below 0.90 are noted in the caption.
zoom_min  <- 0.90
n_below   <- sum(similarity_summary$below_090)

# Work out each emotion's median and outlier limit (Q1 - 1.5 x IQR),
# to label the medians and count the outliers.
similarity_stats <- similarity_data %>%
  group_by(emotion) %>%
  summarise(
    median = median(mean_similarity),
    q1     = quantile(mean_similarity, 0.25),
    q3     = quantile(mean_similarity, 0.75),
    n      = n(),
    .groups = "drop"
  ) %>%
  mutate(lower_limit = q1 - 1.5 * (q3 - q1))

outlier_counts <- similarity_data %>%
  inner_join(similarity_stats, by = "emotion") %>%
  group_by(emotion) %>%
  summarise(n_outliers = sum(mean_similarity < lower_limit),
            n_hidden   = sum(mean_similarity < zoom_min),   # too low to draw on this axis
            n          = first(n), .groups = "drop") %>%
  mutate(label = paste0(comma(n_outliers), " outliers (",
                        sprintf("%.1f", 100 * n_outliers / n), "%)",
                        # Say how many of them are off the chart, if any.
                        ifelse(n_hidden > 0,
                               paste0(", incl. ", comma(n_hidden), " below ",
                                      sprintf("%.2f", zoom_min), " not shown"),
                               "")))
print(outlier_counts)

plot6 <- ggplot(similarity_data, aes(x = mean_similarity, y = emotion, fill = emotion)) +
  # Outliers drawn small and faint grey, so thousands of them don't merge
  # into a solid black bar. Same grey for every emotion.
  geom_boxplot(width = 0.5, colour = "#52514e", linewidth = 0.4,
               outlier.colour = "#52514e", outlier.size = 0.9,
               outlier.alpha = 0.15, outlier.stroke = 0) +
  # 3 decimals here.
  # Scores are stored with full precision and the differences between emotions are small.
  geom_text(data = similarity_stats,
            aes(x = median, y = emotion, label = sprintf("%.3f", median)),
            nudge_y = 0.38, size = 3.3, colour = "#0b0b0b", inherit.aes = FALSE) +
  
  # Check left edge emotion outlier.
  geom_text(data = outlier_counts,
            aes(x = zoom_min, y = emotion, label = label),
            nudge_y = 0.38, hjust = 0, size = 3, colour = "#52514e", inherit.aes = FALSE) +
  scale_fill_manual(values = emotion_colours) +
  scale_x_continuous(breaks = seq(zoom_min, 1, 0.02),
                     expand = expansion(mult = c(0, 0.01))) +   # start exactly at 0.90
  coord_cartesian(xlim = c(zoom_min, 1)) +
  labs(
    title    = "Similarity is high for every emotion, but highest and most consistent for surprise",
    subtitle = paste0("Mean similarity per emotion label (n = ",
                      comma(nrow(similarity_data)), "); labels show medians"),
    x        = "Mean similarity score of the 3 similar songs (0-1)",
    y        = NULL,
    caption  = paste0("Grey dots = outliers (below Q1 - 1.5 x IQR). Axis starts at ", sprintf("%.2f", zoom_min), ":\n",
                      comma(n_below), " songs below it are off the chart but counted in the boxes and outlier totals.")
  ) +
  theme_eda() +
  theme(
    panel.grid.major.y = element_blank(),
    plot.caption       = element_text(colour = "#52514e", size = 9, hjust = 0),
    plot.caption.position = "plot"
  )

ggsave(file.path(output_dir, "plot6_similarity_by_emotion.png"),
       plot6, width = 9, height = 5, dpi = 300, bg = "white")
cat("Saved plot6_similarity_by_emotion.png\n")
