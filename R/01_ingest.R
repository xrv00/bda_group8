# 01_ingest.R
# Owner: Harvey Maxwell Henry (Data Engineer)
# Purpose: Download data, parse JSONL, clean, write parquet and SQLite

# TODO: paste code here

# =====================================================================
# 01_ingest_clean.R  -  Stage 1: Ingestion, cleaning and SQL storage
# KD44103 Big Data Analytics | Owner: Harvey Maxwell Henry (Data Engineer)
#
# Run from the project root:   Rscript scripts/01_ingest_clean.R
#
# Produces
#   data/clean/spotify_clean.parquet/      -> Yi Qing (feature engineering)
#   data/clean/spotify_structured.sqlite   -> Chia Yin (EDA)
#   logs/ingestion_log.txt, logs/step_row_counts.csv
#   docs/data_dictionary.csv / .md, docs/imputation_medians.csv,
#   docs/iqr_bounds.csv
#
# Nothing is ever dropped: duplicates, outliers and invalid labels are
# FLAGGED so the row count stays at 551,443 end to end.
# =====================================================================

suppressPackageStartupMessages({
  library(sparklyr)
  library(dplyr)
  library(DBI)
  library(RSQLite)
})

# ---------------------------------------------------------------------
# 0. CONFIG  (edit PROJECT_DIR and DATA_DIR if your folders are different)
# ---------------------------------------------------------------------
PROJECT_DIR <- "C:/Users/60145/Desktop/BDA/BDA_Stage1_R"
DATA_DIR    <- file.path(PROJECT_DIR, "dataset")   # folder holding the JSON file
if (!dir.exists(PROJECT_DIR)) stop("PROJECT_DIR does not exist: ", PROJECT_DIR)
setwd(PROJECT_DIR)
cat("Working directory:", getwd(), "\n")

CFG <- list(
  kaggle_slug   = "devdope/900k-spotify",
  raw_file      = NULL,            # leave NULL to auto-detect inside DATA_DIR
  spark_version = "3.5.9",         # must match the version you installed
  driver_memory = "4G",            # use about HALF of your laptop RAM (8 GB RAM -> "4G")
  cores         = 2,               # CPU cores for Spark; more cores = more memory used
  top_n_genres  = 12,
  expected_rows = 551443,
  emotions      = c("joy", "sadness", "anger", "fear", "love", "surprise"),
  parquet_parts = 16,
  sample_rows   = NULL             # NULL = FULL dataset. Use 20000 for a quick test run
)

`%||%` <- function(a, b) if (is.null(a)) b else a
P <- function(...) gsub("\\\\", "/", normalizePath(file.path(...), mustWork = FALSE))

for (d in c("data/raw", "data/interim", "data/clean", "logs", "docs"))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

PATH <- list(
  interim = P("data/interim/stage_ids.parquet"),
  clean   = P("data/clean/spotify_clean.parquet"),
  sqlite  = P("data/clean/spotify_structured.sqlite"),
  log     = P("logs/ingestion_log.txt"),
  steps   = P("logs/step_row_counts.csv")
)
if (file.exists(PATH$log)) file.remove(PATH$log)

# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
log_msg <- function(...) {
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste0(...))
  cat(line, "\n")
  write(line, PATH$log, append = TRUE)
}
step_log <- data.frame(step = character(), rows = numeric(), cols = numeric())
log_step <- function(step, df) {
  n <- sdf_nrow(df); p <- sdf_ncol(df)
  log_msg(sprintf("[%s] rows=%s cols=%s", step, format(n, big.mark = ","), p))
  step_log <<- rbind(step_log, data.frame(step = step, rows = n, cols = p))
  write.csv(step_log, PATH$steps, row.names = FALSE)
  invisible(n)
}

q <- function(x) paste0("`", x, "`")                      # quote SQL identifier
run_sql <- function(sql, name) sdf_register(sdf_sql(sc, sql), name)

# ---------------------------------------------------------------------
# 1. DOWNLOAD (scripted)  - needs the Kaggle CLI + ~/.kaggle/kaggle.json
# ---------------------------------------------------------------------
find_raw <- function() {
  dirs <- c(DATA_DIR, "data/raw")
  dirs <- dirs[dir.exists(dirs)]
  f <- unlist(lapply(dirs, list.files, recursive = TRUE, full.names = TRUE))
  f <- f[!file.info(f)$isdir]
  f <- f[grepl("\\.(json|jsonl)$", f, ignore.case = TRUE) |
         grepl("BERT", basename(f), ignore.case = TRUE)]   # also catches no-extension
  f <- f[!grepl("\\.(zip|csv)$", f, ignore.case = TRUE)]
  if (length(f) > 1) f <- c(f[grepl("BERT", f)], f)       # prefer the BERT file
  if (length(f)) P(f[1]) else character()
}
raw_path <- CFG$raw_file %||% find_raw()
if (length(raw_path) == 0) {
  log_msg("Raw file not found, downloading from Kaggle: ", CFG$kaggle_slug)
  rc <- tryCatch(suppressWarnings(system2("kaggle", c("datasets", "download", "-d",
                   CFG$kaggle_slug, "-p", "data/raw", "--unzip"))),
                 error = function(e) 1L)
  if (rc != 0) stop("No data file found in ", DATA_DIR, " and the Kaggle CLI is not set up. ",
                    "Download the dataset from Kaggle manually and put the .json ",
                    "file in ", DATA_DIR)
  raw_path <- find_raw()
}
if (length(raw_path) == 0) stop("No .json/.jsonl file found in data/raw/")
log_msg("Raw file: ", raw_path, " (",
        round(file.size(raw_path) / 1024^2, 1), " MB)")

# ---------------------------------------------------------------------
# 2. SPARK SESSION
# ---------------------------------------------------------------------
conf <- spark_config()
conf$`sparklyr.shell.driver-memory`  <- CFG$driver_memory
conf$spark.sql.ansi.enabled          <- "false"      # bad casts -> NULL, not error
conf$spark.sql.legacy.timeParserPolicy <- "CORRECTED"
conf$spark.sql.session.timeZone      <- "UTC"
conf$spark.sql.shuffle.partitions    <- "32"
conf$spark.sql.files.maxPartitionBytes <- "33554432"   # 32 MB read splits (lyrics rows are big)
conf$spark.hadoop.parquet.block.size   <- "33554432"   # 32 MB parquet row groups = less RAM
conf$spark.driver.maxResultSize      <- "1g"
spark_disconnect_all()   # close any session opened earlier so the settings below apply
sc <- spark_connect(master = sprintf("local[%d]", CFG$cores), version = CFG$spark_version, config = conf)
# Set again at runtime so they are guaranteed active (bad dates/casts -> NULL, no crash)
DBI::dbExecute(sc, "SET spark.sql.legacy.timeParserPolicy=CORRECTED")
DBI::dbExecute(sc, "SET spark.sql.ansi.enabled=false")
log_msg("Spark ", spark_version(sc), " connected (driver memory ",
        CFG$driver_memory, ", cores ", CFG$cores, ")")

# ---------------------------------------------------------------------
# 3. READ JSONL AND INSPECT SCHEMA
# ---------------------------------------------------------------------
raw <- spark_read_json(sc, "raw_json", raw_path, memory = FALSE,
                       options = list(samplingRatio = 0.2))   # infer schema from 20% = faster
if ("_corrupt_record" %in% colnames(raw)) {
  log_msg("Not line-delimited JSON, retrying with multiLine=TRUE")
  raw <- spark_read_json(sc, "raw_json", raw_path, memory = FALSE,
                         options = list(multiLine = TRUE))
}
n_raw <- log_step("03_raw_loaded", raw)
if (n_raw != CFG$expected_rows)
  log_msg("WARNING: expected ", CFG$expected_rows, " rows, got ", n_raw)

sch   <- sdf_schema(raw)
types <- vapply(sch, function(z) z$type, character(1))
names(types) <- vapply(sch, function(z) z$name, character(1))
log_msg("Raw schema (", length(types), " columns):")
for (nm in names(types)) log_msg("   ", nm, " : ", types[[nm]])

# snake_case every column name (keeps the original -> new mapping)
clean_name <- function(x) gsub("^_+|_+$", "", gsub("[^a-z0-9]+", "_", tolower(x)))
old_names <- names(types)
new_names <- make.unique(clean_name(old_names), sep = "_")
raw_clean <- run_sql(
  sprintf("SELECT %s FROM raw_json",
          paste(sprintf("%s AS %s", q(old_names), q(new_names)), collapse = ", ")),
  "raw_clean")
names(types) <- new_names
cols <- new_names

# ---------------------------------------------------------------------
# 4. DETECT COLUMN ROLES  (so the script survives small name differences)
# ---------------------------------------------------------------------
pick1 <- function(patterns) {
  for (p in patterns) { m <- grep(p, cols, value = TRUE); if (length(m)) return(m[1]) }
  NA_character_
}
COL <- list(
  artist  = pick1("^artist"),
  song    = pick1(c("^song$", "^song_", "^track", "^title")),
  lyrics  = pick1(c("^text$", "^lyrics")),
  emotion = pick1("^emotion$"),
  genre   = pick1("^genre"),
  album   = pick1("^album"),
  release = pick1("release"),
  length  = pick1("^length"),
  key     = pick1("^key$"),
  timesig = pick1("^time_signature")
)
need <- c("artist", "song", "lyrics", "emotion")
if (any(is.na(COL[need])))
  stop("Could not find columns for: ", paste(need[is.na(COL[need])], collapse = ", "),
       ". Check the schema printed above and edit COL.")
log_msg("Detected roles: ",
        paste(names(COL), unlist(lapply(COL, as.character)), sep = "=", collapse = "; "))

AUD_PAT <- list(energy = "^energy$", valence = "^(valence|positiveness)$",
                danceability = "^danceability$", tempo = "^tempo",
                loudness = "^loudness", speechiness = "^speechiness$",
                liveness = "^liveness$", acousticness = "^acousticness$",
                instrumentalness = "^instrumentalness$", popularity = "^popularity$")
AUD <- Filter(Negate(is.na), lapply(AUD_PAT, pick1))      # target -> source col
log_msg("Audio features: ", paste(names(AUD), AUD, sep = "<-", collapse = ", "))

# --- Similar songs: handles nested array<struct>, struct, or flat columns
detect_similar <- function() {
  nested <- cols[grepl("similar", cols) &
                   grepl("Array|Struct|array<|struct<", types[cols])]
  if (length(nested)) {
    nc <- nested[1]; t <- types[[nc]]
    flds <- if (grepl("struct<", t)) {
      gsub("^[<,]|:$", "", regmatches(t, gregexpr("[<,]([^:<>,]+):", t))[[1]])
    } else {
      gsub("^StructField\\(|,$", "", regmatches(t, gregexpr("StructField\\(([^,()]+),", t))[[1]])
    }
    log_msg("Nested similar-songs column '", nc, "' with fields: ",
            paste(flds, collapse = " | "))
    if (grepl("^ArrayType|^array<", t)) {
      f_sc <- grep("score", flds, ignore.case = TRUE, value = TRUE)[1]
      f_ar <- grep("artist", flds, ignore.case = TRUE, value = TRUE)[1]
      f_so <- grep("song|track|title", flds, ignore.case = TRUE, value = TRUE)[1]
      ex <- function(f) if (is.na(f)) character() else
        sprintf("%s[%d].%s", q(nc), 0:2, q(f))
      # artist/song names: element i may hold "Similar Artist/Song 1..3", so take the
      # first non-null of those fields for element i
      f_ar_all <- grep("artist", flds, ignore.case = TRUE, value = TRUE)
      f_so_all <- grep("song|track|title", flds, ignore.case = TRUE, value = TRUE)
      ex_co <- function(fs) if (!length(fs)) character() else
        vapply(0:2, function(i) sprintf("coalesce(%s)",
               paste(sprintf("%s[%d].%s", q(nc), i, q(fs)), collapse = ", ")), character(1))
      return(list(score = ex(f_sc), artist = ex_co(f_ar_all), song = ex_co(f_so_all),
                  score_numeric = FALSE))
    }
    ex <- function(pat) {
      f <- grep(pat, flds, ignore.case = TRUE, value = TRUE)
      f <- head(f[order(f)], 3)
      sprintf("%s.%s", q(nc), q(f))
    }
    return(list(score = ex("score"), artist = ex("artist"), song = ex("song"),
                score_numeric = FALSE))
  }
  by_idx <- function(pat) {
    m <- grep(pat, cols, value = TRUE); m <- m[grepl("[0-9]$", m)]
    m[order(as.integer(sub("^.*[^0-9]([0-9]+)$", "\\1", m)))]
  }
  sc_cols <- by_idx("(similarity|score)")
  list(score = q(head(sc_cols, 3)),
       artist = q(head(by_idx("similar.*artist"), 3)),
       song = q(head(by_idx("similar.*song"), 3)),
       score_numeric = length(sc_cols) > 0 && all(vapply(head(sc_cols, 3), function(cn)
         grepl("^(Double|Float|Long|Integer|Decimal)", types[[cn]]), logical(1))))
}
SIM <- detect_similar()
SIM$score <- head(SIM$score, 3); SIM$artist <- head(SIM$artist, 3); SIM$song <- head(SIM$song, 3)
if (length(SIM$score) == 0)
  stop("Could not locate the similarity scores. Check the schema in the log.")
log_msg("Similarity score expressions: ", paste(SIM$score, collapse = " , "))

# ---------------------------------------------------------------------
# 5. FLATTEN + CAST TYPES (one SQL pass)
# ---------------------------------------------------------------------
is_num <- function(cn) grepl("^(Double|Float|Long|Integer|Short|Byte|Decimal)", types[[cn]])
num_expr <- function(expr, numeric_type) {
  if (numeric_type) sprintf("CAST(%s AS DOUBLE)", expr) else
    sprintf("CAST(regexp_extract(CAST(%s AS STRING), '(-?[0-9]*[.]?[0-9]+)', 1) AS DOUBLE)", expr)
}
flag01 <- function(x) sprintf(
  "CAST(CASE WHEN lower(CAST(%1$s AS STRING)) IN ('yes','true','1','1.0') THEN 1 WHEN lower(CAST(%1$s AS STRING)) IN ('no','false','0','0.0') THEN 0 END AS INT)",
  q(x))
txt <- function(x) sprintf("nullif(trim(CAST(%s AS STRING)), '')", q(x))
as_ <- function(expr, alias) sprintf("%s AS %s", expr, q(alias))

sel <- c(
  as_(txt(COL$artist), "artist"),
  as_(sprintf("trim(split(CAST(%s AS STRING), ',')[0])", q(COL$artist)), "first_artist"),
  as_(txt(COL$song), "song"),
  as_(sprintf("CASE WHEN lower(%1$s) = 'angry' THEN 'anger' ELSE lower(%1$s) END", txt(COL$emotion)), "emotion"),
  as_(txt(COL$lyrics), "lyrics")
)
used <- c(COL$artist, COL$song, COL$emotion, COL$lyrics)

if (!is.na(COL$album))  { sel <- c(sel, as_(txt(COL$album), "album"));  used <- c(used, COL$album) }
if (!is.na(COL$genre))  { sel <- c(sel, as_(sprintf("lower(%s)", txt(COL$genre)), "genre")); used <- c(used, COL$genre) }
if (!is.na(COL$key))    { sel <- c(sel, as_(txt(COL$key), "key")); used <- c(used, COL$key) }
if (!is.na(COL$timesig)){ sel <- c(sel, as_(txt(COL$timesig), "time_signature")); used <- c(used, COL$timesig) }

# Length "mm:ss" -> seconds
if (!is.na(COL$length)) {
  L <- COL$length
  sel <- c(sel, as_(sprintf(
    "CASE WHEN CAST(%1$s AS STRING) LIKE '%%:%%' THEN CAST(split(CAST(%1$s AS STRING), ':')[0] AS INT) * 60 + CAST(split(CAST(%1$s AS STRING), ':')[1] AS INT) ELSE %2$s END",
    q(L), num_expr(q(L), is_num(L))), "length_sec"))
  used <- c(used, L)
}
# Release date -> DATE (+ year)
if (!is.na(COL$release)) {
  rd <- sprintf("regexp_replace(CAST(%s AS STRING), '(?<=[0-9])(st|nd|rd|th)', '')", q(COL$release))
  fmts <- c("yyyy-MM-dd", "d MMMM yyyy", "d MMM yyyy", "MMMM d yyyy", "MMM d yyyy",
            "MMMM d, yyyy", "MM/dd/yyyy", "dd/MM/yyyy", "yyyy")
  sel <- c(sel, as_(sprintf("coalesce(%s)",
          paste(sprintf("to_date(%s, '%s')", rd, fmts), collapse = ", ")), "release_date"))
  used <- c(used, COL$release)
}
# Audio features (Loudness "-5.2 db" -> -5.2, Tempo -> double, ...)
for (tg in names(AUD)) {
  src <- AUD[[tg]]
  sel <- c(sel, as_(num_expr(q(src), is_num(src)), tg)); used <- c(used, src)
}
# Explicit + "Good for ..." context tags -> 0/1
if (any(cols == "explicit")) { sel <- c(sel, as_(flag01("explicit"), "explicit")); used <- c(used, "explicit") }
for (g in grep("^good_for", cols, value = TRUE)) {
  sel <- c(sel, as_(flag01(g), g)); used <- c(used, g)
}
# Similar songs: flatten to sim_score_i / sim_artist_i / sim_song_i
for (i in seq_along(SIM$score))
  sel <- c(sel, as_(num_expr(SIM$score[i], SIM$score_numeric), sprintf("sim_score_%d", i)))
for (i in seq_along(SIM$artist))
  sel <- c(sel, as_(sprintf("CAST(%s AS STRING)", SIM$artist[i]), sprintf("sim_artist_%d", i)))
for (i in seq_along(SIM$song))
  sel <- c(sel, as_(sprintf("CAST(%s AS STRING)", SIM$song[i]), sprintf("sim_song_%d", i)))

unused <- setdiff(cols, used)
log_msg("Source columns NOT carried into the clean table: ",
        if (length(unused)) paste(unused, collapse = ", ") else "(none)")

lim <- if (is.null(CFG$sample_rows)) "" else sprintf("\nLIMIT %d", as.integer(CFG$sample_rows))
if (nzchar(lim)) log_msg("*** TEST RUN: only the first ", CFG$sample_rows, " rows are used ***")
df_a <- run_sql(paste0("SELECT\n  ", paste(sel, collapse = ",\n  "), "\nFROM raw_clean", lim), "stage_a")
n_a <- log_step("05_flattened_and_cast", df_a)
sim_peek <- df_a %>% select(starts_with("sim_")) %>% head(3) %>% collect()
log_msg("Sample of flattened similar-song columns:")
for (ln in capture.output(print(as.data.frame(sim_peek), right = FALSE))) log_msg("   ", ln)

# ---------------------------------------------------------------------
# 6. row_id (unique, materialised to parquet so it can never change)
# ---------------------------------------------------------------------
# monotonically_increasing_id is cheap (the old sequential id forced extra full passes)
df_b <- run_sql("SELECT monotonically_increasing_id() AS row_id, * FROM stage_a", "stage_b")
spark_write_parquet(df_b, PATH$interim, mode = "overwrite")
df_c <- spark_read_parquet(sc, "stage_c", PATH$interim, memory = FALSE)
log_step("06_row_id_added", df_c)

# dup_flag: same artist + song (case-insensitive). Later copies are flagged.
run_sql(sprintf(
"SELECT row_id,
   CASE WHEN artist IS NULL OR song IS NULL THEN 0
        WHEN ROW_NUMBER() OVER (PARTITION BY lower(artist), lower(song) ORDER BY row_id) > 1 THEN 1
        ELSE 0 END AS dup_flag
 FROM stage_c"), "dupflags")

# mean_similarity = average of the available similarity scores
score_cols <- sprintf("sim_score_%d", seq_along(SIM$score))
mean_expr <- sprintf("(%s) / nullif(%s, 0) AS mean_similarity",
  paste(sprintf("coalesce(%s, 0)", score_cols), collapse = " + "),
  paste(sprintf("CASE WHEN %s IS NULL THEN 0 ELSE 1 END", score_cols), collapse = " + "))
year_expr <- if ("release_date" %in% colnames(df_c)) "year(b.release_date) AS release_year, " else ""
df_d <- run_sql(sprintf(
  "SELECT b.*, d.dup_flag, %s%s FROM stage_c b JOIN dupflags d ON b.row_id = d.row_id",
  year_expr, mean_expr), "stage_d")
n_dup <- df_d %>% filter(dup_flag == 1) %>% sdf_nrow()
log_step("06_dup_flag_and_mean_similarity", df_d)
log_msg(sprintf("Duplicates flagged (artist+song): %s of %s (%.2f%%). NOT dropped.",
                format(n_dup, big.mark = ","), format(n_a, big.mark = ","),
                100 * n_dup / n_a))

# ---------------------------------------------------------------------
# 7. MISSING VALUES: report, then median-impute numeric columns
# ---------------------------------------------------------------------
null_counts <- function(view, df) {
  cn <- colnames(df)
  r <- collect(sdf_sql(sc, sprintf("SELECT %s FROM %s",
        paste(sprintf("SUM(CASE WHEN %s IS NULL THEN 1 ELSE 0 END) AS %s", q(cn), q(cn)),
              collapse = ", "), view)))
  data.frame(column = names(r), n_missing = as.numeric(unlist(r[1, ])))
}
miss_before <- null_counts("stage_d", df_d)
miss_before$pct <- round(100 * miss_before$n_missing / n_a, 2)
log_msg("Missing values BEFORE imputation (columns with >0):")
for (i in which(miss_before$n_missing > 0))
  log_msg(sprintf("   %-22s %9s (%.2f%%)", miss_before$column[i],
                  format(miss_before$n_missing[i], big.mark = ","), miss_before$pct[i]))

n_bad_emotion <- df_d %>% filter(is.na(emotion) | !(emotion %in% !!CFG$emotions)) %>% sdf_nrow()
log_msg("Rows with missing or unexpected emotion label (kept, excluded from modelling): ",
        n_bad_emotion)

imp_cols <- intersect(c(names(AUD), "length_sec", "mean_similarity"), colnames(df_d))
med <- vapply(imp_cols, function(cn) {
  unname(sdf_quantile(df_d, cn, probabilities = 0.5, relative.error = 0.001)[1])
}, numeric(1))
if (anyNA(med)) stop("Median is NA for: ", paste(names(med)[is.na(med)], collapse = ", "))
write.csv(data.frame(column = names(med), median = med),
          "docs/imputation_medians.csv", row.names = FALSE)
log_msg("Imputation medians: ", paste(sprintf("%s=%.4g", names(med), med), collapse = ", "))

cn_d <- colnames(df_d)
sel_i <- vapply(cn_d, function(x) {
  if (x %in% imp_cols) sprintf("coalesce(%s, %s) AS %s", q(x), sprintf("%.15g", med[[x]]), q(x)) else q(x)
}, character(1))
df_e <- run_sql(sprintf("SELECT %s, CASE WHEN %s THEN 1 ELSE 0 END AS imputed_any FROM stage_d",
    paste(sel_i, collapse = ", "),
    paste(sprintf("%s IS NULL", q(imp_cols)), collapse = " OR ")), "stage_e")
log_step("07_imputed", df_e)

# ---------------------------------------------------------------------
# 8. IQR OUTLIER FLAGS  +  9. GENRE GROUPING  +  use_for_model
# ---------------------------------------------------------------------
feat <- intersect(names(AUD), colnames(df_e))
bounds <- do.call(rbind, lapply(feat, function(cn) {
  qs <- unname(sdf_quantile(df_e, cn, probabilities = c(0.25, 0.75), relative.error = 0.001))
  iqr <- qs[2] - qs[1]
  data.frame(feature = cn, q1 = qs[1], q3 = qs[2], iqr = iqr,
             lower = qs[1] - 1.5 * iqr, upper = qs[2] + 1.5 * iqr)
}))
write.csv(bounds, "docs/iqr_bounds.csv", row.names = FALSE)
zero_iqr <- bounds$feature[bounds$iqr <= 0]
if (length(zero_iqr)) log_msg("IQR = 0 for: ", paste(zero_iqr, collapse = ", "),
                              " -> IQR rule not meaningful, no outlier flag created")
bounds <- bounds[bounds$iqr > 0, ]
out_exprs <- sprintf("CASE WHEN %1$s < %2$.15g OR %1$s > %3$.15g THEN 1 ELSE 0 END AS %4$s",
                     q(bounds$feature), bounds$lower, bounds$upper, q(paste0("out_", bounds$feature)))

if (!is.na(COL$genre)) {
  # the source "genre" is a comma list ("hip hop,trap,cloud rap"); group on the first genre
  pg_expr <- "regexp_replace(lower(trim(split(genre, ',')[0])), '[-_]', ' ')"
  top_g <- collect(sdf_sql(sc, sprintf(
    "SELECT %s AS g, COUNT(*) AS n FROM stage_e WHERE genre IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT %d",
    pg_expr, as.integer(CFG$top_n_genres))))
  glist <- paste0("'", gsub("'", "''", top_g$g), "'", collapse = ", ")
  genre_expr <- sprintf(
    "%1$s AS genre_primary, CASE WHEN %1$s IN (%2$s) THEN %1$s ELSE 'other' END AS genre_group",
    pg_expr, glist)
} else genre_expr <- "CAST(NULL AS STRING) AS genre_primary, 'other' AS genre_group"

use_expr <- sprintf("CASE WHEN dup_flag = 0 AND emotion IN (%s) THEN 1 ELSE 0 END AS use_for_model",
                    paste0("'", CFG$emotions, "'", collapse = ", "))
cnt_out <- paste(q(paste0("out_", bounds$feature)), collapse = " + ")
tmp <- run_sql(sprintf("SELECT *, %s FROM stage_e", paste(out_exprs, collapse = ", ")), "stage_f1")
df_f <- run_sql(sprintf("SELECT *, (%s) AS outlier_count, %s, %s FROM stage_f1",
                        cnt_out, genre_expr, use_expr), "stage_f")

# final column order: row_id first, lyrics last
final_cols <- c("row_id", setdiff(colnames(df_f), c("row_id", "lyrics")), "lyrics")
df_final <- df_f %>% select(all_of(final_cols))
log_step("08_09_flags_and_genre_group", df_final)

for (i in seq_len(nrow(bounds))) {
  n_o <- df_final %>% filter(!!sym(paste0("out_", bounds$feature[i])) == 1) %>% sdf_nrow()
  log_msg(sprintf("IQR outliers  %-18s %9s  [%.4g, %.4g]", bounds$feature[i],
                  format(n_o, big.mark = ","), bounds$lower[i], bounds$upper[i]))
}
if (!is.na(COL$genre)) {
  log_msg("Genre groups (top ", CFG$top_n_genres, " + other):")
  gg <- df_final %>% count(genre_group, sort = TRUE) %>% collect()
  for (i in seq_len(nrow(gg))) log_msg(sprintf("   %-28s %s", gg$genre_group[i], format(gg$n[i], big.mark = ",")))
}
log_msg("Emotion distribution (all rows):")
ee <- df_final %>% count(emotion, sort = TRUE) %>% collect()
for (i in seq_len(nrow(ee))) log_msg(sprintf("   %-12s %s", ee$emotion[i], format(ee$n[i], big.mark = ",")))
n_model <- df_final %>% filter(use_for_model == 1) %>% sdf_nrow()
log_msg("Rows with use_for_model = 1: ", format(n_model, big.mark = ","))

# ---------------------------------------------------------------------
# 10. WRITE spotify_clean.parquet  (a folder of part files) and verify
# ---------------------------------------------------------------------
df_final %>% sdf_coalesce(CFG$parquet_parts) %>%
  spark_write_parquet(PATH$clean, mode = "overwrite")
chk <- spark_read_parquet(sc, "clean_chk", PATH$clean, memory = FALSE)
n_clean <- log_step("10_parquet_written", chk)
if (n_clean != n_a) stop("Row count changed between flatten step and clean parquet!")
log_msg("Parquet verified: row count matches (", n_clean, ")")

# ---------------------------------------------------------------------
# 11. SQLite: structured fields only (no lyrics, no similar names)
# ---------------------------------------------------------------------
sch_f <- sdf_schema(chk)
types_f <- vapply(sch_f, function(z) z$type, character(1))
names(types_f) <- vapply(sch_f, function(z) z$name, character(1))
nested_f <- names(types_f)[grepl("Array|Struct|Map|array<|struct<|map<", types_f)]
sql_cols <- setdiff(names(types_f),
  c("lyrics", grep("^sim_(artist|song)_", names(types_f), value = TRUE), nested_f))

tab <- chk %>% select(all_of(sql_cols)) %>% collect()
for (cn in names(tab)) if (inherits(tab[[cn]], "Date")) tab[[cn]] <- as.character(tab[[cn]])

sqlite_type <- function(x) if (is.integer(x) || is.logical(x)) "INTEGER" else
  if (is.numeric(x)) "REAL" else "TEXT"
col_defs <- vapply(names(tab), function(cn)
  if (cn == "row_id") "\"row_id\" INTEGER NOT NULL UNIQUE"
  else sprintf("\"%s\" %s", cn, sqlite_type(tab[[cn]])), character(1))

if (file.exists(PATH$sqlite)) file.remove(PATH$sqlite)
con <- dbConnect(RSQLite::SQLite(), PATH$sqlite)
dbExecute(con, sprintf("CREATE TABLE songs (%s)", paste(col_defs, collapse = ", ")))
chunks <- split(seq_len(nrow(tab)), ceiling(seq_len(nrow(tab)) / 100000))
for (ix in chunks) dbWithTransaction(con, dbAppendTable(con, "songs", tab[ix, , drop = FALSE]))

for (cn in intersect(c("emotion", "genre_group", "dup_flag", "use_for_model"), names(tab)))
  dbExecute(con, sprintf("CREATE INDEX idx_songs_%1$s ON songs(%1$s)", cn))

dbExecute(con, sprintf(
  "CREATE VIEW model_ready AS SELECT * FROM songs WHERE dup_flag = 0 AND emotion IN (%s)",
  paste0("'", CFG$emotions, "'", collapse = ", ")))

n_tab  <- dbGetQuery(con, "SELECT COUNT(*) n FROM songs")$n
n_view <- dbGetQuery(con, "SELECT COUNT(*) n FROM model_ready")$n
log_msg(sprintf("SQLite: songs=%s rows, model_ready view=%s rows, columns=%d",
                format(n_tab, big.mark = ","), format(n_view, big.mark = ","), ncol(tab)))
log_msg("SQLite integrity_check: ", dbGetQuery(con, "PRAGMA integrity_check")[[1]])
if (n_tab != n_clean) stop("SQLite row count does not match parquet!")
if (n_view != n_model) log_msg("WARNING: view rows differ from use_for_model count")
log_msg("SQLite emotion distribution in model_ready:")
em <- dbGetQuery(con, "SELECT emotion, COUNT(*) n FROM model_ready GROUP BY emotion ORDER BY n DESC")
for (i in seq_len(nrow(em))) log_msg(sprintf("   %-12s %s", em$emotion[i], format(em$n[i], big.mark = ",")))
dbDisconnect(con)

# ---------------------------------------------------------------------
# 12. DATA DICTIONARY
# ---------------------------------------------------------------------
desc <- c(
  row_id = "Unique row ID (not consecutive). Same in parquet and SQLite; use it for joins.",
  artist = "Artist string as given in the source.",
  first_artist = "First listed artist (text before the first comma). Use for the artist-grouped split.",
  song = "Song title.", album = "Album name.", genre = "Original genre (lower-case).",
  genre_primary = "First genre in the source genre list (lower-case, hyphens -> spaces).",
  genre_group = paste0("Top ", CFG$top_n_genres, " primary genres by count; everything else = 'other'."),
  release_date = "Release date parsed to DATE (NULL if unparseable).",
  release_year = "Year of release_date.",
  length_sec = "Song length in seconds (parsed from mm:ss). Median-imputed.",
  key = "Musical key (text).", time_signature = "Time signature (text).",
  explicit = "1 = explicit, 0 = not explicit.",
  emotion = "TARGET label (joy, sadness, anger, fear, love, surprise). Machine-generated from lyrics.",
  lyrics = "Raw lyrics text. Parquet only (not stored in SQLite).",
  dup_flag = "1 = later duplicate of the same artist+song. Flagged, not dropped.",
  imputed_any = "1 = at least one numeric field was median-imputed in this row.",
  mean_similarity = "Mean of the available similarity scores. Median-imputed if none.",
  outlier_count = "Number of audio features flagged as IQR outliers for this row.",
  use_for_model = "1 = dup_flag = 0 AND emotion is one of the six valid classes."
)
describe <- function(cn) {
  if (!is.na(desc[cn])) return(unname(desc[cn]))
  if (cn %in% names(AUD)) return(paste0("Audio feature '", cn, "' cast to double, median-imputed."))
  if (grepl("^out_", cn)) return(paste0("1 = '", sub("^out_", "", cn), "' outside Q1-1.5*IQR / Q3+1.5*IQR. Flag only."))
  if (grepl("^sim_score_", cn)) return("Similarity score of the similar song (flattened from nested JSON).")
  if (grepl("^sim_artist_", cn)) return("Similar artist name (flattened). Parquet only.")
  if (grepl("^sim_song_", cn)) return("Similar song title (flattened). Parquet only.")
  if (grepl("^good_for", cn)) return("Context tag, 1/0.")
  "Source column (cleaned)."
}
dict <- data.frame(column = names(types_f), type = unname(types_f), stringsAsFactors = FALSE)
dict$description <- vapply(dict$column, describe, character(1))
dict$in_sqlite <- dict$column %in% sql_cols
dict$missing_before_imputation <- miss_before$n_missing[match(dict$column, miss_before$column)]
write.csv(dict, "docs/data_dictionary.csv", row.names = FALSE)
md <- c("| Column | Type | In SQLite | Missing (before) | Description |",
        "|---|---|---|---|---|",
        sprintf("| %s | %s | %s | %s | %s |", dict$column, dict$type,
                ifelse(dict$in_sqlite, "yes", "no"),
                ifelse(is.na(dict$missing_before_imputation), "-", dict$missing_before_imputation),
                dict$description))
writeLines(md, "docs/data_dictionary.md")
log_msg("Data dictionary written (", nrow(dict), " columns)")

# ---------------------------------------------------------------------
log_msg("DONE. Hand off: ", PATH$clean, " | ", PATH$sqlite,
        " | logs/ingestion_log.txt | docs/data_dictionary.csv")
spark_disconnect(sc)

