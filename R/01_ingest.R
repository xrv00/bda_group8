# =====================================================================
# 01_ingest.R  -  Stage 1 (Data Engineer): the WHOLE stage in one script
# KD44103 Big Data Analytics | Owner: Harvey Maxwell Henry
#
# One run does everything:
#   1. scripted Kaggle download (falls back to a file placed in /dataset)
#   2. parse the JSONL in Spark (sparklyr) and flatten the nested fields
#   3. clean: type casts, duplicates, missing values, IQR outliers, genres
#   4. flag duplicates two ways (never dropped):
#        dup_flag        = same artist + song title
#        lyrics_dup_flag = identical lyrics under ANY artist (leakage risk)
#   5. write spotify_clean.parquet and spotify_structured.sqlite (+ view)
#   6. build the data dictionary and run the automatic verification checks
#
# TEST FIRST:  set sample_rows = 20000 in CFG (takes a few minutes), then
#              set it back to NULL for the full dataset (30-60 minutes).
#
# Produces
#   data/clean/spotify_clean.parquet/      -> Yi Qing (feature engineering)
#   data/clean/spotify_structured.sqlite   -> Chia Yin (EDA)
#   logs/ingestion_log.txt, logs/step_row_counts.csv
#   docs/data_dictionary.csv/.md, imputation_medians.csv, iqr_bounds.csv,
#   docs/duplicates_summary.csv, stage1_verification.csv,
#   docs/stage1_numbers_for_report.md
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
  audio_percent_features = c("energy", "valence", "danceability", "speechiness",
                             "liveness", "acousticness", "instrumentalness"),
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
  flags   = P("data/interim/row_flags.parquet"),
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
log_count_step <- function(step, rows, cols) {
  log_msg(sprintf("[%s] rows=%s cols=%s", step, format(rows, big.mark = ","), cols))
  step_log <<- rbind(step_log, data.frame(step = step, rows = rows, cols = cols))
  write.csv(step_log, PATH$steps, row.names = FALSE)
  invisible(rows)
}

q <- function(x) paste0("`", x, "`")                      # quote SQL identifier
run_sql <- function(sql, name) sdf_register(sdf_sql(sc, sql), name)

# Windows can refuse Spark's "clear output directory" (folder locked by a leftover
# Java process, Explorer or antivirus). Delete the old folder ourselves first; if it
# is still locked, write to a new time-stamped folder instead of failing.
fresh_path <- function(path) {
  if (dir.exists(path) || file.exists(path)) {
    unlink(path, recursive = TRUE, force = TRUE)
    Sys.sleep(1)
  }
  if (dir.exists(path) || file.exists(path)) {
    alt <- sub("\\.parquet$", paste0("_", format(Sys.time(), "%H%M%S"), ".parquet"), path)
    log_msg("WARNING: could not delete ", path, " (locked by another program?). Using ", alt)
    return(alt)
  }
  path
}

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
  dl <- tryCatch(suppressWarnings(system2("kaggle", c("datasets", "download", "-d",
                                                      CFG$kaggle_slug, "-p", "data/raw", "--unzip"),
                                          stdout = TRUE, stderr = TRUE)),
                 error = function(e) structure(conditionMessage(e), status = 1L))
  rc <- attr(dl, "status") %||% 0L
  if (length(rc) != 1 || is.na(rc)) rc <- 1L
  if (rc != 0) stop("Kaggle download failed (exit ", rc, "). Output: ",
                    paste(tail(as.character(dl), 5), collapse = " | "), ". ",
                    "No data file found in ", DATA_DIR, ". Check the Kaggle CLI and credentials.")
  raw_path <- find_raw()
}
if (length(raw_path) == 0) stop("No .json/.jsonl file found in data/raw/")
if (length(raw_path) != 1 || !file.exists(raw_path))
  stop("Expected one existing raw JSON/JSONL file; found: ", paste(raw_path, collapse = ", "))
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
# Infer the schema from a 20% sample (faster). If this Spark/sparklyr combination
# rejects the option, fall back to reading the whole file for schema inference.
raw <- tryCatch(
  spark_read_json(sc, "raw_json", raw_path, memory = FALSE,
                  options = list(samplingRatio = "0.2")),
  error = function(e) {
    log_msg("samplingRatio option rejected (", substr(conditionMessage(e), 1, 80),
            ") - reading without sampling")
    spark_read_json(sc, "raw_json", raw_path, memory = FALSE)
  })
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
# These roles are required later by the SQLite schema and verification stage.
# Fail here, before Spark transformations, instead of discovering a missing field at the end.
need <- c("artist", "song", "lyrics", "emotion", "genre", "release", "length")
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
required_audio <- names(AUD_PAT)
missing_audio <- setdiff(required_audio, names(AUD))
if (length(missing_audio)) {
  stop("Required Spotify audio feature column(s) not found: ",
       paste(missing_audio, collapse = ", "),
       ". Check the raw schema and update AUD_PAT only if the source uses different names.")
}
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
      score_numeric <- !is.na(f_sc) && grepl(
        "score[^,)]*,(Double|Float|Long|Integer|Decimal)Type", t, ignore.case = TRUE)
      return(list(score = ex(f_sc), artist = ex_co(f_ar_all), song = ex_co(f_so_all),
                  score_numeric = score_numeric))
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
if (length(SIM$score) != 3)
  stop("Expected exactly 3 similarity scores, found ", length(SIM$score),
       ". Check the source schema and similarity-column detection.")
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
  audio_value <- num_expr(q(src), is_num(src))
  # This Kaggle source stores these seven normalized Spotify features on a
  # 0-100 scale. Apply one documented column-level scale, not a per-row guess.
  if (tg %in% CFG$audio_percent_features)
    audio_value <- sprintf("(%s) / 100.0", audio_value)
  sel <- c(sel, as_(audio_value, tg)); used <- c(used, src)
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
# 6. row_id (unique, materialised) + DUPLICATE FLAGS
# ---------------------------------------------------------------------
# monotonically_increasing_id is cheap; writing it to parquet fixes the ids for good
df_b <- run_sql("SELECT monotonically_increasing_id() AS row_id, * FROM stage_a", "stage_b")
PATH$interim <- fresh_path(PATH$interim)
spark_write_parquet(df_b, PATH$interim, mode = "overwrite")
df_c <- spark_read_parquet(sc, "stage_c", PATH$interim, memory = FALSE)
log_step("06_row_id_added", df_c)

# Two kinds of duplicate. Both are FLAGGED, never dropped.
#   dup_flag        : same artist + song title (case-insensitive); later copies = 1
#   lyrics_dup_flag : identical lyrics under ANY artist/title; later copies = 1
#                     among rows with dup_flag = 0 (one representative survives).
#                     Identical lyrics in train AND test would leak the answer.
#   lyrics_group_id : shared by all rows with the same lyrics (for a group split)
flags_sql <- "
WITH b AS (
  SELECT row_id, lower(artist) AS a, lower(song) AS s,
         CASE WHEN lyrics IS NULL THEN NULL
              ELSE sha2(lower(trim(regexp_replace(lyrics, '\\\\s+', ' '))), 256) END AS lk
  FROM stage_c
),
d AS (
  SELECT row_id, lk,
         CASE WHEN a IS NULL OR s IS NULL THEN 0
              WHEN ROW_NUMBER() OVER (PARTITION BY a, s ORDER BY row_id) > 1 THEN 1
              ELSE 0 END AS dup_flag
  FROM b
),
x AS (
  SELECT row_id, lk,
         MIN(row_id) OVER (PARTITION BY lk) AS lyrics_group_id,
         COUNT(*)    OVER (PARTITION BY lk) AS lyrics_group_size
  FROM d WHERE lk IS NOT NULL
)
SELECT d.row_id, d.dup_flag,
       COALESCE(x.lyrics_group_id, d.row_id) AS lyrics_group_id,
       COALESCE(x.lyrics_group_size, 1)      AS lyrics_group_size,
       CASE WHEN x.lk IS NULL THEN 0
            WHEN d.dup_flag = 0 AND
                 ROW_NUMBER() OVER (PARTITION BY x.lk, d.dup_flag ORDER BY d.row_id) > 1 THEN 1
            ELSE 0 END AS lyrics_dup_flag
FROM d LEFT JOIN x ON d.row_id = x.row_id"
log_msg("Computing duplicate flags (artist+song and identical lyrics)...")
PATH$flags <- fresh_path(PATH$flags)
spark_write_parquet(sdf_sql(sc, flags_sql), PATH$flags, mode = "overwrite")
row_flags <- spark_read_parquet(sc, "row_flags", PATH$flags, memory = FALSE)

# mean_similarity = average of the available similarity scores
score_cols <- sprintf("sim_score_%d", seq_along(SIM$score))
mean_expr <- sprintf("(%s) / nullif(%s, 0) AS mean_similarity",
                     paste(sprintf("coalesce(%s, 0)", score_cols), collapse = " + "),
                     paste(sprintf("CASE WHEN %s IS NULL THEN 0 ELSE 1 END", score_cols), collapse = " + "))
year_expr <- if ("release_date" %in% colnames(df_c)) "year(b.release_date) AS release_year, " else ""
df_d <- run_sql(sprintf(
  "SELECT b.*, f.dup_flag, f.lyrics_group_id, f.lyrics_group_size, f.lyrics_dup_flag, %s%s
   FROM stage_c b JOIN row_flags f ON b.row_id = f.row_id",
  year_expr, mean_expr), "stage_d")
log_step("06_dup_flags_and_mean_similarity", df_d)

n_dup <- row_flags %>% filter(dup_flag == 1) %>% sdf_nrow()
n_lyr <- row_flags %>% filter(lyrics_dup_flag == 1) %>% sdf_nrow()
log_msg(sprintf("Duplicates flagged (artist+song): %s of %s (%.2f%%). NOT dropped.",
                format(n_dup, big.mark = ","), format(n_a, big.mark = ","), 100 * n_dup / n_a))
n_lyr_groups <- collect(sdf_sql(sc, "SELECT COUNT(*) AS n FROM row_flags
                  WHERE row_id = lyrics_group_id AND lyrics_group_size > 1"))$n
cross <- collect(sdf_sql(sc, "SELECT COUNT(*) AS n,
    SUM(CASE WHEN lower(a.artist) <> lower(r.artist) THEN 1 ELSE 0 END) AS n_diff
  FROM row_flags f
  JOIN stage_c a ON a.row_id = f.row_id
  JOIN stage_c r ON r.row_id = f.lyrics_group_id
  WHERE f.lyrics_dup_flag = 1"))
log_msg(sprintf("Identical-lyrics duplicates flagged (lyrics_dup_flag): %s (%.2f%%) in %s repeated lyric groups. NOT dropped.",
                format(n_lyr, big.mark = ","), 100 * n_lyr / n_a, format(n_lyr_groups, big.mark = ",")))
log_msg(sprintf("   of which %s have a DIFFERENT artist from the first copy",
                format(if (is.na(cross$n_diff)) 0 else cross$n_diff, big.mark = ",")))
top <- collect(sdf_sql(sc, "SELECT DISTINCT lyrics_group_id, lyrics_group_size FROM row_flags
                            ORDER BY lyrics_group_size DESC LIMIT 10"))
if (nrow(top) && top$lyrics_group_size[1] > 1) {
  pk <- collect(sdf_sql(sc, sprintf(
    "SELECT row_id, artist, song, substr(lyrics, 1, 50) AS lyrics_start FROM stage_c WHERE row_id IN (%s)",
    paste(sprintf("%.0f", top$lyrics_group_id), collapse = ","))))
  pk$copies <- top$lyrics_group_size[match(pk$row_id, top$lyrics_group_id)]
  pk <- pk[order(-pk$copies), ]
  log_msg("Largest identical-lyrics groups (first copy shown):")
  for (i in seq_len(nrow(pk)))
    log_msg(sprintf("   %s copies | %s - %s | %s", format(pk$copies[i], big.mark = ","),
                    pk$artist[i], pk$song[i], gsub("[\r\n]+", " ", pk$lyrics_start[i])))
}

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
log_msg("Missing values BEFORE imputation (all columns):")
for (i in seq_len(nrow(miss_before)))
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
miss_after_imp <- null_counts("stage_e", df_e)
log_msg("Missing values AFTER imputation (all columns):")
for (i in seq_len(nrow(miss_after_imp)))
  log_msg(sprintf("   %-22s %9s", miss_after_imp$column[i],
                  format(miss_after_imp$n_missing[i], big.mark = ",")))

# ---------------------------------------------------------------------
# 8. IQR OUTLIER FLAGS  +  9. GENRE GROUPING  +  use_for_model
# ---------------------------------------------------------------------
# Check continuous numeric features, including length and the flattened
# similarity scores. Binary 0/1 flags are not continuous features.
continuous_features <- c(names(AUD), "length_sec", "mean_similarity",
                         grep("^sim_score_", colnames(df_e), value = TRUE))
feat <- intersect(unique(continuous_features), colnames(df_e))
bounds <- do.call(rbind, lapply(feat, function(cn) {
  qs <- unname(sdf_quantile(df_e, cn, probabilities = c(0.25, 0.75), relative.error = 0.001))
  iqr <- qs[2] - qs[1]
  data.frame(feature = cn, q1 = qs[1], q3 = qs[2], iqr = iqr,
             lower = qs[1] - 1.5 * iqr, upper = qs[2] + 1.5 * iqr)
}))
write.csv(bounds, "docs/iqr_bounds.csv", row.names = FALSE)
# Keep zero-IQR features in the output. Their equal lower/upper fences produce
# a valid flag column instead of silently omitting that feature.
bounds <- bounds[!is.na(bounds$iqr), , drop = FALSE]
out_exprs <- if (nrow(bounds) > 0) {
  sprintf("CASE WHEN %1$s < %2$.15g OR %1$s > %3$.15g THEN 1 ELSE 0 END AS %4$s",
          q(bounds$feature), bounds$lower, bounds$upper, q(paste0("out_", bounds$feature)))
} else character(0)

if (!is.na(COL$genre)) {
  # the source "genre" is a comma list ("hip hop,trap,cloud rap"); group on the first genre
  pg_expr <- "regexp_replace(lower(trim(split(genre, ',')[0])), '[-_]', ' ')"
  top_g <- collect(sdf_sql(sc, sprintf(
    "SELECT %s AS g, COUNT(*) AS n FROM stage_e WHERE genre IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT %d",
    pg_expr, as.integer(CFG$top_n_genres))))
  # An empty genre table must not generate invalid SQL: IN ().
  glist <- if (nrow(top_g) > 0) {
    paste0("'", gsub("'", "''", top_g$g), "'", collapse = ", ")
  } else "'__NO_GENRE_MATCH__'"
  genre_expr <- sprintf(
    "%1$s AS genre_primary, CASE WHEN %1$s IN (%2$s) THEN %1$s ELSE 'other' END AS genre_group",
    pg_expr, glist)
} else genre_expr <- "CAST(NULL AS STRING) AS genre_primary, 'other' AS genre_group"

use_expr <- sprintf("CASE WHEN dup_flag = 0 AND lyrics_dup_flag = 0 AND emotion IN (%s) THEN 1 ELSE 0 END AS use_for_model",
                    paste0("'", CFG$emotions, "'", collapse = ", "))
# If no continuous feature has calculable quartiles, create no out_* columns and
# set outlier_count = 0. Zero-IQR features remain in bounds and get all-zero flags.
if (length(out_exprs) > 0) {
  tmp <- run_sql(sprintf("SELECT *, %s FROM stage_e", paste(out_exprs, collapse = ", ")), "stage_f1")
  cnt_out <- paste(q(paste0("out_", bounds$feature)), collapse = " + ")
} else {
  tmp <- run_sql("SELECT * FROM stage_e", "stage_f1")
  cnt_out <- "CAST(0 AS INT)"
}
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
PATH$clean <- fresh_path(PATH$clean)
df_final %>% sdf_coalesce(CFG$parquet_parts) %>%
  spark_write_parquet(PATH$clean, mode = "overwrite")
chk <- spark_read_parquet(sc, "clean_chk", PATH$clean, memory = FALSE)
n_clean <- log_step("10_parquet_written", chk)
if (n_clean != n_a) stop("Row count changed between flatten step and clean parquet!")
log_msg("Parquet verified: row count matches (", n_clean, ")")

# ---------------------------------------------------------------------
# 11. SQLite: structured fields only, loaded in chunks (keeps RAM low)
# ---------------------------------------------------------------------
sch_f <- sdf_schema(chk)
types_f <- vapply(sch_f, function(z) z$type, character(1))
names(types_f) <- vapply(sch_f, function(z) z$name, character(1))
log_msg("Final Parquet field types:")
for (cn in names(types_f)) log_msg("   ", cn, " : ", types_f[[cn]])
nested_f <- names(types_f)[grepl("Array|Struct|Map|array<|struct<|map<", types_f)]
sql_cols <- setdiff(names(types_f),
                    c("lyrics", grep("^sim_(artist|song)_", names(types_f), value = TRUE), nested_f))

sqlite_type <- function(t) {
  if (grepl("^(Long|Integer|Short|Byte|Boolean)", t)) "INTEGER"
  else if (grepl("^(Double|Float|Decimal)", t)) "REAL"
  else "TEXT"
}
col_defs <- vapply(sql_cols, function(cn)
  if (cn == "row_id") "\"row_id\" INTEGER NOT NULL UNIQUE"
  else sprintf("\"%s\" %s", cn, sqlite_type(types_f[[cn]])), character(1))

if (file.exists(PATH$sqlite)) {
  removed <- file.remove(PATH$sqlite)
  if (!isTRUE(removed) || file.exists(PATH$sqlite))
    stop("Could not remove the previous SQLite output (possibly locked): ", PATH$sqlite)
}
con <- dbConnect(RSQLite::SQLite(), PATH$sqlite)
dbExecute(con, sprintf("CREATE TABLE songs (%s)", paste(col_defs, collapse = ", ")))

N_CHUNKS <- 20
sel_sql <- paste(q(sql_cols), collapse = ", ")
for (k in 0:(N_CHUNKS - 1)) {
  t0 <- Sys.time()
  tab <- collect(sdf_sql(sc, sprintf(
    "SELECT %s FROM clean_chk WHERE pmod(row_id, %d) = %d", sel_sql, N_CHUNKS, k)))
  for (cn in names(tab)) if (inherits(tab[[cn]], "Date")) tab[[cn]] <- as.character(tab[[cn]])
  if (nrow(tab)) dbWithTransaction(con, dbAppendTable(con, "songs", tab))
  log_msg(sprintf("SQLite chunk %d/%d: %s rows (%.0f s)", k + 1, N_CHUNKS,
                  format(nrow(tab), big.mark = ","), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  rm(tab); invisible(gc(verbose = FALSE))
}

for (cn in intersect(c("emotion", "genre_group", "dup_flag", "lyrics_dup_flag", "use_for_model"), sql_cols))
  dbExecute(con, sprintf("CREATE INDEX idx_songs_%1$s ON songs(%1$s)", cn))

# The model-ready view: not a duplicate, not identical lyrics, valid emotion class
dbExecute(con, sprintf(
  "CREATE VIEW model_ready AS SELECT * FROM songs
   WHERE dup_flag = 0 AND lyrics_dup_flag = 0 AND emotion IN (%s)",
  paste0("'", CFG$emotions, "'", collapse = ", ")))

n_tab  <- dbGetQuery(con, "SELECT COUNT(*) n FROM songs")$n
n_view <- dbGetQuery(con, "SELECT COUNT(*) n FROM model_ready")$n
log_count_step("11_sqlite_written", n_tab, length(sql_cols))
log_count_step("12_model_ready_view", n_view, length(sql_cols))
log_msg(sprintf("SQLite: songs=%s rows, model_ready view=%s rows, columns=%d",
                format(n_tab, big.mark = ","), format(n_view, big.mark = ","), length(sql_cols)))
log_msg("SQLite integrity_check: ", dbGetQuery(con, "PRAGMA integrity_check")[[1]])
if (n_tab != n_clean) stop("SQLite row count does not match parquet!")
if (n_view != n_model) log_msg("WARNING: view rows differ from use_for_model count")
log_msg("SQLite emotion distribution in model_ready:")
em <- dbGetQuery(con, "SELECT emotion, COUNT(*) n FROM model_ready GROUP BY emotion ORDER BY n DESC")
for (i in seq_len(nrow(em))) log_msg(sprintf("   %-12s %s", em$emotion[i], format(em$n[i], big.mark = ",")))
dbDisconnect(con)

# ---------------------------------------------------------------------
# 12. DATA DICTIONARY (docs/data_dictionary.csv and .md)
# ---------------------------------------------------------------------
build_dictionary <- function(chk) {
  n_rows <- sdf_nrow(chk)
  
  # ---- schema ----------------------------------------------------------
  sch <- sdf_schema(chk)
  types <- vapply(sch, function(z) z$type, character(1))
  names(types) <- vapply(sch, function(z) z$name, character(1))
  cols <- names(types)
  nested <- cols[grepl("Array|Struct|Map|array<|struct<|map<", types)]
  in_sqlite <- !(cols %in% c("lyrics", grep("^sim_(artist|song)_", cols, value = TRUE), nested))
  
  # ---- null counts AFTER cleaning (one pass) ---------------------------
  cat("Counting missing values (one pass over the parquet)...\n")
  nl <- collect(sdf_sql(sc, sprintf("SELECT %s FROM clean_chk",
                                    paste(sprintf("SUM(CASE WHEN %s IS NULL THEN 1 ELSE 0 END) AS %s", q(cols), q(cols)),
                                          collapse = ", "))))
  missing_after <- as.numeric(unlist(nl[1, cols]))
  
  # ---- min / max / mean of numeric columns -----------------------------
  num_cols <- setdiff(cols[grepl("^(Double|Float|Long|Integer|Decimal)", types)], "row_id")
  if (length(num_cols) > 0) {
    stat_exprs <- c(sprintf("MIN(%s) AS %s", q(num_cols), q(paste0("min__", num_cols))),
                    sprintf("MAX(%s) AS %s", q(num_cols), q(paste0("max__", num_cols))),
                    sprintf("AVG(%s) AS %s", q(num_cols), q(paste0("avg__", num_cols))))
    st <- collect(sdf_sql(sc, sprintf("SELECT %s FROM clean_chk", paste(stat_exprs, collapse = ", "))))
  } else {
    st <- NULL
  }
  get_stat <- function(prefix) {
    v <- rep(NA_real_, length(cols)); names(v) <- cols
    if (length(num_cols) > 0)
      v[num_cols] <- as.numeric(unlist(st[1, paste0(prefix, num_cols)]))
    unname(v)
  }
  
  # ---- missing BEFORE imputation: parse the 01 log ---------------------
  parse_missing_before <- function() {
    f <- "logs/ingestion_log.txt"
    if (!file.exists(f)) return(NULL)
    lg <- readLines(f, warn = FALSE)
    h <- grep("Missing values BEFORE imputation", lg)
    if (!length(h)) return(NULL)
    h <- tail(h, 1)
    out <- list()
    if (h < length(lg)) for (ln in lg[(h + 1):length(lg)]) {
      m <- regmatches(ln, regexec("\\|\\s+([a-z0-9_]+)\\s+([0-9,]+) \\(([0-9.]+)%\\)\\s*$", ln))[[1]]
      if (length(m) == 4)
        out[[length(out) + 1]] <- data.frame(column = m[2], n = as.numeric(gsub(",", "", m[3])),
                                             pct = as.numeric(m[4]))
      else break
    }
    if (length(out)) do.call(rbind, out) else data.frame(column = character(), n = numeric(), pct = numeric())
  }
  mb <- parse_missing_before()
  if (is.null(mb)) {
    cat("NOTE: could not find the 'Missing values BEFORE imputation' block in the log.\n")
    miss_before <- rep(NA_real_, length(cols)); miss_before_pct <- miss_before
  } else {
    miss_before     <- ifelse(cols %in% mb$column, mb$n[match(cols, mb$column)], 0)
    miss_before_pct <- ifelse(cols %in% mb$column, mb$pct[match(cols, mb$column)], 0)
  }
  
  # ---- imputation medians and IQR bounds from docs ---------------------
  med <- if (file.exists("docs/imputation_medians.csv")) read.csv("docs/imputation_medians.csv") else NULL
  iqr <- if (file.exists("docs/iqr_bounds.csv")) read.csv("docs/iqr_bounds.csv") else NULL
  imp_median <- if (is.null(med)) rep(NA_real_, length(cols)) else med$median[match(cols, med$column)]
  
  # ---- descriptions ----------------------------------------------------
  AUDIO <- c("energy", "valence", "danceability", "tempo", "loudness", "speechiness",
             "liveness", "acousticness", "instrumentalness", "popularity")
  DERIVED <- c("row_id", "first_artist", "genre_primary", "genre_group", "release_year",
               "dup_flag", "mean_similarity", "imputed_any", "outlier_count", "use_for_model",
               "lyrics_group_id", "lyrics_group_size", "lyrics_dup_flag")
  src_map <- c(artist = "Artist(s)", first_artist = "Artist(s)", song = "song", emotion = "emotion",
               lyrics = "text", album = "Album", genre = "Genre", genre_primary = "Genre",
               genre_group = "Genre", release_date = "Release Date", release_year = "Release Date",
               length_sec = "Length", key = "Key", time_signature = "Time signature",
               explicit = "Explicit", energy = "Energy", valence = "Positiveness",
               danceability = "Danceability", tempo = "Tempo", loudness = "Loudness (db)",
               speechiness = "Speechiness", liveness = "Liveness", acousticness = "Acousticness",
               instrumentalness = "Instrumentalness", popularity = "Popularity")
  describe <- function(cn) {
    d <- switch(cn,
                row_id = "Unique row ID (not consecutive). Identical in parquet and SQLite; use it for joins.",
                artist = "Artist string as given in the source.",
                first_artist = "First listed artist (text before the first comma). Use for the artist-grouped train/test split.",
                song = "Song title.", album = "Album name.",
                genre = "Original genre list (lower-case, comma separated).",
                genre_primary = "First genre in the genre list (lower-case, hyphens turned into spaces).",
                genre_group = sprintf("Top %d primary genres by count; every other genre = 'other'.", CFG$top_n_genres),
                release_date = "Release date as a DATE (NULL if it could not be parsed).",
                release_year = "Year of release_date.",
                length_sec = "Song length in seconds, parsed from the mm:ss text.",
                key = "Musical key (text).", time_signature = "Time signature (text).",
                explicit = "1 = explicit lyrics, 0 = not explicit.",
                emotion = "TARGET label: joy, sadness, anger, fear, love or surprise. Machine-generated from lyrics.",
                lyrics = "Raw lyrics text. Stored in parquet only (not in SQLite).",
                dup_flag = "1 = a later copy of the same artist + song (case-insensitive). Flagged, never dropped.",
                imputed_any = "1 = at least one numeric field in this row was median-imputed.",
                mean_similarity = "Mean of the available similarity scores of the similar songs.",
                outlier_count = "Number of continuous audio, length, and similarity fields flagged as IQR outliers.",
                lyrics_group_id = "ID shared by all rows with identical lyrics (smallest row_id of the group). Use it for a group-aware train/test split.",
                lyrics_group_size = "Number of rows (any artist) that share these exact lyrics.",
                lyrics_dup_flag = "1 = a later copy of identical lyrics among rows with dup_flag = 0 (possible data leakage). Flagged, never dropped.",
                use_for_model = "1 = dup_flag = 0 AND lyrics_dup_flag = 0 AND emotion is one of the six valid classes (the model_ready rows).",
                NULL)
    if (!is.null(d)) return(d)
    if (cn %in% AUDIO) return(paste0("Audio feature '", cn, "' as a number."))
    if (grepl("^out_", cn)) {
      f <- sub("^out_", "", cn); b <- if (!is.null(iqr)) iqr[iqr$feature == f, ] else NULL
      rng <- if (!is.null(b) && nrow(b) == 1) sprintf(" Bounds: [%.4g, %.4g].", b$lower, b$upper) else ""
      return(paste0("1 = '", f, "' is outside Q1 - 1.5*IQR / Q3 + 1.5*IQR. Flag only, nothing removed.", rng))
    }
    if (grepl("^sim_score_", cn)) return("Similarity score of similar song n (flattened from the nested JSON).")
    if (grepl("^sim_artist_", cn)) return("Artist of similar song n (flattened). Parquet only.")
    if (grepl("^sim_song_", cn)) return("Title of similar song n (flattened). Parquet only.")
    if (grepl("^good_for_", cn)) return("Context tag, 1 = the song suits this activity, 0 = it does not.")
    "Source column (cleaned)."
  }
  cleaning <- function(cn) {
    if (cn == "row_id") return("Generated; written to parquet once so it never changes.")
    if (cn %in% AUDIO) return(if (cn == "loudness") "Number extracted from text (dB); missing -> median."
                              else if (cn %in% c("energy", "valence", "danceability", "speechiness",
                                                 "liveness", "acousticness", "instrumentalness"))
                                "Text cast to double; percentage-scale values divided by 100; missing -> median."
                              else "Text cast to double; missing -> median.")
    if (cn == "length_sec") return("mm:ss converted to seconds; missing -> median.")
    if (cn == "release_date") return("Several date formats parsed; unparseable -> NULL.")
    if (cn == "emotion") return("Trimmed and lower-cased; 'angry' mapped to 'anger'.")
    if (cn %in% c("artist", "song", "album", "genre", "key", "time_signature", "lyrics"))
      return("Trimmed; empty string -> NULL.")
    if (cn == "mean_similarity") return("Average of available scores; missing -> median.")
    if (cn == "explicit" || grepl("^good_for_", cn)) return("Converted to 0/1.")
    if (grepl("^sim_", cn)) return("Flattened from the nested Similar Songs array.")
    if (cn %in% c("dup_flag", "imputed_any", "outlier_count", "use_for_model", "genre_group",
                  "lyrics_group_id", "lyrics_group_size", "lyrics_dup_flag",
                  "genre_primary", "first_artist", "release_year") || grepl("^out_", cn))
      return("Derived by the pipeline.")
    ""
  }
  source_col <- function(cn) {
    if (cn %in% names(src_map)) return(src_map[[cn]])
    if (grepl("^good_for_", cn)) return(paste("Good for", gsub("_", " ", sub("^good_for_", "", cn))))
    if (grepl("^sim_(score|artist|song)_", cn)) return("Similar Songs (nested)")
    NA_character_
  }
  
  dict <- data.frame(
    column = cols, type = unname(types),
    origin = ifelse(cols %in% DERIVED | grepl("^out_", cols), "derived", "source"),
    source_column = vapply(cols, source_col, character(1)),
    in_sqlite = in_sqlite,
    description = vapply(cols, describe, character(1)),
    cleaning = vapply(cols, cleaning, character(1)),
    missing_before_imputation = miss_before,
    missing_before_pct = miss_before_pct,
    missing_after = missing_after,
    imputation_median = imp_median,
    min = get_stat("min__"), max = get_stat("max__"), mean = round(get_stat("avg__"), 4),
    stringsAsFactors = FALSE, row.names = NULL)
  
  write.csv(dict, "docs/data_dictionary.csv", row.names = FALSE, na = "")
  
  fmt <- function(x) ifelse(is.na(x), "-", format(x, big.mark = ",", trim = TRUE, scientific = FALSE))
  md <- c(sprintf("# Data dictionary - spotify_clean (%s rows, %d columns)", format(n_rows, big.mark = ","), length(cols)),
          "",
          "| Column | Type | Origin | Source column | In SQLite | Missing before | Missing after | Description |",
          "|---|---|---|---|---|---|---|---|",
          sprintf("| %s | %s | %s | %s | %s | %s | %s | %s |", dict$column, dict$type, dict$origin,
                  ifelse(is.na(dict$source_column), "-", dict$source_column),
                  ifelse(dict$in_sqlite, "yes", "no"),
                  fmt(dict$missing_before_imputation), fmt(dict$missing_after), dict$description))
  writeLines(md, "docs/data_dictionary.md")
  
  cat("Wrote docs/data_dictionary.csv and docs/data_dictionary.md (", nrow(dict), " columns)\n", sep = "")
  invisible(dict)
}
build_dictionary(chk)
log_msg("Data dictionary written to docs/")
spark_disconnect(sc)

# ---------------------------------------------------------------------
# 13. AUTOMATIC VERIFICATION of every Stage 1 deliverable (PASS / WARN / FAIL)
# ---------------------------------------------------------------------
verify_stage1 <- function(expected_rows, parquet_path = PATH$clean, sqlite_path = PATH$sqlite) {
  EXPECTED <- expected_rows
  EMOTIONS <- CFG$emotions
  checks <- data.frame(area = character(), check = character(),
                       status = character(), detail = character(), stringsAsFactors = FALSE)
  add <- function(area, check, ok, detail = "", warn = FALSE) {
    status <- if (isTRUE(ok)) "Complete" else if (warn) "WARN" else "FAIL"
    checks <<- rbind(checks, data.frame(area = area, check = check, status = status,
                                        detail = detail, stringsAsFactors = FALSE))
    cat(sprintf("[%s] %-10s %s %s\n", status, area, check, if (nzchar(detail)) paste0("- ", detail) else ""))
  }
  fmt <- function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE)
  `%||%` <- function(a, b) if (is.null(a) || is.na(a)) b else a
  
  # ---- A. files --------------------------------------------------------
  needed <- c(parquet_path, sqlite_path,
              "logs/ingestion_log.txt", "logs/step_row_counts.csv",
              "docs/data_dictionary.csv", "docs/data_dictionary.md",
              "docs/imputation_medians.csv", "docs/iqr_bounds.csv")
  for (f in needed) add("Files", f, file.exists(f) || dir.exists(f))
  parts <- if (dir.exists(parquet_path))
    list.files(parquet_path, pattern = "\\.parquet$", recursive = TRUE, full.names = TRUE) else character(0)
  add("Files", "parquet part files present", length(parts) > 0, paste(length(parts), "part files"))
  if (!file.exists(sqlite_path)) stop("SQLite file missing - cannot continue: ", sqlite_path)
  
  con <- dbConnect(RSQLite::SQLite(), sqlite_path)
  fields <- dbListFields(con, "songs")
  q1 <- function(sql) dbGetQuery(con, sql)[[1]]
  
  # ---- B. row counts and keys -----------------------------------------
  n <- q1("SELECT COUNT(*) FROM songs")
  add("Rows", "SQLite songs = 551,443", n == EXPECTED, fmt(n))
  add("Keys", "row_id is unique", q1("SELECT COUNT(DISTINCT row_id) FROM songs") == n)
  hdr <- readLines("docs/data_dictionary.md", n = 1)
  pr <- suppressWarnings(as.numeric(gsub(",", "", sub(".*\\(([0-9,]+) rows.*", "\\1", hdr))))
  add("Rows", "parquet rows = SQLite rows", isTRUE(pr == n), paste("parquet", fmt(pr), "| sqlite", fmt(n)))
  sqlite_info <- dbGetQuery(con, "PRAGMA table_info(songs)")
  sqlite_types <- setNames(toupper(sqlite_info$type), sqlite_info$name)
  expected_real <- intersect(c("energy", "valence", "danceability", "tempo", "loudness",
                               "speechiness", "liveness", "acousticness", "instrumentalness",
                               "popularity", "length_sec", "mean_similarity"), fields)
  bad_real <- expected_real[sqlite_types[expected_real] != "REAL"]
  add("Types", "cast numeric columns are stored as REAL", length(bad_real) == 0,
      paste(bad_real, collapse = ", "))
  add("Types", "row_id is stored as INTEGER", identical(unname(sqlite_types[["row_id"]]), "INTEGER"),
      sqlite_types[["row_id"]])
  
  req <- c("row_id", "artist", "first_artist", "song", "emotion", "genre", "genre_group",
           "release_date", "release_year", "length_sec", "energy", "valence", "danceability",
           "tempo", "loudness", "speechiness", "liveness", "acousticness", "instrumentalness",
           "popularity", "dup_flag", "lyrics_dup_flag", "lyrics_group_id", "mean_similarity",
           "imputed_any", "outlier_count", "use_for_model")
  miss_cols <- setdiff(req, fields)
  add("Columns", "required columns exist in SQLite", length(miss_cols) == 0, paste(miss_cols, collapse = ", "))
  outc <- grep("^out_", fields, value = TRUE)
  expected_out <- paste0("out_", c(names(AUD), "length_sec", "mean_similarity",
                                   sprintf("sim_score_%d", seq_along(SIM$score))))
  missing_out <- setdiff(expected_out, outc)
  n_outlier_nonzero <- q1("SELECT COUNT(*) FROM songs WHERE outlier_count IS NOT NULL AND outlier_count <> 0")
  add("Columns", "IQR outlier flags are represented correctly",
      length(missing_out) == 0,
      paste(length(outc), "feature flags; missing:", paste(missing_out, collapse = ", "),
            ";", n_outlier_nonzero, "rows with non-zero outlier_count"))
  add("Columns", "lyrics kept out of SQLite (parquet only)", !("lyrics" %in% fields))
  
  # ---- C. SQL view -----------------------------------------------------
  views <- q1("SELECT GROUP_CONCAT(name) FROM sqlite_master WHERE type = 'view'")
  add("SQL", "model_ready view exists", grepl("model_ready", views %||% ""), views)
  n_view <- q1("SELECT COUNT(*) FROM model_ready")
  n_use  <- q1("SELECT COUNT(*) FROM songs WHERE use_for_model = 1")
  add("SQL", "view rows = rows with use_for_model = 1", n_view == n_use, fmt(n_view))
  add("SQL", "integrity_check", q1("PRAGMA integrity_check") == "ok")
  idx <- q1("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_songs_%'")
  add("SQL", "indexes created", idx > 0, paste(idx, "indexes"), warn = TRUE)
  
  # ---- D. cleaning rules ----------------------------------------------
  imp <- intersect(c("energy", "valence", "danceability", "tempo", "loudness", "speechiness",
                     "liveness", "acousticness", "instrumentalness", "popularity",
                     "length_sec", "mean_similarity"), fields)
  nn <- vapply(imp, function(cn) q1(sprintf('SELECT COUNT(*) FROM songs WHERE "%s" IS NULL', cn)), numeric(1))
  add("Cleaning", "no NULLs left in imputed numeric columns", all(nn == 0),
      paste(names(nn)[nn > 0], collapse = ", "))
  n_dup <- q1("SELECT COALESCE(SUM(dup_flag), 0) FROM songs")
  n_lyr <- if ("lyrics_dup_flag" %in% fields) q1("SELECT COALESCE(SUM(lyrics_dup_flag), 0) FROM songs") else NA
  bad_dup_flags <- q1("SELECT COUNT(*) FROM songs WHERE dup_flag IS NULL OR dup_flag NOT IN (0, 1)")
  add("Cleaning", "dup_flag contains only 0/1 values", bad_dup_flags == 0, paste(bad_dup_flags, "invalid values"))
  add("Cleaning", "row count preserved (duplicates flagged, not dropped)", n == EXPECTED,
      sprintf("%s rows; %s duplicate rows flagged (%.2f%%)", fmt(n), fmt(n_dup), 100 * n_dup / max(n, 1)))
  emo <- dbGetQuery(con, "SELECT emotion, COUNT(*) n FROM songs GROUP BY emotion ORDER BY n DESC")
  n_invalid_model <- q1(sprintf(
    "SELECT COUNT(*) FROM songs WHERE (emotion IS NULL OR emotion NOT IN (%s)) AND use_for_model = 1",
    paste0("'", EMOTIONS, "'", collapse = ", ")))
  n_invalid_emotion <- sum(emo$n[is.na(emo$emotion) | !(emo$emotion %in% EMOTIONS)])
  add("Cleaning", "unexpected emotion labels are excluded from modeling", n_invalid_model == 0,
      sprintf("%s rows retained and excluded from modeling", fmt(n_invalid_emotion)))
  emo_m <- dbGetQuery(con, "SELECT DISTINCT emotion FROM model_ready")$emotion
  add("Cleaning", "model_ready holds only valid classes", all(emo_m %in% EMOTIONS))
  ng <- q1("SELECT COUNT(DISTINCT genre_group) FROM songs")
  add("Cleaning", "genre_group = top N + 'other'", ng <= CFG$top_n_genres + 1,
      paste(ng, "groups; maximum", CFG$top_n_genres + 1))
  n_nodate <- q1("SELECT COUNT(*) FROM songs WHERE release_date IS NULL")
  add("Cleaning", "release_date parsed", n_nodate == 0,
      sprintf("%s unparsed (%.2f%%)", fmt(n_nodate), 100 * n_nodate / n), warn = TRUE)
  yr <- dbGetQuery(con, "SELECT MIN(release_year) lo, MAX(release_year) hi FROM songs")
  max_release_year <- as.integer(format(Sys.Date(), "%Y")) + 1L
  add("Cleaning", "release_year in a sensible range", isTRUE(yr$lo >= 1800 && yr$hi <= max_release_year),
      paste(yr$lo, "to", yr$hi), warn = TRUE)
  unit01 <- intersect(c("energy", "valence", "danceability", "speechiness", "liveness",
                        "acousticness", "instrumentalness"), fields)
  off01 <- vapply(unit01, function(cn)
    q1(sprintf('SELECT COUNT(*) FROM songs WHERE "%1$s" < 0 OR "%1$s" > 1', cn)), numeric(1))
  add("Cleaning", "normalized audio features stay inside 0-1", all(off01 == 0),
      paste(names(off01)[off01 > 0], collapse = ", "), warn = TRUE)
  if ("popularity" %in% fields) {
    off_pop <- q1('SELECT COUNT(*) FROM songs WHERE popularity < 0 OR popularity > 100')
    add("Cleaning", "popularity stays inside 0-100", off_pop == 0, paste(off_pop, "out-of-range rows"), warn = TRUE)
  }
  if ("tempo" %in% fields) {
    bad_tempo <- q1('SELECT COUNT(*) FROM songs WHERE tempo < 0')
    add("Cleaning", "tempo is non-negative", bad_tempo == 0, paste(bad_tempo, "negative rows"), warn = TRUE)
  }
  
  # ---- E. parquet-only columns, from the dictionary ---------------------
  dict <- read.csv("docs/data_dictionary.csv", stringsAsFactors = FALSE)
  sim <- dict[grepl("^sim_", dict$column), ]
  if (nrow(sim)) {
    bad_sim <- sim[sim$missing_after / n > 0.5, ]
    add("Similar", "flattened sim_* columns are filled", nrow(bad_sim) == 0,
        paste(bad_sim$column, collapse = ", "), warn = TRUE)
  }
  add("Dictionary", "'missing before imputation' filled from the log",
      any(!is.na(dict$missing_before_imputation)), warn = TRUE)
  add("Dictionary", "every column has a description", all(nzchar(dict$description)))
  
  # ---- F. logs ---------------------------------------------------------
  sl <- read.csv("logs/step_row_counts.csv", stringsAsFactors = FALSE)
  add("Log", "row count logged at each step", nrow(sl) >= 9, paste(nrow(sl), "steps logged"))
  add("Log", "raw record count = 551,443", sl$rows[sl$step == "03_raw_loaded"] == CFG$expected_rows,
      fmt(sl$rows[sl$step == "03_raw_loaded"]))
  preserved_steps <- sl$step != "03_raw_loaded" & sl$step != "12_model_ready_view"
  add("Log", "row count preserved through cleaning and SQLite export",
      all(sl$rows[preserved_steps] == EXPECTED),
      paste(unique(sl$rows[preserved_steps]), collapse = ", "))
  logged_view_rows <- sl$rows[sl$step == "12_model_ready_view"]
  add("Log", "model_ready view row count logged", length(logged_view_rows) == 1 && logged_view_rows == n_view,
      paste(logged_view_rows, "logged;", n_view, "in SQLite"))
  lg <- readLines("logs/ingestion_log.txt", warn = FALSE)
  for (key in c("Raw schema", "Duplicates flagged", "Missing values BEFORE imputation",
                "Imputation medians", "IQR outliers", "Genre groups", "Parquet verified"))
    add("Log", paste0("log contains: ", key), any(grepl(key, lg, fixed = TRUE)), warn = TRUE)
  add("Log", "full run (not the 20,000-row test)", !any(grepl("TEST RUN", lg, fixed = TRUE)))
  
  # ---- G. write documentation outputs ----------------------------------
  dups <- dbGetQuery(con, "SELECT lower(artist) AS artist, lower(song) AS song, COUNT(*) AS copies
  FROM songs WHERE artist IS NOT NULL AND song IS NOT NULL
  GROUP BY lower(artist), lower(song) HAVING COUNT(*) > 1 ORDER BY copies DESC LIMIT 50")
  write.csv(dups, "docs/duplicates_summary.csv", row.names = FALSE)
  n_groups <- q1("SELECT COUNT(*) FROM (SELECT 1 FROM songs WHERE artist IS NOT NULL AND song IS NOT NULL
                GROUP BY lower(artist), lower(song) HAVING COUNT(*) > 1)")
  
  md_table <- function(df) {
    if (!nrow(df)) return("(none)")
    df[] <- lapply(df, function(x) if (is.numeric(x)) fmt(signif(x, 6)) else as.character(x))
    c(paste0("| ", paste(names(df), collapse = " | "), " |"),
      paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|"),
      apply(df, 1, function(r) paste0("| ", paste(trimws(r), collapse = " | "), " |")))
  }
  genre <- dbGetQuery(con, "SELECT genre_group, COUNT(*) AS songs FROM songs GROUP BY genre_group ORDER BY songs DESC")
  emo_model <- dbGetQuery(con, "SELECT emotion, COUNT(*) AS songs FROM model_ready GROUP BY emotion ORDER BY songs DESC")
  outs <- data.frame(feature = sub("^out_", "", outc),
                     outliers = vapply(outc, function(cn) q1(sprintf('SELECT SUM("%s") FROM songs', cn)), numeric(1)))
  meds <- read.csv("docs/imputation_medians.csv"); iqr <- read.csv("docs/iqr_bounds.csv")
  mb <- dict[!is.na(dict$missing_before_imputation) & dict$missing_before_imputation > 0,
             c("column", "missing_before_imputation", "missing_before_pct")]
  names(mb) <- c("column", "missing_before", "pct")
  
  md <- c("# Stage 1 numbers for Section 2.0", "",
          sprintf("- Records: %s raw = %s in parquet = %s in SQLite (nothing dropped)", fmt(EXPECTED), fmt(pr), fmt(n)),
          sprintf("- Duplicates (same artist + song): %s rows flagged (%.2f%%) in %s repeated groups", fmt(n_dup), 100 * n_dup / n, fmt(n_groups)),
          sprintf("- Identical lyrics (any artist), later copies flagged: %s rows (%.2f%%)", fmt(n_lyr), 100 * n_lyr / n),
          sprintf("- model_ready view: %s rows (dup_flag = 0, lyrics_dup_flag = 0 and a valid emotion class)", fmt(n_view)),
          sprintf("- Rows with an unparsed release date: %s", fmt(n_nodate)),
          sprintf("- Rows with at least one imputed value: %s", fmt(q1("SELECT SUM(imputed_any) FROM songs"))),
          "", "## Missing values before imputation", md_table(mb),
          "", "## Imputation medians", md_table(meds),
          "", "## IQR bounds", md_table(iqr),
          "", "## IQR outliers flagged (all rows kept)", md_table(outs),
          "", "## Genre groups", md_table(genre),
          "", "## Emotion distribution - all rows", md_table(emo),
          "", "## Emotion distribution - model_ready", md_table(emo_model))
  writeLines(md, "docs/stage1_numbers_for_report.md")
  write.csv(checks, "docs/stage1_verification.csv", row.names = FALSE)
  dbDisconnect(con)
  
  cat("\n=========== SUMMARY ===========\n")
  print(table(factor(checks$status, levels = c("Complete", "WARN", "FAIL"))))
  cat("Anything marked WARN or FAIL is listed in docs/stage1_verification.csv\n")
  cat("Numbers for your report: docs/stage1_numbers_for_report.md\n")
  
}
log_msg("DONE. Hand off: ", PATH$clean, " | ", PATH$sqlite,
        " | logs/ingestion_log.txt | docs/data_dictionary.csv")
verify_stage1(if (is.null(CFG$sample_rows)) CFG$expected_rows else CFG$sample_rows,
              parquet_path = PATH$clean, sqlite_path = PATH$sqlite)
