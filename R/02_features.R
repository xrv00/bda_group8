# 02_features.R
# Owner: Tam Yi Qing (Spark Feature Engineer)
# Purpose: Train/test split, TF-IDF, audio, lyrics and combined feature sets
# AI-assisted (Claude), 20 Nov 2026
# =====================================================================
# Steps
#   1  connect Spark
#   2  base table (dup_flag = 0, valid emotion)
#   3  clean lyrics
#   4  remove duplicate lyrics
#   5  80/20 split by artist
#   6  tokenise, remove stop words (keep negations)
#   7  check no duplicate lyrics left
#   8  min-max scaling (train only)
#   9  TF-IDF (train only)
#   10 feature vectors, class weights
#   11 write six parquet files
#
# Feature sets
#   audio_*    = 9 scaled audio values
#   lyrics_*   = TF-IDF vector
#   combined_* = 9 audio + mean_similarity + TF-IDF
# =====================================================================

# ---------------------------------------------------------------------
# 0. SETTINGS
# ---------------------------------------------------------------------
PILOT        <- FALSE    # TRUE = quick test with 5% of artists
PILOT_FRAC   <- 0.05
SPOT_CHECK   <- PILOT    # TRUE = show what each tag rule removes
PREFERRED_SPARK <- "3.5.1"   # falls back to newest installed
DRIVER_MEM   <- "8G"     # memory for Spark
CORES        <- 4        # CPU cores for Spark

# Project root = working directory (or its parent if you are inside R/)
PROJECT_ROOT <- normalizePath(getwd(), winslash = "/")
if (basename(PROJECT_ROOT) == "R") PROJECT_ROOT <- dirname(PROJECT_ROOT)
PROC_DIR     <- file.path(PROJECT_ROOT, "data", "processed")

# Input file from 01_ingest.R
IN_CANDIDATES <- c(file.path(PROJECT_ROOT, "data", "clean", "spotify_clean.parquet"),
                   Sys.glob(file.path(PROJECT_ROOT, "data", "clean", "spotify_clean_*.parquet")),
                   file.path(PROJECT_ROOT, "data", "spotify_clean.parquet"),
                   file.path(PROC_DIR, "spotify_clean.parquet"))
IN_FOUND    <- IN_CANDIDATES[file.exists(IN_CANDIDATES)]
IN_PATH     <- if (length(IN_FOUND)) IN_FOUND[which.max(file.mtime(IN_FOUND))] else IN_CANDIDATES[1]
FEATURE_DIR <- if (PILOT) file.path(PROC_DIR, "pilot") else PROC_DIR
RES_DIR     <- if (PILOT) file.path(PROJECT_ROOT, "outputs", "stage2", "pilot") else
  file.path(PROJECT_ROOT, "outputs", "stage2")

# Spark temp folder (set BDA_WORK_ROOT to change)
WORK_ROOT <- Sys.getenv("BDA_WORK_ROOT",
                        unset = if (.Platform$OS.type == "windows") "C:/BDA_work"
                        else file.path(path.expand("~"), "BDA_work"))
WORK_DIR  <- if (PILOT) file.path(WORK_ROOT, "pilot") else WORK_ROOT
TMP_DIR   <- file.path(WORK_DIR, "_tmp")

SALTS     <- c("bda_stage2_v1", "bda_stage2_v2")  # second salt used only if the first fails
TRAIN_PCT <- 80
TOL_PTS   <- 0.5          # class-share tolerance (percentage points)
VOCAB_SIZE <- 10000
MIN_DF     <- 5
USE_HASHING_FALLBACK <- FALSE   # TRUE = HashingTF if memory fails
HASH_FEATURES <- 4096

# expected rows with use_for_model = 1 (NA = skip)
EXPECTED_MODEL_READY <- NA_real_

# TRUE = also remove lyrics identical after cleaning
REMOVE_CLEANED_DUPS <- TRUE

AUDIO9 <- c("energy", "valence", "danceability", "tempo", "loudness",
            "speechiness", "liveness", "acousticness", "instrumentalness")
CLASSES <- c("anger", "fear", "joy", "love", "sadness", "surprise")  # labels 0..5

# ---------------------------------------------------------------------
# 1. PACKAGES, LOGGING, SPARK CONNECTION
# ---------------------------------------------------------------------
pk <- c("sparklyr", "dplyr", "readr")
miss <- pk[!pk %in% rownames(installed.packages())]
if (length(miss)) install.packages(miss)
suppressPackageStartupMessages({ library(sparklyr); library(dplyr) })

for (d in c(FEATURE_DIR, RES_DIR, WORK_DIR, TMP_DIR)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
LOG <- file.path(RES_DIR, "stage2_log.txt")
if (file.exists(LOG)) file.remove(LOG)
say  <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = LOG, append = TRUE) }
hdr  <- function(t) say("\n==== ", t, " ====")
say_df <- function(d) for (l in capture.output(print(as.data.frame(d), row.names = FALSE))) say(l)
reg <- function(x, name) invisible(sdf_register(x, name))   # register a Spark view
T0 <- Sys.time()

hdr("STEP 1: Spark connection")
say("Run mode: ", if (PILOT) "PILOT (5% of artists)" else "FULL")
say("Started : ", format(T0))
say("Project root : ", PROJECT_ROOT)
say("Input        : ", IN_PATH)
say("Spark work   : ", WORK_DIR)

# Windows: point Spark to the Hadoop files if they exist
if (.Platform$OS.type == "windows") {
  hh <- Sys.getenv("HADOOP_HOME")
  if (hh == "" && dir.exists("C:/hadoop/bin")) { hh <- "C:/hadoop"; Sys.setenv(HADOOP_HOME = hh) }
  if (hh != "") Sys.setenv(PATH = paste(normalizePath(file.path(hh, "bin"), winslash = "\\"), Sys.getenv("PATH"), sep = ";"))
}

if (!file.exists(IN_PATH))
  stop("Input not found. Looked in: ", paste(IN_CANDIDATES, collapse = " | "),
       "\nRun R/01_ingest.R first.")

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
# 2. BASE TABLE
# Keep dup_flag = 0 and a valid emotion. Step 4 removes lyrics_dup_flag = 1.
# ---------------------------------------------------------------------
hdr("STEP 2: read parquet and select columns")
in_norm <- normalizePath(IN_PATH, winslash = "/")
raw <- spark_read_parquet(sc, "raw_clean", in_norm, memory = FALSE)
cols <- colnames(raw)
say("Parquet has ", length(cols), " columns.")

need <- c("row_id", "emotion", "use_for_model", "dup_flag", "lyrics_dup_flag", "mean_similarity", AUDIO9)
absent <- setdiff(need, cols)
lyrics_col <- intersect(c("lyrics", "text"), cols)[1]
artist_col <- intersect(c("first_artist", "artist"), cols)[1]
if (length(absent) || is.na(lyrics_col) || is.na(artist_col)) {
  say("Missing columns: ", paste(absent, collapse = ", "))
  say("Lyrics column found: ", lyrics_col, " | artist column found: ", artist_col)
  say("All columns: ", paste(cols, collapse = ", "))
  if ("lyrics_dup_flag" %in% absent)
    say("lyrics_dup_flag is missing. Rerun 01_ingest.R.")
  stop("Required columns not found. See 'All columns' in the log.")
}
artist_expr <- if (artist_col == "first_artist") "first_artist" else "trim(split(artist, ',')[0])"
say("Lyrics column: ", lyrics_col, " | artist source: ", artist_col,
    if (artist_col != "first_artist") " (first_artist not found, taking text before first comma)" else "")

label_expr <- paste0("CASE emotion_c ",
                     paste(sprintf("WHEN '%s' THEN %d.0", CLASSES, seq_along(CLASSES) - 1L), collapse = " "), " END")
audio_cast <- paste(sprintf("CAST(`%s` AS DOUBLE) AS %s", AUDIO9, AUDIO9), collapse = ", ")
cls_in <- paste(sprintf("'%s'", CLASSES), collapse = ",")

reg(sdf_sql(sc, sprintf("
  SELECT row_id, artist_key, emotion_c AS emotion, %s AS label, %s,
         mean_similarity_d AS mean_similarity, dup_flag, lyrics_dup_flag, use_for_model, lyrics
  FROM (
    SELECT CAST(row_id AS BIGINT) AS row_id,
           coalesce(lower(trim(CAST(%s AS STRING))), concat('__row_', CAST(row_id AS STRING))) AS artist_key,
           lower(trim(emotion)) AS emotion_c,
           %s,
           CAST(mean_similarity AS DOUBLE) AS mean_similarity_d,
           CAST(dup_flag AS INT) AS dup_flag,
           CAST(lyrics_dup_flag AS INT) AS lyrics_dup_flag,
           CAST(use_for_model AS INT) AS use_for_model,
           CAST(`%s` AS STRING) AS lyrics
    FROM raw_clean
    WHERE CAST(dup_flag AS INT) = 0
  ) WHERE emotion_c IN (%s)",
                        label_expr, paste(AUDIO9, collapse = ", "),
                        artist_expr, audio_cast, lyrics_col, cls_in)), "base_all")

n_raw_rows <- sdf_nrow(raw)
n_all <- sdf_nrow(tbl(sc, "base_all"))
say("Rows in parquet: ", format(n_raw_rows, big.mark = ","))
say("Rows with dup_flag = 0 and a valid emotion (base table): ", format(n_all, big.mark = ","))
fl_all <- sdf_sql(sc, "SELECT COALESCE(SUM(CASE WHEN lyrics_dup_flag = 1 THEN 1 ELSE 0 END), 0) AS n_lyr_flag,
                              COALESCE(SUM(use_for_model), 0) AS n_ufm FROM base_all") %>% collect()
n_lyr_flag_all <- as.numeric(fl_all$n_lyr_flag)
n_ufm          <- as.numeric(fl_all$n_ufm)
say("  of which lyrics_dup_flag = 1 (identical lyrics, removed in Step 4): ", format(n_lyr_flag_all, big.mark = ","))
say("  rows with use_for_model = 1: ", format(n_ufm, big.mark = ","))
if (n_all - n_lyr_flag_all != n_ufm) {
  say("WARNING: base rows minus lyrics_dup_flag rows (", format(n_all - n_lyr_flag_all, big.mark = ","),
      ") differ from use_for_model = 1 (", format(n_ufm, big.mark = ","), "). Check use_for_model.")
  } else say("Check: base rows minus lyrics_dup_flag rows = use_for_model rows -> PASS")
if (!is.na(EXPECTED_MODEL_READY) && n_ufm != EXPECTED_MODEL_READY)
  say("WARNING: expected ", format(EXPECTED_MODEL_READY, big.mark = ","), " model_ready rows, found ", format(n_ufm, big.mark = ","))

n_dup_id <- sdf_sql(sc, "SELECT COUNT(*) - COUNT(DISTINCT row_id) AS d FROM base_all") %>% collect()
say("Duplicate row_id values: ", n_dup_id$d, " (must be 0)")
if (n_dup_id$d != 0) stop("row_id is not unique.")

# ---------------------------------------------------------------------
# 3. LYRICS CLEANING
# Lyrics cleaning rules:
#   1a/1b  strip [...] tags (one line, max 300 chars, two passes)
#   1c     strip tags with bad brackets: [Verse 1}  {Chorus]
#   1d     delete lines that are only a section label
#   1e     delete a label prefix: "Chorus: la la" -> "la la"
#   1f     delete (Chorus) / (Verse 2) in round brackets
#   2      curly apostrophes -> '
#   3      lower-case
#   4      keep only a-z, apostrophes, spaces
#   5      drop apostrophes outside words
#   6      collapse whitespace
# Common words like break, drop, part are not in the label list.
# ---------------------------------------------------------------------
rx  <- function(e, pat, rep) sprintf("regexp_replace(%s, %s, %s)", e, pat, rep)
lit <- function(p) paste0("'", p, "'")        # regex -> Spark SQL string literal
tag_pat <- r"---('\\[[^\\[\\]\\r\\n]{0,300}\\]')---"

LAB <- "verse|chorus|bridge|intro|outro|hook|refrain|interlude|instrumental|solo|breakdown|spoken|skit"
# 1c: [ or { + section word + up to 60 chars + optional ] or }
br_mis_pat <- lit(paste0(r"---((?i)[\\[{][ \\t]*(?:pre[- ]?|post[- ]?)?()---", LAB,
                         r"---()(?![a-z])[^\\[\\]{}\\r\\n]{0,60}[\\]}]?)---"))
# 1d: whole line is only a label
whole_pat  <- lit(paste0(r"---((?mi)^[ \\t]*(?:pre[- ]?|post[- ]?)?()---", LAB,
                         r"---()(?:[ \\t]*[0-9]+|[ \\t]+[ivx]+)?(?:[ \\t]*(?:x[0-9]+|[0-9]+x))?[ \\t:]*[\\r]*$)---"))
# 1e: label with colon at line start, followed by text
prefix_pat <- lit(paste0(r"---((?mi)^[ \\t]*(?:pre[- ]?|post[- ]?)?()---", LAB,
                         r"---()(?:[ \\t]*[0-9]+)?[ \\t]*:[ \\t]*(?=\\S))---"))
# 1f: label-only round brackets, real lyrics are kept
paren_pat  <- lit(paste0(r"---((?i)\\((?:pre[- ]?|post[- ]?)?()---", LAB,
                         r"---()(?![a-z])(?:[ \\t]*[0-9]+|[ \\t]+[ivx]+)?(?:[ \\t]*(?:x[0-9]+|[0-9]+x))?[ \\t]*\\))---"))

e <- "coalesce(lyrics, '')"
e <- rx(e, tag_pat, "' '")                                        # 1a
e <- rx(e, tag_pat, "' '")                                        # 1b
e <- rx(e, br_mis_pat, "' '")                                     # 1c
e <- rx(e, whole_pat, "' '")                                      # 1d
e <- rx(e, prefix_pat, "''")                                      # 1e
e <- rx(e, paren_pat, "' '")                                      # 1f
e <- rx(e, r"---('[\\u2019\\u2018\\u0060]')---", "\"'\"")        # 2
e <- sprintf("lower(%s)", e)                                      # 3
e <- rx(e, "\"[^a-z' ]\"", "' '")                                 # 4
e <- rx(e, "\"(?<![a-z])'|'(?![a-z])\"", "' '")                  # 5
e <- sprintf("trim(%s)", rx(e, r"---('\\s+')---", "' '"))         # 6

hdr("STEP 3: lyrics cleaning and usable-lyrics check")

# PILOT only: show the most common strings each rule deletes
if (SPOT_CHECK) {
  say("Spot-check on a 5% sample of songs: most common strings each rule deletes")
  spot <- function(pat, n = 30) sdf_sql(sc, paste0(
    "SELECT lower(trim(m)) AS matched, COUNT(*) AS n FROM (",
    "SELECT explode(regexp_extract_all(lyrics, ", pat, ", 0)) AS m FROM base_all ",
    "WHERE pmod(hash(row_id), 100) < 5) GROUP BY lower(trim(m)) ORDER BY n DESC LIMIT ", n)) %>% collect()
  for (nm in c("1c mismatched brackets", "1d whole-line labels", "1e prefix labels", "1f round-bracket labels")) {
    p <- switch(substr(nm, 1, 2), "1c" = br_mis_pat, "1d" = whole_pat, "1e" = prefix_pat, "1f" = paren_pat)
    say("\n-- ", nm)
    say_df(spot(p))
  }
  say("")
}

lyr_q <- sprintf("
  SELECT row_id,
         CASE WHEN lyrics IS NULL THEN 'null'
              WHEN trim(lyrics) = '' THEN 'blank'
              WHEN lyr_clean = '' THEN 'empty_after_cleaning'
              ELSE 'ok' END AS lyr_status,
         lyr_clean,
         md5(lyr_clean) AS lyr_hash
  FROM (SELECT row_id, lyrics, %s AS lyr_clean FROM base_all)", e)
CLEAN_PATH <- file.path(TMP_DIR, "lyrics_clean.parquet")
say("Writing cleaned lyrics to ", CLEAN_PATH, " (heavy step)...")
spark_write_parquet(sdf_sql(sc, lyr_q), CLEAN_PATH, mode = "overwrite")
invisible(spark_read_parquet(sc, "lyr_clean_tbl", CLEAN_PATH, memory = FALSE))

st <- sdf_sql(sc, "SELECT lyr_status, COUNT(*) AS n FROM lyr_clean_tbl GROUP BY lyr_status") %>% collect()
say("Lyrics status (null = missing, blank = whitespace only, empty_after_cleaning = no a-z letters left):")
say_df(st %>% arrange(desc(n)))
n_bad <- sum(st$n[st$lyr_status != "ok"])
say("Songs with no usable lyrics (removed): ", format(n_bad, big.mark = ","),
    " (", round(100 * n_bad / n_all, 4), "% of the base).")
if (n_bad > 0) {
  bad_s <- sdf_sql(sc, "
    SELECT b.row_id, b.emotion, substr(b.lyrics, 1, 80) AS lyrics_start
    FROM base_all b JOIN lyr_clean_tbl f ON b.row_id = f.row_id
    WHERE f.lyr_status <> 'ok' LIMIT 10") %>% collect()
  say("First removed songs:"); say_df(bad_s)
  if (n_bad / n_all > 0.01) say("WARNING: more than 1% of rows have no usable lyrics. Check the cleaning rule.")
}

# ---------------------------------------------------------------------
# 4. REMOVE DUPLICATE LYRICS (before the split)
# 4a: lyrics_dup_flag = 1 (from 01_ingest.R) marks later copies of identical raw lyrics
# 4b: songs identical only after lyrics cleaning (keep smallest row_id)
# ---------------------------------------------------------------------
hdr("STEP 4: remove duplicate lyrics (lyrics_dup_flag), BEFORE the split")
reg(sdf_sql(sc, "
  SELECT b.row_id, b.emotion, b.lyrics_dup_flag, c.lyr_hash
  FROM base_all b JOIN lyr_clean_tbl c ON b.row_id = c.row_id
  WHERE c.lyr_status = 'ok'"), "lyr_ok")

fl <- sdf_sql(sc, "SELECT COUNT(*) AS n_ok,
                          COALESCE(SUM(CASE WHEN lyrics_dup_flag = 1 THEN 1 ELSE 0 END), 0) AS n_flagged
                   FROM lyr_ok") %>% collect()
n_flag_removed <- as.numeric(fl$n_flagged)
say("4a. Songs with usable lyrics: ", format(as.numeric(fl$n_ok), big.mark = ","))
say("    Removed because lyrics_dup_flag = 1: ", format(n_flag_removed, big.mark = ","),
    " (", round(100 * n_flag_removed / max(1, n_all - n_bad), 3), "% of the base)")
if (n_lyr_flag_all > n_flag_removed)
  say("    (", format(n_lyr_flag_all - n_flag_removed, big.mark = ","),
      " other flagged rows were already counted as unusable lyrics in Step 3)")
if (n_flag_removed > 0) {
  rem_emo <- sdf_sql(sc, "SELECT emotion, COUNT(*) AS removed_by_flag FROM lyr_ok
                          WHERE lyrics_dup_flag = 1 GROUP BY emotion") %>% collect()
  say("    Removed by emotion:"); say_df(rem_emo %>% arrange(desc(removed_by_flag)))
}

# 4b
reg(sdf_sql(sc, "SELECT row_id, lyr_hash FROM lyr_ok WHERE lyrics_dup_flag = 0"), "flag_kept")
reg(sdf_sql(sc, "SELECT row_id, ROW_NUMBER() OVER (PARTITION BY lyr_hash ORDER BY row_id) AS rn
                 FROM flag_kept"), "dedup_rank")
n_resid <- as.numeric(sdf_sql(sc, "SELECT COALESCE(SUM(CASE WHEN rn > 1 THEN 1 ELSE 0 END), 0) AS n
                                   FROM dedup_rank") %>% collect() %>% pull(n))
n_resid_removed <- if (REMOVE_CLEANED_DUPS) n_resid else 0
say("4b. Extra duplicates after tag cleaning (not caught by the flag): ", format(n_resid, big.mark = ","),
    if (REMOVE_CLEANED_DUPS) " -> removed (smallest row_id kept)" else " -> NOT removed (REMOVE_CLEANED_DUPS = FALSE)")
if (n_resid > 0) {
  top_g <- sdf_sql(sc, "
    SELECT COUNT(*) AS songs, MIN(substr(c.lyr_clean, 1, 70)) AS cleaned_lyrics_start
    FROM flag_kept k JOIN lyr_clean_tbl c ON k.row_id = c.row_id
    GROUP BY c.lyr_hash HAVING COUNT(*) > 1 ORDER BY songs DESC LIMIT 5") %>% collect()
  say("    Five largest residual groups (placeholder texts such as 'instrumental' collapse to one song):")
  say_df(top_g)
}
reg(sdf_sql(sc, if (REMOVE_CLEANED_DUPS) "SELECT row_id FROM dedup_rank WHERE rn = 1"
            else "SELECT row_id FROM flag_kept"), "keep_ids")

n_dup_removed <- n_flag_removed + n_resid_removed
if (n_dup_removed / max(1, n_all - n_bad) > 0.05)
  say("WARNING: more than 5% of songs removed as duplicate lyrics. Check the cleaning.")

reg(sdf_sql(sc, sprintf("
  SELECT b.row_id, b.artist_key, b.emotion, b.label, %s, b.mean_similarity
  FROM base_all b JOIN keep_ids k ON b.row_id = k.row_id", paste0("b.", AUDIO9, collapse = ", "))), "base_dedup")
n_dedup <- sdf_nrow(tbl(sc, "base_dedup"))
say("Base rows after removals: ", format(n_dedup, big.mark = ","), " (", format(n_all, big.mark = ","),
    " minus ", format(n_bad, big.mark = ","), " unusable minus ", format(n_flag_removed, big.mark = ","),
    " lyrics_dup_flag minus ", format(n_resid_removed, big.mark = ","), " cleaned-lyric duplicates)")
if (n_dedup != n_all - n_bad - n_dup_removed) stop("Row count after removals does not reconcile.")
say("Row-count: ", format(n_raw_rows, big.mark = ","), " raw -> ", format(n_all, big.mark = ","),
    " after dup_flag and emotion filter -> ", format(n_dedup, big.mark = ","), " after removals.")

if (PILOT) {
  reg(sdf_sql(sc, sprintf(
    "SELECT * FROM base_dedup WHERE pmod(hash(artist_key, 'pilot_v1'), 1000) < %d",
    as.integer(PILOT_FRAC * 1000))), "base_pre")
  say("PILOT: kept about ", PILOT_FRAC * 100, "% of artists. Rows: ",
      format(sdf_nrow(tbl(sc, "base_pre")), big.mark = ","))
} else {
  reg(sdf_sql(sc, "SELECT * FROM base_dedup"), "base_pre")
}
n_pre <- sdf_nrow(tbl(sc, "base_pre"))

# ---------------------------------------------------------------------
# 5. ARTIST SPLIT
# All songs of one artist go to the same side.
# ---------------------------------------------------------------------
hdr("STEP 5: 80/20 artist split")
evaluate_split <- function(salt) {
  reg(sdf_sql(sc, sprintf(
    "SELECT *, CASE WHEN pmod(hash(artist_key, '%s'), 1000) < %d THEN 'train' ELSE 'test' END AS split FROM base_pre",
    salt, as.integer(TRAIN_PCT * 10))), "base_split")
  cnt <- sdf_sql(sc, "SELECT split, emotion, COUNT(*) AS n FROM base_split GROUP BY split, emotion") %>% collect()
  overall <- cnt %>% group_by(emotion) %>% summarise(n_all = sum(n), .groups = "drop") %>%
    mutate(share_all = 100 * n_all / sum(n_all))
  tab <- cnt %>% group_by(split) %>% mutate(share = 100 * n / sum(n)) %>% ungroup() %>%
    left_join(overall, by = "emotion") %>% mutate(dev_pts = share - share_all) %>%
    arrange(emotion, split)
  list(maxdev = max(abs(tab$dev_pts)), tab = tab)
}

chosen <- NA
for (i in seq_along(SALTS)) {
  r <- evaluate_split(SALTS[i])
  say("Salt '", SALTS[i], "': max class-share deviation = ", round(r$maxdev, 3), " points (tolerance ", TOL_PTS, ")")
  if (r$maxdev <= TOL_PTS || PILOT || i == length(SALTS)) { chosen <- i; break }
  say("  Check failed, trying the second salt.")
}
if (!PILOT && r$maxdev > TOL_PTS) say("WARNING: second salt also above tolerance.")
say("Salt used: ", SALTS[chosen])
say_df(r$tab %>% mutate(across(c(share, share_all, dev_pts), ~ round(.x, 2))))

by_split <- sdf_sql(sc, "SELECT split, COUNT(DISTINCT artist_key) AS artists, COUNT(*) AS songs FROM base_split GROUP BY split") %>% collect()
say_df(by_split)
say("Artist share in train : ", round(100 * by_split$artists[by_split$split == "train"] / sum(by_split$artists), 2), "%")
say("Row ratio (train/all) : ", round(100 * by_split$songs[by_split$split == "train"] / sum(by_split$songs), 2), "%")
leak <- sdf_sql(sc, "SELECT COUNT(*) AS n FROM (SELECT artist_key FROM base_split GROUP BY artist_key HAVING COUNT(DISTINCT split) > 1)") %>% collect()
say("Artists present in BOTH train and test: ", leak$n, " (must be 0)")
if (leak$n != 0) stop("Artist leakage in split.")

readr::write_csv(
  sdf_sql(sc, "SELECT artist_key, split, COUNT(*) AS songs FROM base_split GROUP BY artist_key, split") %>% collect(),
  file.path(RES_DIR, "split_table.csv"))

# ---------------------------------------------------------------------
# 6. TOKENISE AND REMOVE STOP WORDS
# ---------------------------------------------------------------------
hdr("STEP 6: tokenising and stop words")
reg(sdf_sql(sc, sprintf("
  SELECT s.row_id, s.artist_key, s.emotion, s.label, s.split, %s, s.mean_similarity, c.lyr_clean
  FROM base_split s JOIN lyr_clean_tbl c ON s.row_id = c.row_id",
                        paste0("s.", AUDIO9, collapse = ", "))), "cleaned")

stops_all <- unlist(ml_default_stop_words(sc, "english"))
neg <- stops_all[stops_all %in% c("no", "nor", "not", "cannot") | grepl("n't$", stops_all)]
STOPS <- setdiff(stops_all, neg)   # keep negations
say("Spark English stop words: ", length(stops_all), " | negations kept (removed from list): ", length(neg),
    " | final list: ", length(STOPS))
say("Negations kept: ", paste(sort(neg), collapse = ", "))

tok <- tbl(sc, "cleaned") %>%
  ft_regex_tokenizer(input_col = "lyr_clean", output_col = "tok_raw", pattern = "\\s+",
                     gaps = TRUE, min_token_length = 2, to_lower_case = FALSE) %>%
  ft_stop_words_remover(input_col = "tok_raw", output_col = "tokens", stop_words = STOPS) %>%
  select(-lyr_clean, -tok_raw)

TOK_PATH <- file.path(TMP_DIR, "tokens_all.parquet")
say("Writing cleaned tokens to ", TOK_PATH, " (heavy step, can take several minutes on the full run)...")
spark_write_parquet(tok, TOK_PATH, mode = "overwrite")
tok_all <- spark_read_parquet(sc, "tok_all", TOK_PATH, memory = FALSE)
say("Rows after cleaning/tokenising: ", format(sdf_nrow(tok_all), big.mark = ","))
n_empty <- sdf_sql(sc, "SELECT SUM(CASE WHEN size(tokens) = 0 THEN 1 ELSE 0 END) AS z FROM tok_all") %>% collect()
say("Songs whose tokens are all stop words (kept, zero lyric vector): ", n_empty$z)

# Check leftover section words
smp <- sdf_sql(sc, "SELECT emotion, concat_ws(' ', slice(tokens, 1, 25)) AS first_tokens FROM tok_all LIMIT 3") %>% collect()
say("Sample cleaned tokens (first 25 per song):")
for (i in seq_len(nrow(smp))) say("  [", smp$emotion[i], "] ", smp$first_tokens[i])
tagchk <- sdf_sql(sc, "
  SELECT SUM(CASE WHEN array_contains(tokens, 'verse') OR array_contains(tokens, 'chorus')
                       OR array_contains(tokens, 'bridge') THEN 1 ELSE 0 END) AS n_tag,
         COUNT(*) AS n FROM tok_all") %>% collect()
share_tag <- tagchk$n_tag / tagchk$n
say("Songs still containing the token verse, chorus or bridge: ", tagchk$n_tag, " (",
    round(100 * share_tag, 2), "%). Most are real lyric words.")
if (share_tag > 0.10) stop("Tag cleaning failed: over 10% of songs still contain section words.")

# ---------------------------------------------------------------------
# 7. ZERO CHECK
# ---------------------------------------------------------------------
hdr("STEP 7: zero check for duplicate lyrics")
n_flag_left <- as.numeric(sdf_sql(sc, "
  SELECT COUNT(*) AS n FROM base_split s JOIN base_all b ON s.row_id = b.row_id
  WHERE b.lyrics_dup_flag = 1") %>% collect() %>% pull(n))
say("Rows with lyrics_dup_flag = 1 still in the split: ", n_flag_left, " (must be 0)")
if (n_flag_left != 0) stop("Flagged duplicate lyrics are still in the split.")

dchk <- sdf_sql(sc, "
  SELECT COUNT(*) - COUNT(DISTINCT c.lyr_hash) AS repeated_rows,
         (SELECT COUNT(*) FROM (SELECT c2.lyr_hash FROM base_split s2 JOIN lyr_clean_tbl c2 ON s2.row_id = c2.row_id
                                GROUP BY c2.lyr_hash HAVING COUNT(DISTINCT s2.split) > 1)) AS hashes_in_both_splits
  FROM base_split s JOIN lyr_clean_tbl c ON s.row_id = c.row_id") %>% collect()
say("Repeated cleaned-lyric rows left: ", dchk$repeated_rows, " | cleaned-lyric hashes in both train and test: ",
    dchk$hashes_in_both_splits, if (REMOVE_CLEANED_DUPS) " (both must be 0)" else " (info only)")
if (dchk$repeated_rows != 0 || dchk$hashes_in_both_splits != 0) {
  if (REMOVE_CLEANED_DUPS) stop("Duplicate lyrics remain after Step 4.")
  say("WARNING: duplicate lyrics remain, possible small leakage.")
}

# ---------------------------------------------------------------------
# 8. MIN-MAX SCALING (train only, test clipped to [0, 1])
# ---------------------------------------------------------------------
hdr("STEP 8: min-max scaling (train only) with test clipped to [0, 1]")
src_expr <- c(setNames(AUDIO9, AUDIO9), mean_similarity = "mean_similarity")

chk <- paste(sprintf("SUM(CASE WHEN %s IS NULL THEN 1 ELSE 0 END) AS na_%s",
                     names(src_expr), names(src_expr)), collapse = ", ")
na_res <- sdf_sql(sc, paste("SELECT", chk, "FROM tok_all")) %>% collect()
say("NULL values in numeric features: ", sum(unlist(na_res)), " (must be 0)")
if (sum(unlist(na_res)) > 0) stop("NULLs found in numeric features. Check 01_ingest.R output.")

q <- paste0("SELECT ",
            paste(sprintf("MIN(%1$s) AS mn_%2$s, MAX(%1$s) AS mx_%2$s", src_expr, names(src_expr)), collapse = ", "),
            " FROM tok_all WHERE split = 'train'")
mm_row <- sdf_sql(sc, q) %>% collect()
mm <- data.frame(feature = names(src_expr),
                 train_min = unlist(mm_row[paste0("mn_", names(src_expr))], use.names = FALSE),
                 train_max = unlist(mm_row[paste0("mx_", names(src_expr))], use.names = FALSE))
readr::write_csv(mm, file.path(RES_DIR, "minmax_table.csv"))
say_df(mm)

scale_one <- function(expr, mn, mx) {
  if (mx > mn) sprintf("least(1.0, greatest(0.0, ((%s) - (%.15g)) / (%.15g)))", expr, mn, mx - mn) else "0.0"
}
scaled_cols <- mapply(scale_one, src_expr, mm$train_min, mm$train_max)
reg(sdf_sql(sc, paste0(
  "SELECT row_id, artist_key, emotion, label, split, tokens, ",
  paste(sprintf("%s AS s_%s", scaled_cols, names(src_expr)), collapse = ", "),
  " FROM tok_all")), "scaled_all")

s_names <- paste0("s_", names(src_expr))
rng <- sdf_sql(sc, paste0("SELECT least(", paste(sprintf("MIN(%s)", s_names), collapse = ","), ") AS mn, greatest(",
                          paste(sprintf("MAX(%s)", s_names), collapse = ","), ") AS mx FROM scaled_all")) %>% collect()
say("Scaled numeric features: min = ", round(rng$mn, 4), ", max = ", round(rng$mx, 4), " (must be within 0 and 1)")
say("Negative values check: ", if (rng$mn >= 0) "PASS (no negatives, safe for Naive Bayes)" else "FAIL")
if (rng$mn < 0) stop("Negative values present; Naive Bayes would fail.")

# ---------------------------------------------------------------------
# 9. TF-IDF (train only)
# ---------------------------------------------------------------------
hdr("STEP 9: TF-IDF (fitted on train only)")
scaled_all   <- tbl(sc, "scaled_all")
scaled_train <- scaled_all %>% dplyr::filter(split == "train")

if (!USE_HASHING_FALLBACK) {
  cv_model <- ft_count_vectorizer(sc, input_col = "tokens", output_col = "tf",
                                  vocab_size = VOCAB_SIZE, min_df = MIN_DF) %>% ml_fit(scaled_train)
  vocab <- cv_model$vocabulary
  say("CountVectorizer vocabulary size: ", length(vocab), " (cap ", VOCAB_SIZE, ", minDF ", MIN_DF, ")")
  say("Top 15 terms by frequency: ", paste(head(vocab, 15), collapse = ", "))
  # Section words should rank low in the vocabulary
  lab_rank <- match(c("verse", "chorus", "hook", "bridge", "intro", "outro"), vocab)
  say("Vocabulary rank (1 = most frequent) of verse, chorus, hook, bridge, intro, outro: ",
      paste(ifelse(is.na(lab_rank), "not in vocab", lab_rank), collapse = ", "))
  readr::write_csv(data.frame(index = seq_along(vocab) - 1L, term = vocab), file.path(RES_DIR, "vocabulary.csv"))
  tf_fun <- function(df) ml_transform(cv_model, df)
} else {
  hv <- ft_hashing_tf(sc, input_col = "tokens", output_col = "tf", num_features = HASH_FEATURES)
  say("HashingTF fallback with ", HASH_FEATURES, " features (no vocabulary file).")
  tf_fun <- function(df) ml_transform(hv, df)
}
idf_model <- ft_idf(sc, input_col = "tf", output_col = "tfidf") %>% ml_fit(tf_fun(scaled_train))
say("IDF fitted on ", format(sdf_nrow(scaled_train), big.mark = ","), " training rows.")

with_tfidf <- scaled_all %>% tf_fun() %>% ml_transform(idf_model, .)

# ---------------------------------------------------------------------
# 10. FEATURE VECTORS AND CLASS WEIGHTS
# ---------------------------------------------------------------------
hdr("STEP 10: assemble vectors and class weights")
a9  <- ft_vector_assembler(sc, input_cols = paste0("s_", AUDIO9), output_col = "f_audio9")
ac  <- ft_vector_assembler(sc, input_cols = c(paste0("s_", AUDIO9), "s_mean_similarity", "tfidf"), output_col = "f_combined")
assembled <- with_tfidf %>% ml_transform(a9, .) %>% ml_transform(ac, .)

cls_cnt <- sdf_sql(sc, "SELECT label, COUNT(*) AS n FROM scaled_all WHERE split = 'train' GROUP BY label ORDER BY label") %>% collect()
if (nrow(cls_cnt) != length(CLASSES)) stop("A class is missing from train. Cannot compute class weights.")
N_train <- sum(cls_cnt$n)
cls_cnt$weight <- N_train / (length(CLASSES) * cls_cnt$n)
cls_cnt$emotion <- CLASSES[cls_cnt$label + 1]
say("Class weights = N / (6 x class count), from train only (N = ", format(N_train, big.mark = ","), ")")
say_df(cls_cnt %>% mutate(weight = round(weight, 4)) %>% select(label, emotion, n, weight))
readr::write_csv(cls_cnt %>% select(label, emotion, n, weight), file.path(RES_DIR, "class_weights.csv"))
w_expr <- paste0("CASE label ", paste(sprintf("WHEN %d.0 THEN %.15g", cls_cnt$label, cls_cnt$weight), collapse = " "), " END")

reg(assembled %>% select(row_id, artist_key, emotion, label, split, f_audio9, tfidf, f_combined), "assembled")
final <- sdf_sql(sc, sprintf("SELECT *, %s AS weight FROM assembled", w_expr))

FINAL_PATH <- file.path(TMP_DIR, "final_all.parquet")
say("Writing assembled vectors to ", FINAL_PATH, " ...")
spark_write_parquet(final, FINAL_PATH, mode = "overwrite")
fin <- spark_read_parquet(sc, "final_all", FINAL_PATH, memory = FALSE)

# ---------------------------------------------------------------------
# 11. WRITE, CHECK AND COPY THE SIX FILES
# ---------------------------------------------------------------------
hdr("STEP 11: write the six parquet files")
out_specs <- list(
  audio_train    = "SELECT row_id, artist_key, emotion, label, weight, f_audio9 AS features FROM final_all WHERE split = 'train'",
  audio_test     = "SELECT row_id, artist_key, emotion, label, f_audio9 AS features FROM final_all WHERE split = 'test'",
  lyrics_train   = "SELECT row_id, artist_key, emotion, label, weight, tfidf AS features FROM final_all WHERE split = 'train'",
  lyrics_test    = "SELECT row_id, artist_key, emotion, label, tfidf AS features FROM final_all WHERE split = 'test'",
  combined_train = "SELECT row_id, artist_key, emotion, label, weight, f_combined AS features FROM final_all WHERE split = 'train'",
  combined_test  = "SELECT row_id, artist_key, emotion, label, f_combined AS features FROM final_all WHERE split = 'test'"
)

counts <- data.frame(file = character(), rows = numeric())
for (nm in names(out_specs)) {
  p <- file.path(WORK_DIR, paste0(nm, ".parquet"))
  spark_write_parquet(sdf_sql(sc, out_specs[[nm]]), p, mode = "overwrite")
  n <- sdf_nrow(spark_read_parquet(sc, paste0("chk_", nm), p, memory = FALSE))
  counts <- rbind(counts, data.frame(file = paste0(nm, ".parquet"), rows = n))
}
say_df(counts)
say("Row-count consistency (train files equal, test files equal): ",
    if (length(unique(counts$rows[grepl("train", counts$file)])) == 1 &&
        length(unique(counts$rows[grepl("test", counts$file)])) == 1) "PASS" else "FAIL")

tot_files <- counts$rows[1] + counts$rows[2]
say("Rows in train + test files: ", format(tot_files, big.mark = ","),
    " | expected (rows after Step 4", if (PILOT) ", pilot sample" else "", "): ", format(n_pre, big.mark = ","),
    " -> ", if (tot_files == n_pre) "PASS" else "FAIL")

n_unmatched <- function(a, b) sdf_sql(sc, sprintf(
  "SELECT COUNT(*) AS n FROM chk_%s x LEFT ANTI JOIN chk_%s y ON x.row_id = y.row_id", a, b)) %>% collect() %>% pull(n)
for (sp in c("train", "test")) {
  m1 <- n_unmatched(paste0("lyrics_", sp), paste0("audio_", sp))
  m2 <- n_unmatched(paste0("combined_", sp), paste0("audio_", sp))
  say("row_id alignment (", sp, "): lyrics rows missing from audio = ", m1,
      ", combined rows missing from audio = ", m2, " -> ", if (m1 == 0 && m2 == 0) "PASS" else "FAIL")
}

fe <- sdf_sql(sc, "SELECT split, emotion, COUNT(*) AS n FROM final_all GROUP BY split, emotion ORDER BY split, emotion") %>% collect()
say("Final emotion counts per split:"); say_df(fe)

# Close Spark, then copy files to data/processed
spark_disconnect(sc)
say("Copying the six files to ", FEATURE_DIR, " ...")
for (nm in names(out_specs)) {
  src <- file.path(WORK_DIR, paste0(nm, ".parquet"))
  dst <- file.path(FEATURE_DIR, paste0(nm, ".parquet"))
  if (identical(normalizePath(src, winslash = "/", mustWork = FALSE),
                normalizePath(dst, winslash = "/", mustWork = FALSE))) next
  unlink(dst, recursive = TRUE)
  ok <- file.copy(src, FEATURE_DIR, recursive = TRUE)
  if (!isTRUE(ok)) stop("Could not copy ", src, " to ", FEATURE_DIR)
}
say("Copy finished: ", paste(file.exists(file.path(FEATURE_DIR, paste0(names(out_specs), ".parquet"))), collapse = ", "))

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------
hdr("SUMMARY")
n_text <- if (USE_HASHING_FALLBACK) HASH_FEATURES else length(vocab)
say("Mode               : ", if (PILOT) "PILOT" else "FULL")
say("Salt used          : ", SALTS[chosen])
say("Rows: raw ", format(n_raw_rows, big.mark = ","), " -> dup_flag/emotion filter ", format(n_all, big.mark = ","),
    " -> unusable lyrics ", format(n_bad, big.mark = ","),
    " -> lyrics_dup_flag removed ", format(n_flag_removed, big.mark = ","),
    " -> cleaned-lyric duplicates removed ", format(n_resid_removed, big.mark = ","),
    " -> final ", format(n_dedup, big.mark = ","))
say("Train / test rows  : ", format(counts$rows[1], big.mark = ","), " / ", format(counts$rows[2], big.mark = ","))
say("Vector sizes       : audio = 9 | lyrics = ", n_text, " | combined = ", 10 + n_text,
    " (first 9 = audio, 10th = mean_similarity, rest = TF-IDF)")
say("Elapsed            : ", round(difftime(Sys.time(), T0, units = "mins"), 1), " minutes")
say("Feature files      : ", FEATURE_DIR)
say("Log and tables     : ", RES_DIR)
say("Temp folder (safe to delete): ", TMP_DIR)
