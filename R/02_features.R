# 02_features.R
# Owner: Tam Yi Qing (Spark Feature Engineer)
# Purpose: Train/test split, TF-IDF, audio, lyrics and combined feature sets

# ---------------------------------------------------------------------
# 0. SWITCHES AND SETTINGS 
# ---------------------------------------------------------------------
PILOT        <- FALSE      # TRUE = 5% of artists, separate folder, scores not reported
PILOT_FRAC   <- 0.05
SPARK_VERSION <- "3.5.1"
DRIVER_MEM   <- "8G"      # if Spark stalls, use "6G"
CORES        <- 4         # if Spark stalls, use 2

BDA_DIR   <- "C:/Users/USER/OneDrive - ums.edu.my/Desktop/BDA/BDA"
IN_PATH   <- file.path(BDA_DIR, "data", "spotify_clean.parquet")
WORK_ROOT <- "C:/BDA_work"
OUT_DIR   <- if (PILOT) file.path(WORK_ROOT, "pilot") else WORK_ROOT
TMP_DIR   <- file.path(OUT_DIR, "_tmp")

SALTS     <- c("bda_stage2_v1", "bda_stage2_v2")  # second is used only if the first fails the check
TRAIN_PCT <- 80
TOL_PTS   <- 0.5          # class-share tolerance in percentage points
VOCAB_SIZE <- 10000
MIN_DF     <- 5
USE_HASHING_FALLBACK <- FALSE   # TRUE = HashingTF with 4096 features if memory fails
HASH_FEATURES <- 4096
EXPECTED_BASE <- 493820   # Harvey's model_ready view

AUDIO9 <- c("energy", "valence", "danceability", "tempo", "loudness",
            "speechiness", "liveness", "acousticness", "instrumentalness")
CLASSES <- c("anger", "fear", "joy", "love", "sadness", "surprise")  # label 0..5

# ---------------------------------------------------------------------
# 1. PACKAGES, LOGGING, ENVIRONMENT CHECK
# ---------------------------------------------------------------------
pk <- c("sparklyr", "dplyr", "readr")
miss <- pk[!pk %in% rownames(installed.packages())]
if (length(miss)) install.packages(miss)
suppressPackageStartupMessages({ library(sparklyr); library(dplyr) })

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TMP_DIR, recursive = TRUE, showWarnings = FALSE)
LOG <- file.path(OUT_DIR, "stage2_log.txt")
if (file.exists(LOG)) file.remove(LOG)
say  <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = LOG, append = TRUE) }
hdr  <- function(t) say("\n==== ", t, " ====")
say_df <- function(d) for (l in capture.output(print(as.data.frame(d), row.names = FALSE))) say(l)
reg <- function(x, name) invisible(sdf_register(x, name))   # register a Spark view quietly
T0 <- Sys.time()

hdr("STEP 1: environment")
say("Script version 3 (new bracket rule, lyric dedup before split, 9 audio features)")
say("Run mode: ", if (PILOT) "PILOT (5% of artists)" else "FULL")
say("Started : ", format(T0))

jv <- tryCatch(system2("java", "-version", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
if (is.null(jv)) stop("Java was not found on PATH. Install Temurin 17 and restart R.")
m <- regmatches(jv[1], regexec('version "([0-9]+)\\.?([0-9]*)', jv[1]))[[1]]
java_major <- as.integer(m[2]); if (java_major == 1L) java_major <- as.integer(m[3])
say("Java major version: ", java_major)
if (!java_major %in% c(8L, 11L, 17L))
  stop("Spark 3.5 needs Java 8, 11 or 17. Found Java ", java_major, ".")

if (Sys.getenv("JAVA_HOME") == "") {            # only set it if it clearly points to a real JDK
  jp <- Sys.which("java")
  if (nzchar(jp)) {
    cand <- dirname(dirname(normalizePath(jp, winslash = "/")))
    if (file.exists(file.path(cand, "release"))) {
      Sys.setenv(JAVA_HOME = cand); say("JAVA_HOME set to ", cand)
    }
  }
}

inst <- spark_installed_versions()$spark
if (!SPARK_VERSION %in% inst)
  stop("Spark ", SPARK_VERSION, " is not installed. Installed: ", paste(inst, collapse = ", "),
       ". Run spark_install('", SPARK_VERSION, "') or change SPARK_VERSION.")
say("Spark ", SPARK_VERSION, " found.")

# Windows needs winutils.exe + hadoop.dll, otherwise Spark fails with
# "UnsatisfiedLinkError ... NativeIO$Windows.access0" when it reads or writes files.
if (.Platform$OS.type == "windows") {
  hh <- Sys.getenv("HADOOP_HOME")
  if (hh == "" && file.exists("C:/hadoop/bin/winutils.exe")) { hh <- "C:/hadoop"; Sys.setenv(HADOOP_HOME = hh) }
  has_files <- hh != "" && file.exists(file.path(hh, "bin", "winutils.exe")) &&
    file.exists(file.path(hh, "bin", "hadoop.dll"))
  if (!has_files)
    stop("winutils.exe and/or hadoop.dll not found (HADOOP_HOME = '", hh, "'). ",
         "Put both files in C:/hadoop/bin, then restart R (Session > Restart R) and rerun.")
  Sys.setenv(PATH = paste(normalizePath(file.path(hh, "bin"), winslash = "\\"), Sys.getenv("PATH"), sep = ";"))
  say("HADOOP_HOME = ", hh, " (winutils.exe and hadoop.dll found)")
}

if (!file.exists(IN_PATH)) stop("Input not found: ", IN_PATH)

cfg <- spark_config()
cfg$`sparklyr.shell.driver-memory` <- DRIVER_MEM
cfg$sparklyr.cores.local           <- CORES
cfg$spark.sql.shuffle.partitions   <- 32
cfg$spark.local.dir                <- file.path(WORK_ROOT, "spark_tmp")
dir.create(cfg$spark.local.dir, recursive = TRUE, showWarnings = FALSE)
sc <- spark_connect(master = "local", version = SPARK_VERSION, config = cfg)
say("Spark connected. Driver memory ", DRIVER_MEM, ", cores ", CORES)

# ---------------------------------------------------------------------
# 2. READ THE PARQUET AND BUILD THE BASE TABLE
#    Harvey's filter: use_for_model = 1 (dup_flag = 0 and a valid emotion)
# ---------------------------------------------------------------------
hdr("STEP 2: read parquet and select columns")
in_norm <- normalizePath(IN_PATH, winslash = "/")
raw <- spark_read_parquet(sc, "raw_clean", in_norm, memory = FALSE)
cols <- colnames(raw)
say("Parquet has ", length(cols), " columns.")

need <- c("row_id", "emotion", "use_for_model", "mean_similarity", AUDIO9)
absent <- setdiff(need, cols)
lyrics_col <- intersect(c("lyrics", "text"), cols)[1]
artist_col <- intersect(c("first_artist", "artist"), cols)[1]
if (length(absent) || is.na(lyrics_col) || is.na(artist_col)) {
  say("Missing columns: ", paste(absent, collapse = ", "))
  say("Lyrics column found: ", lyrics_col, " | artist column found: ", artist_col)
  say("All columns: ", paste(cols, collapse = ", "))
  stop("Column names do not match the plan. Paste the 'All columns' line to Claude.")
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
         mean_similarity_d AS mean_similarity, lyrics
  FROM (
    SELECT CAST(row_id AS BIGINT) AS row_id,
           coalesce(lower(trim(CAST(%s AS STRING))), concat('__row_', CAST(row_id AS STRING))) AS artist_key,
           lower(trim(emotion)) AS emotion_c,
           %s,
           CAST(mean_similarity AS DOUBLE) AS mean_similarity_d,
           CAST(`%s` AS STRING) AS lyrics
    FROM raw_clean
    WHERE CAST(use_for_model AS INT) = 1
  ) WHERE emotion_c IN (%s)",
                        label_expr, paste(AUDIO9, collapse = ", "),
                        artist_expr, audio_cast, lyrics_col, cls_in)), "base_all")

n_all <- sdf_nrow(tbl(sc, "base_all"))
say("Rows with use_for_model = 1 and a valid emotion: ", format(n_all, big.mark = ","))
if (n_all != EXPECTED_BASE) say("WARNING: expected ", format(EXPECTED_BASE, big.mark = ","), " rows (Harvey's modelling set).")
n_dup_id <- sdf_sql(sc, "SELECT COUNT(*) - COUNT(DISTINCT row_id) AS d FROM base_all") %>% collect()
say("Duplicate row_id values: ", n_dup_id$d, " (must be 0)")
if (n_dup_id$d != 0) stop("row_id is not unique.")

# ---------------------------------------------------------------------
# 3. LYRICS CLEANING AND USABLE-LYRICS CHECK
# Cleaning chain (the same expression is used for hashing and for tokenising):
#   1a/1b strip [...] tags: one line, no inner bracket, at most 300 characters, run twice
#   2     curly apostrophes and backtick -> '
#   3     lower-case
#   4     keep letters a-z, apostrophes, spaces
#   5     drop apostrophes that are not inside a word (friends' -> friends, 'cause -> cause)
#   6     collapse whitespace
# ---------------------------------------------------------------------
rx <- function(e, pat, rep) sprintf("regexp_replace(%s, %s, %s)", e, pat, rep)
tag_pat <- r"---('\\[[^\\[\\]\\r\\n]{0,300}\\]')---"
e <- "coalesce(lyrics, '')"
e <- rx(e, tag_pat, "' '")                                        # 1a. strip tags
e <- rx(e, tag_pat, "' '")                                        # 1b. second pass (nested tags)
e <- rx(e, r"---('[\\u2019\\u2018\\u0060]')---", "\"'\"")        # 2. curly apostrophes -> '
e <- sprintf("lower(%s)", e)                                      # 3. lower-case
e <- rx(e, "\"[^a-z' ]\"", "' '")                                 # 4. keep letters, apostrophes, spaces
e <- rx(e, "\"(?<![a-z])'|'(?![a-z])\"", "' '")                  # 5. drop apostrophes not inside a word
e <- sprintf("trim(%s)", rx(e, r"---('\\s+')---", "' '"))         # 6. collapse whitespace

hdr("STEP 3: lyrics cleaning and usable-lyrics check")
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
    " (", round(100 * n_bad / n_all, 4), "% of the base). The old bracket rule removed 2 songs here.")
if (n_bad > 0) {
  bad_s <- sdf_sql(sc, "
    SELECT b.row_id, b.emotion, substr(b.lyrics, 1, 80) AS lyrics_start
    FROM base_all b JOIN lyr_clean_tbl f ON b.row_id = f.row_id
    WHERE f.lyr_status <> 'ok' LIMIT 10") %>% collect()
  say("First removed songs:"); say_df(bad_s)
  if (n_bad / n_all > 0.01) say("WARNING: more than 1% of rows have no usable lyrics. Check the cleaning rule.")
}

# ---------------------------------------------------------------------
# 4. REMOVE SONGS WITH IDENTICAL CLEANED LYRICS, BEFORE THE SPLIT
# Rule: songs with the same MD5 of the cleaned lyrics form a group; the song
# with the smallest row_id is kept, the others are dropped. Harvey's dup_flag
# (same artist and title) has already been applied in Step 2.
# ---------------------------------------------------------------------
hdr("STEP 4: duplicate lyrics removed BEFORE the split")
reg(sdf_sql(sc, "
  SELECT row_id, lyr_hash,
         ROW_NUMBER() OVER (PARTITION BY lyr_hash ORDER BY row_id) AS rn
  FROM lyr_clean_tbl WHERE lyr_status = 'ok'"), "dedup_rank")
reg(sdf_sql(sc, "SELECT row_id FROM dedup_rank WHERE rn = 1"), "keep_ids")

dg <- sdf_sql(sc, "
  SELECT COUNT(*) AS dup_groups,
         COALESCE(SUM(n - 1), 0) AS rows_removed,
         COALESCE(MAX(n), 0) AS largest_group,
         COALESCE(SUM(CASE WHEN n_emo > 1 THEN 1 ELSE 0 END), 0) AS groups_conflicting_emotion
  FROM (SELECT c.lyr_hash, COUNT(*) AS n, COUNT(DISTINCT b.emotion) AS n_emo
        FROM lyr_clean_tbl c JOIN base_all b ON c.row_id = b.row_id
        WHERE c.lyr_status = 'ok' GROUP BY c.lyr_hash HAVING COUNT(*) > 1)") %>% collect()
n_dup_removed <- as.numeric(dg$rows_removed)
say("Groups of identical cleaned lyrics: ", format(dg$dup_groups, big.mark = ","))
say("Songs removed (all but the smallest row_id in each group): ", format(n_dup_removed, big.mark = ","),
    " (", round(100 * n_dup_removed / (n_all - n_bad), 3), "% of the base)")
say("Largest group: ", dg$largest_group, " songs | groups whose songs carry different emotion labels: ",
    format(dg$groups_conflicting_emotion, big.mark = ","))

if (n_dup_removed > 0) {
  rem_emo <- sdf_sql(sc, "
    SELECT b.emotion, COUNT(*) AS removed FROM base_all b JOIN dedup_rank d ON b.row_id = d.row_id
    WHERE d.rn > 1 GROUP BY b.emotion") %>% collect()
  say("Removed songs by emotion:"); say_df(rem_emo %>% arrange(desc(removed)))
  top_g <- sdf_sql(sc, "
    SELECT COUNT(*) AS songs, MIN(substr(lyr_clean, 1, 70)) AS cleaned_lyrics_start
    FROM lyr_clean_tbl WHERE lyr_status = 'ok'
    GROUP BY lyr_hash HAVING COUNT(*) > 1 ORDER BY songs DESC LIMIT 5") %>% collect()
  say("Five largest duplicate groups (placeholder texts such as 'instrumental' collapse to one song):")
  say_df(top_g)
  if (n_dup_removed / (n_all - n_bad) > 0.05) say("WARNING: more than 5% of songs removed as duplicate lyrics. Look at why before going on.")
}

reg(sdf_sql(sc, sprintf("
  SELECT b.row_id, b.artist_key, b.emotion, b.label, %s, b.mean_similarity
  FROM base_all b JOIN keep_ids k ON b.row_id = k.row_id", paste0("b.", AUDIO9, collapse = ", "))), "base_dedup")
n_dedup <- sdf_nrow(tbl(sc, "base_dedup"))
say("Base rows after removals: ", format(n_dedup, big.mark = ","), " (", format(n_all, big.mark = ","),
    " minus ", format(n_bad, big.mark = ","), " unusable minus ", format(n_dup_removed, big.mark = ","), " duplicate lyrics)")
if (n_dedup != n_all - n_bad - n_dup_removed) stop("Row count after removals does not reconcile.")
say("Count table for Section 4.1: 551,443 raw -> ", format(n_all, big.mark = ","),
    " after Harvey's filter -> ", format(n_dedup, big.mark = ","), " after Stage 2 removals.")

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
# 5. ARTIST SPLIT WITH ACCEPTANCE CHECK
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
  say("  Check failed, changing the salt once (document this in Section 4.2).")
}
if (!PILOT && r$maxdev > TOL_PTS) say("WARNING: second salt also above tolerance. Report the deviation honestly.")
say("Salt used: ", SALTS[chosen])
say_df(r$tab %>% mutate(across(c(share, share_all, dev_pts), ~ round(.x, 2))))

by_split <- sdf_sql(sc, "SELECT split, COUNT(DISTINCT artist_key) AS artists, COUNT(*) AS songs FROM base_split GROUP BY split") %>% collect()
say_df(by_split)
say("Artist share in train : ", round(100 * by_split$artists[by_split$split == "train"] / sum(by_split$artists), 2), "%")
say("Row ratio (train/all) : ", round(100 * by_split$songs[by_split$split == "train"] / sum(by_split$songs), 2), "%  (this is the ratio actually obtained)")
leak <- sdf_sql(sc, "SELECT COUNT(*) AS n FROM (SELECT artist_key FROM base_split GROUP BY artist_key HAVING COUNT(DISTINCT split) > 1)") %>% collect()
say("Artists present in BOTH train and test: ", leak$n, " (must be 0)")
if (leak$n != 0) stop("Artist leakage in split.")

readr::write_csv(
  sdf_sql(sc, "SELECT artist_key, split, COUNT(*) AS songs FROM base_split GROUP BY artist_key, split") %>% collect(),
  file.path(OUT_DIR, "split_table.csv"))

# ---------------------------------------------------------------------
# 6. TOKENISING AND STOP WORDS (cleaned text comes from Step 3)
# ---------------------------------------------------------------------
hdr("STEP 6: tokenising and stop words")
reg(sdf_sql(sc, sprintf("
  SELECT s.row_id, s.artist_key, s.emotion, s.label, s.split, %s, s.mean_similarity, c.lyr_clean
  FROM base_split s JOIN lyr_clean_tbl c ON s.row_id = c.row_id",
                        paste0("s.", AUDIO9, collapse = ", "))), "cleaned")

stops_all <- unlist(ml_default_stop_words(sc, "english"))   # sparklyr returns a list; make it a character vector
neg <- stops_all[stops_all %in% c("no", "nor", "not", "cannot") | grepl("n't$", stops_all)]
STOPS <- setdiff(stops_all, neg)
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

# Sanity check: leftover [Verse]/[Chorus] words. The old rule gave 4.9% (full run).
# A large rise would mean the new rule leaves many tags in the text.
smp <- sdf_sql(sc, "SELECT emotion, concat_ws(' ', slice(tokens, 1, 25)) AS first_tokens FROM tok_all LIMIT 3") %>% collect()
say("Sample cleaned tokens (first 25 per song):")
for (i in seq_len(nrow(smp))) say("  [", smp$emotion[i], "] ", smp$first_tokens[i])
tagchk <- sdf_sql(sc, "
  SELECT SUM(CASE WHEN array_contains(tokens, 'verse') OR array_contains(tokens, 'chorus')
                       OR array_contains(tokens, 'bridge') THEN 1 ELSE 0 END) AS n_tag,
         COUNT(*) AS n FROM tok_all") %>% collect()
share_tag <- tagchk$n_tag / tagchk$n
say("Songs still containing the token verse, chorus or bridge: ", tagchk$n_tag, " (",
    round(100 * share_tag, 2), "%). Old rule: 4.9% (full run). Expect a similar value.")
if (share_tag > 0.10) stop("Tag stripping looks like it failed (over 10% of songs still contain verse/chorus/bridge). Paste the sample lines to Claude.")

# ---------------------------------------------------------------------
# 7. ZERO CHECK: no identical cleaned lyrics left, none across the boundary
# ---------------------------------------------------------------------
hdr("STEP 7: zero check for duplicate lyrics")
dchk <- sdf_sql(sc, "
  SELECT COUNT(*) - COUNT(DISTINCT c.lyr_hash) AS repeated_rows,
         (SELECT COUNT(*) FROM (SELECT c2.lyr_hash FROM base_split s2 JOIN lyr_clean_tbl c2 ON s2.row_id = c2.row_id
                                GROUP BY c2.lyr_hash HAVING COUNT(DISTINCT s2.split) > 1)) AS hashes_in_both_splits
  FROM base_split s JOIN lyr_clean_tbl c ON s.row_id = c.row_id") %>% collect()
say("Repeated cleaned-lyric rows left: ", dchk$repeated_rows, " (must be 0) | lyric hashes present in both train and test: ",
    dchk$hashes_in_both_splits, " (must be 0)")
if (dchk$repeated_rows != 0 || dchk$hashes_in_both_splits != 0) stop("Duplicate lyrics remain after Step 4.")

# ---------------------------------------------------------------------
# 8. MIN-MAX SCALING FITTED ON TRAIN ONLY
# ---------------------------------------------------------------------
hdr("STEP 8: min-max scaling (train only) with test clipped to [0, 1]")
src_expr <- c(setNames(AUDIO9, AUDIO9), mean_similarity = "mean_similarity")

chk <- paste(sprintf("SUM(CASE WHEN %s IS NULL THEN 1 ELSE 0 END) AS na_%s",
                     names(src_expr), names(src_expr)), collapse = ", ")
na_res <- sdf_sql(sc, paste("SELECT", chk, "FROM tok_all")) %>% collect()
say("NULL values in numeric features: ", sum(unlist(na_res)), " (must be 0 after Stage 1 imputation)")
if (sum(unlist(na_res)) > 0) stop("NULLs found in numeric features. Check Stage 1 output.")

q <- paste0("SELECT ",
            paste(sprintf("MIN(%1$s) AS mn_%2$s, MAX(%1$s) AS mx_%2$s", src_expr, names(src_expr)), collapse = ", "),
            " FROM tok_all WHERE split = 'train'")
mm_row <- sdf_sql(sc, q) %>% collect()
mm <- data.frame(feature = names(src_expr),
                 train_min = unlist(mm_row[paste0("mn_", names(src_expr))], use.names = FALSE),
                 train_max = unlist(mm_row[paste0("mx_", names(src_expr))], use.names = FALSE))
readr::write_csv(mm, file.path(OUT_DIR, "minmax_table.csv"))
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
# 9. TF-IDF FITTED ON TRAIN ONLY
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
  readr::write_csv(data.frame(index = seq_along(vocab) - 1L, term = vocab), file.path(OUT_DIR, "vocabulary.csv"))
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
# 10. FEATURE VECTORS, LABELS AND CLASS WEIGHTS
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
readr::write_csv(cls_cnt %>% select(label, emotion, n, weight), file.path(OUT_DIR, "class_weights.csv"))
w_expr <- paste0("CASE label ", paste(sprintf("WHEN %d.0 THEN %.15g", cls_cnt$label, cls_cnt$weight), collapse = " "), " END")

reg(assembled %>% select(row_id, artist_key, emotion, label, split, f_audio9, tfidf, f_combined), "assembled")
final <- sdf_sql(sc, sprintf("SELECT *, %s AS weight FROM assembled", w_expr))

FINAL_PATH <- file.path(TMP_DIR, "final_all.parquet")
say("Writing assembled vectors to ", FINAL_PATH, " ...")
spark_write_parquet(final, FINAL_PATH, mode = "overwrite")
fin <- spark_read_parquet(sc, "final_all", FINAL_PATH, memory = FALSE)

# ---------------------------------------------------------------------
# 11. WRITE THE SIX HANDOFF FILES AND VERIFY THEM
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
  p <- file.path(OUT_DIR, paste0(nm, ".parquet"))
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
say("Final emotion counts per split (use these numbers in the report):"); say_df(fe)

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------
hdr("SUMMARY")
n_text <- if (USE_HASHING_FALLBACK) HASH_FEATURES else length(vocab)
say("Mode               : ", if (PILOT) "PILOT" else "FULL")
say("Salt used          : ", SALTS[chosen])
say("Rows: raw 551,443 -> Harvey filter ", format(n_all, big.mark = ","),
    " -> unusable lyrics removed ", format(n_bad, big.mark = ","),
    " -> duplicate lyrics removed ", format(n_dup_removed, big.mark = ","),
    " -> final ", format(n_dedup, big.mark = ","))
say("Train / test rows  : ", format(counts$rows[1], big.mark = ","), " / ", format(counts$rows[2], big.mark = ","))
say("Vector sizes       : audio = 9 | lyrics = ", n_text, " | combined = ", 10 + n_text,
    " (first 9 = audio, 10th = mean_similarity, rest = TF-IDF)")
say("Elapsed            : ", round(difftime(Sys.time(), T0, units = "mins"), 1), " minutes")
say("Output folder      : ", OUT_DIR)
say("Temp folder _tmp can be deleted after the six files are handed off.")
say("Reminder: read these files with sparklyr::spark_read_parquet (they hold Spark vectors), not arrow.")
say("Tell teammates: artist_key and row_id are identifiers, not features.")
spark_disconnect(sc)
