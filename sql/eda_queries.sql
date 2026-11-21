-- eda_queries.sql
-- Owner: Ong Chia Yin (Data Scientist)
-- Purpose: Exploratory data analysis (SQL queries used by 03_eda.R)
-- AI-assisted (Claude), 2026-11-20


-- ---------------------------------------------------------------
-- SECTION 1: DATA CHECKS
-- ---------------------------------------------------------------

-- name: count_songs
-- Every cleaned record, before any filtering.
SELECT COUNT(*) AS n
FROM songs;

-- name: count_model_ready
-- Rows left after removing duplicates and invalid labels.
SELECT COUNT(*) AS n
FROM model_ready;

-- name: count_use_for_model
-- The use_for_model flag should mark exactly the same rows as the view.
SELECT COUNT(*) AS n
FROM songs
WHERE use_for_model = 1;

-- name: emotion_counts
-- Amount of songs that carry each emotion label, and their share of the total.
-- SUM(COUNT(*)) OVER () is the grand total, so pct adds up to 100.
-- Feeds Plot 1.
SELECT emotion,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM model_ready
GROUP BY emotion
ORDER BY n DESC;


-- ---------------------------------------------------------------
-- SECTION 2: AUDIO FEATURES BY EMOTION
-- ---------------------------------------------------------------

-- name: audio_means_by_emotion
-- Average of each audio feature per emotion, for the report table and
-- for checking what the plots show. Most features are on a 0-1 scale;
-- tempo is in BPM, loudness in dB, popularity 0-100, and length in seconds.
SELECT emotion,
       COUNT(*)                         AS n,
       ROUND(AVG(valence), 3)           AS valence,
       ROUND(AVG(energy), 3)            AS energy,
       ROUND(AVG(danceability), 3)      AS danceability,
       ROUND(AVG(acousticness), 3)      AS acousticness,
       ROUND(AVG(speechiness), 3)       AS speechiness,
       ROUND(AVG(liveness), 3)          AS liveness,
       ROUND(AVG(instrumentalness), 3)  AS instrumentalness,
       ROUND(AVG(tempo), 1)             AS tempo,
       ROUND(AVG(loudness), 2)          AS loudness,
       ROUND(AVG(popularity), 1)        AS popularity,
       ROUND(AVG(length_sec), 1)        AS length_sec
FROM model_ready
GROUP BY emotion
ORDER BY n DESC;

-- name: valence_by_emotion
-- One row per song: its emotion and valence (musical positiveness, 0-1).
-- Feeds Plot 2.
SELECT emotion,
       valence
FROM model_ready;

-- name: audio_features
-- The 9 audio features used by the audio-only model, for every song.
-- Feeds the correlation matrix (Plot 3). 
-- Popularity and length are left out because the feature engineering stage doesn't use them.
SELECT energy, loudness, acousticness, valence, danceability, tempo,
       speechiness, liveness, instrumentalness
FROM model_ready;


-- ---------------------------------------------------------------
-- SECTION 3: SIMILAR-SONGS SCORE BY EMOTION (SEMI-STRUCTURED DATA)
-- ---------------------------------------------------------------

-- name: similarity_summary_by_emotion
-- Summary of the similar-songs score for each emotion (used for Plot 6).
-- mean_similarity = the average of the 3 similarity scores listed for each
-- song in the dataset's "Similar Songs" field (0-1, higher = more similar).
-- below_090 = songs scoring below 0.90, which fall outside Plot 6's axis.
SELECT emotion,
       COUNT(*)                                  AS n,
       ROUND(AVG(mean_similarity), 4)            AS avg_similarity,
       ROUND(MIN(mean_similarity), 4)            AS min_similarity,
       ROUND(MAX(mean_similarity), 4)            AS max_similarity,
       SUM(CASE WHEN mean_similarity < 0.90 THEN 1 ELSE 0 END) AS below_090
FROM model_ready
GROUP BY emotion
ORDER BY n DESC;

-- name: similarity_by_emotion
-- One row per song: its emotion and mean similarity score. Feeds Plot 6.
SELECT emotion,
       mean_similarity
FROM model_ready;
