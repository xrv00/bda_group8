# bda_group8
KD44103 Big Data Analytics | Group 8 | Semester 1, 2026/2027
Universiti Malaysia Sabah

## Overview
This project predicts the emotion of a song using audio features, lyrics, and both combined, and compares the three approaches.

- Dataset: Kaggle "500K+ Spotify Songs with Lyrics, Emotions & More"
- Target: emotion (6 classes)
- Tools: R, sparklyr (Spark MLlib), SQLite, ggplot2, R Markdown

## Team
| Name | Role |
|---|---|
| Harvey Maxwell Henry | Data Engineer |
| Tam Yi Qing | Spark Feature Engineer, Naive Bayes|
| Ong Chia Yin | Data Scientist, Logistic Regression |
| Voo Xin Rou | Integration and Evaluation Lead, Random Forest |

## Repository Structure
| Path | Contents |
|---|---|
| R/01_ingest.R | Download, parse JSONL, clean, write parquet and SQLite |
| R/02_features.R | Train/test split, TF-IDF, audio, lyrics and combined feature sets |
| R/03_eda.R | SQL queries, plots 1 to 3 and 6 |
| R/04_model_naive_bayes.R | Naive Bayes on the 3 feature sets |
| R/05_model_logistic.R | Logistic regression on the 3 feature sets |
| R/06_model_random_forest.R | Random forest on the 3 feature sets |
| R/07_evaluate.R | Metrics, majority-class baseline, confusion-matrix plots, macro-F1 plot |
| R/eval_utils.R | Shared evaluation functions (macro F1, confusion matrix) |
| sql/ | SQL queries and views |
| outputs/ | Predictions (CSV), metrics (CSV), and plots (PNG) |
| report/ | Dashboard and report files |
| data/ | Local data folders (not stored on GitHub) |



## Setup
1. On this GitHub page, click **Code**, then **Download ZIP**.
2. Unzip it on your laptop.
3. Open RStudio and go to File, New Project, Existing Directory, then select the unzipped folder. This makes the project folder the working directory, so relative paths work.
4. Install the requirements below.
5. Get the data (see **Data** section) and place it in the folders shown.
6. Follow **How to Run**.

Alternatively, if you use Git: `git clone https://github.com/xrv00/bda_group8.git`

## Requirements
- R (4.1 or newer) and RStudio
- Java (needed by Spark)
- R packages: tidyverse, sparklyr, DBI, RSQLite, ggplot2, rmarkdown
- **Windows only (Spark/Hadoop):** place `winutils.exe` and `hadoop.dll` in `C:\hadoop\bin`, or set `HADOOP_HOME` to a folder whose `bin\` contains both files. Both files are required; incomplete setups can cause confusing Spark/Hadoop native-library errors.
  
## Data
**Raw data (Kaggle):**
"500K+ Spotify Songs with Lyrics, Emotions & More"
<(https://www.kaggle.com/datasets/devdope/900k-spotify)>

Download it with the script `R/01_ingest.R`, or manually from the link above, and place the file in `data/raw/`.
A free Kaggle account and API token (`kaggle.json`) are needed for the scripted download.

**Processed data (Google Drive):**
link to the data files: 
<(https://drive.google.com/drive/folders/1URD0xl4ZbSFGnHpN5ruMKUgx-iZquKyl?usp=drive_link)>
- `stage1_cleaned/` (parquet and SQLite) goes into `data/processed/`
- `stage2_features/` (6 train/test parquet files) goes into `data/processed/`
If you run the scripts from the start, you do not need the Drive files, because they are regenerated.

## Outputs
| File | Location | Produced by |
|---|---|---|
| spotify_clean.parquet, spotify_structured.sqlite | data/processed/ (Drive) | Data Engineer |
| Ingestion log, data dictionary | report/ | Data Engineer |
| audio, lyrics, combined train/test parquet files (6) | data/processed/ (Drive) | Feature Engineer |
| predictions_naive_bayes.csv | outputs/ | Feature Engineer |
| predictions_logistic_regression.csv | outputs/ | Data Scientist |
| predictions_random_forest.csv | outputs/ | Integration Lead |
| plot1 to plot3, plot6 | outputs/ | Data Scientist |
| plot4_confusion.png, plot4b_confusion_by_featureset.png, plot4_confusion_all.png, plot5_macro_f1.png | outputs/ | Integration Lead |
| dashboard.Rmd, final_dashboard.html | report/ | Data Scientist |

## How to Run
Run from the project root, in this order:
1. `R/01_ingest.R`
2. `R/02_features.R`
3. `R/03_eda.R`
4. `R/04_model_naive_bayes.R`
5. `R/05_model_logistic.R`
6. `R/06_model_random_forest.R`
7. `R/07_evaluate.R`
8. Knit `report/dashboard.Rmd` to produce `final_dashboard.html`

## AI Usage
AI-assisted code is marked with `# AI-assisted` comments in the scripts.

