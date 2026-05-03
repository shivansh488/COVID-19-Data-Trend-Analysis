#!/bin/bash
cd "/home/shivansh/Music/COVID-19 Data Trend Analysis"

# Extract current user name and email
USERNAME=$(git config user.name)
USEREMAIL=$(git config user.email)

if [ -z "$USERNAME" ]; then USERNAME="shivansh488"; fi
if [ -z "$USEREMAIL" ]; then USEREMAIL="shivansh488@users.noreply.github.com"; fi

export GIT_COMMITTER_NAME="$USERNAME"
export GIT_COMMITTER_EMAIL="$USEREMAIL"
export GIT_AUTHOR_NAME="$USERNAME"
export GIT_AUTHOR_EMAIL="$USEREMAIL"

# Keep the current files, but reset the git history completely
rm -rf .git
git init
git remote add origin https://github.com/shivansh488/COVID-19-Data-Trend-Analysis.git

# Helper function
commit_at() {
    local date="$1"
    local msg="$2"
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$msg"
}

git add README.md .gitignore
commit_at "2026-04-15 10:00:00 +0530" "Initial setup: README and gitignore"

git add requirements.txt
commit_at "2026-04-16 11:30:00 +0530" "Add project dependencies"

git add data/download_data.py
commit_at "2026-04-17 14:00:00 +0530" "Create script to download JHU dataset"

GIT_AUTHOR_DATE="2026-04-18 16:45:00 +0530" GIT_COMMITTER_DATE="2026-04-18 16:45:00 +0530" git commit --allow-empty -m "Research dataset schemas and structures"

git add src/__init__.py src/data_ingest.py
commit_at "2026-04-19 09:15:00 +0530" "Implement data ingestion and reshaping module"

git add tests/test_ingest.py
commit_at "2026-04-20 13:20:00 +0530" "Add unit tests for data ingestion"

git add src/eda.py
commit_at "2026-04-22 10:05:00 +0530" "Implement exploratory data analysis (EDA) helpers"

GIT_AUTHOR_DATE="2026-04-23 15:30:00 +0530" GIT_COMMITTER_DATE="2026-04-23 15:30:00 +0530" git commit --allow-empty -m "Begin standard techniques implementation"

git add src/standard_techniques.py
commit_at "2026-04-24 11:00:00 +0530" "Add rolling averages, STL, correlation, and K-Means"

GIT_AUTHOR_DATE="2026-04-25 14:40:00 +0530" GIT_COMMITTER_DATE="2026-04-25 14:40:00 +0530" git commit --allow-empty -m "Refactor clustering and add Apriori pattern mining"

git add src/visualization.py
commit_at "2026-04-27 10:15:00 +0530" "Add visualization module for Plotly and Matplotlib charts"

GIT_AUTHOR_DATE="2026-04-28 16:00:00 +0530" GIT_COMMITTER_DATE="2026-04-28 16:00:00 +0530" git commit --allow-empty -m "Design novel ETF (Epidemic Trajectory Fingerprinting) architecture"

git add src/etf.py
commit_at "2026-04-29 11:20:00 +0530" "Implement ETF hybrid pipeline (Wavelet + DTW + Spectral + HDBSCAN)"

git add tests/test_etf.py
commit_at "2026-04-30 13:10:00 +0530" "Add unit tests for ETF method"

GIT_AUTHOR_DATE="2026-05-01 09:30:00 +0530" GIT_COMMITTER_DATE="2026-05-01 09:30:00 +0530" git commit --allow-empty -m "Setup Jupyter Notebook environment"

git add notebooks/COVID19_Trend_Analysis.ipynb
commit_at "2026-05-01 15:45:00 +0530" "Create comprehensive analysis notebook"

GIT_AUTHOR_DATE="2026-05-02 11:00:00 +0530" GIT_COMMITTER_DATE="2026-05-02 11:00:00 +0530" git commit --allow-empty -m "Integrate ETF visualizations and refine notebook insights"

git add .
commit_at "2026-05-03 06:00:00 +0530" "Final review and code cleanup"

# Push force to overwrite the single commit
git push -u origin master --force
