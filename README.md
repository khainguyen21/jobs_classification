# Job Career Level Classification

This project uses machine learning to guess the **career level** of a job (for example, "specialist" or "manager") by reading the job posting.

It is a practice project for learning **scikit-learn**.

## What is in this folder

| File | What it is |
|------|------------|
| `job_classification.py` | The main script. It loads the data, cleans it, turns text into numbers, and trains models. |
| `final_project.ods` | The dataset (a spreadsheet file). It has about 8,000 job postings. |
| `example.py` | A small test of the regular expression used to get the state code (like `CA`) from a location. |
| `README.md` | This file. |

## The data

Each row is one job posting. It has these columns:

- `title` – the job title
- `location` – the city and state, like `San Jose, CA`
- `description` – the full job description
- `function` – the job area (for example, sales or IT)
- `industry` – the company's industry
- `career_level` – **the answer we want to predict**

There are 6 career levels. The data is **not balanced** — some levels have many jobs and some have very few:

| Career level | Number of jobs |
|--------------|---------------:|
| senior_specialist_or_project_manager | 4,338 |
| manager_team_leader | 2,672 |
| bereichsleiter | 960 |
| director_business_unit_leader | 70 |
| specialist | 30 |
| managing_director_small_medium_company | 4 |

## How it works

1. **Load the data** from `final_project.ods`.
2. **Clean the location.** Only keep the two-letter state code (for example, `San Jose, CA` becomes `CA`).
3. **Remove rows with missing values.**
4. **Split the data** into 80% for training and 20% for testing. The split keeps the same mix of career levels in both parts.
5. **Turn text into numbers:**
   - `title`, `description`, and `industry` use **TF-IDF** (common English words like "the" are removed).
   - `location` and `function` use **One-Hot Encoding**.
6. **Pick the most useful features** with `SelectKBest` and the chi-squared test.
7. **Try many models at once** with **LazyPredict** to see which one works best.

## Earlier experiments

Inside the script there is older code (now commented out) that tried:

- A **Random Forest** model inside a scikit-learn `Pipeline`
- **GridSearchCV** to find good settings
- **SMOTEN** oversampling to help the small classes

The best Random Forest result was about **77% accuracy**. It worked well for the big classes, but it had trouble with the very small classes because there are so few examples of them.

## How to run it

1. Install Python 3.11.
2. Install the libraries:

   ```bash
   pip install pandas scikit-learn imbalanced-learn lazypredict odfpy
   ```

3. Run the script from this folder:

   ```bash
   python job_classification.py
   ```

The script prints information about the data and a table of model scores.
