# Oscar Award Timing & Race Analysis
![Logo]https://github.com/o10199/Oscars_Awards_Python/blob/main/images/oscars.jpg
## Overview
The Academy Awards represent the pinnacle of achievement in filmmaking — but do all winners get there at the same time? This project examines whether an actor's racial or ethnic background affects **when** they receive an Oscar. Using linear regression on historical winner data spanning nearly 90 years, the analysis investigates potential systemic patterns in the timing of Oscar recognition across racial groups.

**Research Question:** Does race or ethnicity influence the year in which an actor receives an Oscar award?

---

## 🛠️ Tools & Technologies
- Python
- Pandas
- Statsmodels
- Google Colab

---

## 📂 Dataset
- **Source:** [Demographics of Academy Awards (Oscars) Winners – Kaggle](https://www.kaggle.com/datasets/fmejia21/demographics-of-academy-awards-oscars-winners)
- **Coverage:** Oscar winners from 1927 to 2014
- **Key Variables:**

| Variable | Description |
|---|---|
| year_of_award | The year the Oscar was awarded |
| race_ethnicity | Actor's racial or ethnic background |
| date_of_birth | Actor's date of birth |
| religion | Actor's religion |
| sexual_orientation | Actor's sexual orientation |
| birthplace | Actor's birthplace |

---

## 🗄️ Setup

```python
from google.colab import files
uploaded = files.upload()

import pandas as pd

df = pd.read_csv('Oscars-demographics-DFE 2.csv', encoding='latin-1')
df
```

---

## 🔍 Analysis & Solutions

### 1️⃣ Engineer Age at Award Variable
Derives the actor's age at the time of receiving the award from birth year and award year.

```python
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
df['birth_year'] = df['date_of_birth'].dt.year
df['age_at_award'] = df['year_of_award'] - df['birth_year']
```

### 2️⃣ Import Regression Libraries

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
```

### 3️⃣ Full Model — Multiple Demographic Predictors
Fits a regression model using race, religion, sexual orientation, birthplace, and date of birth to predict award year.

```python
model_ols_mul = ols('year_of_award ~ race_ethnicity + religion + sexual_orientation + birthplace + date_of_birth', data=df).fit()
```

### 4️⃣ Focused Model — Race/Ethnicity Only
Isolates race as the sole predictor to directly test its effect on award timing.

```python
model_ols = ols('year_of_award ~ C(race_ethnicity)', data=df).fit()

print(model_ols.summary())
```

---

## 📊 Key Findings

- Race is a **statistically significant** predictor of award timing overall (p = 0.00214, F-statistic: 3.819)
- However, the model explains only **4.2% of variance** in award year (R² = 0.042)
- No individual racial group showed a statistically significant effect on its own:

| Racial Group | Effect on Award Timing | p-value |
|---|---|---|
| Black | +5.3 years later | 0.691 |
| Hispanic | −11.4 years earlier | 0.433 |
| Middle Eastern | −5.5 years earlier | 0.835 |
| Multiracial | +2.0 years later | 0.922 |
| White | −18.6 years earlier | 0.119 |

---

## Conclusion
Race has a statistically significant but practically small effect on Oscar award timing, explaining just 4.2% of the variance. No specific racial group shows a statistically significant individual advantage or disadvantage. This analysis cannot establish causality — uncontrolled variables such as film genre, studio backing, and role availability likely play a substantial role. Future research should examine gender-race interactions, industry access factors, and the qualitative nature of roles available to different racial groups for a more complete picture.

---
