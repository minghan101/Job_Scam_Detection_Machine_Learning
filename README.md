<img width="1050" height="525" alt="image" src="https://github.com/user-attachments/assets/9c15f486-d456-44e1-a5a3-0ddfb05a38a0" />


# Job Advertisement Fraud Detection Using Machine Learning  
## 🔍 Overview  
This project focuses on detecting fraudulent job postings using **machine learning techniques**. Fake job postings pose serious risks to job seekers, leading to scams and data theft. By leveraging **data science, NLP, and classification algorithms**, this project builds a model capable of distinguishing real job listings from fraudulent ones.

The project demonstrates my ability to handle **end-to-end data science workflows** including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

---

## 📊 Dataset  
- **Source:** Publicly available dataset on fraudulent job postings.  
- **Size:** ~18,000 job listings with labeled classes (fraudulent vs genuine).  
- **Features include:** job title, location, description, requirements, telecommuting flag, benefits, and more.

---

## 🛠️ Technologies & Skills Demonstrated  
- **Languages:** Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)  
- **Techniques:**  
  - Data Cleaning & Preprocessing  
  - Natural Language Processing (TF-IDF, text vectorization)  
  - Handling class imbalance (SMOTE/undersampling)  
  - Machine Learning Models: Logistic Regression, Random Forest, XGBoost, etc.  
  - Model evaluation: Confusion Matrix, ROC-AUC, Precision-Recall  

---
# Data Exploration 
## Identifying if Duplicate Entries are indicators of Fraudulent Job Ads

Repetitive or duplicated job advertisements may indicate fraudulent activity, as scammers often reuse templates across multiple postings to maximize reach with minimal effort.
Checking for recurring/ duplicated job advertisement could be associated with the job advertisement being fraudulent.
**Methodology.** Therefore, we can prepare the data & run a chi-square test to check for correlation between duplicated job postings and that job posting being fraudulent.

**Null Hypothesis**: There is no association between a job ad being duplicated and it being fraudulent. <br>
**Alternative Hypothesis**: Duplicated job ads are more likely to be fraudulent. <br>

If the p-value is below 0.05, we reject the null hypothesis and conclude that duplication is significantly associated with fraudulent postings.
Note: While duplication can be a useful signal, legitimate companies may also repost jobs. The goal is to determine whether duplication is more frequent among scams, not to assume all duplicates are fraudulent.

<img width="262" height="82" alt="image" src="https://github.com/user-attachments/assets/9b80b68a-8114-4e9c-b896-311815a70f1c" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/334780d4-cd52-4af7-ac07-29de5a1a53b2" />

We run a chi-square test to check if duplicates are correlated to a job ad being fraudulent.

Contingency Table: <br>
<img width="164" height="41" alt="image" src="https://github.com/user-attachments/assets/e579de31-53a5-45c8-8f33-07d36b96e57e" />

p-value: 0.05277121187974043

**Conclusion**
>p-value is 0.0527 (>0.05). We do not have sufficient evidence to reject the null hypothesis that there is no association between a job ad being duplicated and it being fraudulent.
---
## Identifying if Missing Values are indicators of fraud

An analysis of missing values across job advertisement fields reveals significant data gaps, particularly in key informational fields.
<img width="219" height="345" alt="image" src="https://github.com/user-attachments/assets/352814e0-8238-4bc3-87cc-59872aa637dc" />

Fields with high missingness include:
- **`salary_range`**: 15,012 missing (84.5% of all jobs)
- **`department`**: 11,553 missing
- **`required_education`**: 8,105 missing
- **`benefits`**: 7,196 missing

In contrast, core descriptive fields like `title`, `description` have no missing values, indicating they are consistently provided across postings.

This pattern suggests that while basic content is present, critical job details are frequently omitted. A behavior often seen in low-effort or potentially fraudulent postings. Scammers may intentionally leave out accountability-rich information (e.g., salary, education requirements) to avoid scrutiny or mislead applicants.

Notably, **fraudulent jobs show higher rates of missingness** in fields like `company_profile` (587 vs. 2,721 in legitimate jobs — but proportionally more common among scams) and `salary_range`, reinforcing the idea that **missingness itself can be a behavioral signal**.

> **Insight**: While not all incomplete ads are scams, **a high number of missing fields increases suspicion**. Our detection strategy should flag job ads with excessive missing information — especially when combined with other red flags like urgency or "no experience" language.

**Next Steps**: To perform qualitative analysis via Data Visualization of the sparsity of data in each fields and quantitative analysis by comparing the percentage of missing datas between legitimate and fraudulent data.
We will also run a chi-square test to check if the missing fields are independent of fraudulent job ads.
This is important as it helps us to identify the key indicators that would pose higher risk to job scams.

### Visualization
<img width="733" height="528" alt="image" src="https://github.com/user-attachments/assets/d1152b4f-dadc-48be-b66b-644712822866" />
<img width="733" height="528" alt="image" src="https://github.com/user-attachments/assets/66cb7fd5-0581-4f97-959e-933468a0f0a1" />
A visual comparison of missing data between fraudulent and non-fraudulent job ads reveals a striking pattern: **fraudulent postings exhibit significantly higher missingness in key fields**, particularly `company_profile`.

The heatmap below visualizes missing or empty values across job ad fields, highlighting systematic gaps in critical information. Fields such as `salary_range`, `department`, and `required_education` show high levels of incompleteness. This could be a hallmark of low-effort or potentially deceptive postings.

This pattern suggests that scammers often omit accountability-rich details to avoid scrutiny, making **missingness itself a potential red flag** for fraud detection.

However, not all missing fields are equally indicative of fraud. To identify the most **discriminative signals**, we need to move beyond visual inspection and perform a **quantitative analysis**:

1. **Compare missingness rates**: Compute the percentage of missing values in each field, separately for fraudulent and legitimate jobs.
2. **Statistical validation**: Conduct chi-square tests of independence to determine which missing fields are **significantly associated** with fraud.

This two-step approach ensures that our detection system is based on evidence, not just intuition. This focuses only on fields where missingness is a statistically meaningful indicator of scam behavior.

<img width="221" height="338" alt="image" src="https://github.com/user-attachments/assets/2f37c5d0-4f0c-45de-8d3f-9c04ce815ad3" />

Using quantitative analysis, it could be seen that fraudulent job advertisements tend to have more missing `company_profile`, `requirements`, `benefits`, `employment_type`, `required_experience`, `required_education`, `industry`, `function` fields. Hence, these are indicators/ red flag that the job advertisement is a scam. <br>

However, to assess whether this association is statistically significant, we perform a **Chi-Square Test of Independence**.

#### Hypotheses
- **Null Hypothesis:** There is no association between missing `<<FIELD_NAME>>` and a job ad being fraudulent.
- **Alternative Hypothesis:** Missing `<<FIELD_NAME>>` is associated with a higher likelihood of fraud.

**Conclusion**
>For the missing fields `comapny_profile`, `requirements`, `employment_type`, `required_experience`, `required_education`, `industry`, the p-value is far below 0.05, so we reject the null hypothesis. There is a statistically significant association between missing these fields and job ad fraud.

>This supports the use of missing {`comapny_profile`, `requirements`, `employment_type`, `required_experience`, `required_education`, `industry`} fields as a high-value red flag in scam detection systems. While not all such jobs are scams, this feature can help prioritize high-risk posts for review.
---
## Identifying Recurring Words and Phrases in Fraudulent Job Ads

This section aims to uncover **commonly used words and phrases** in fraudulent job advertisements by comparing their usage patterns against legitimate job postings. 

The goal is to:
- Identify **recurring linguistic patterns** in scam ads
- Quantify the **likelihood (probability)** of specific terms appearing in fraudulent vs. legitimate jobs
- Highlight **discriminative signals** that can serve as red flags for scam detection

By analyzing language at the word and phrase level, we seek to expose the "fingerprint" of job scams. Such as overuse of urgency, vague promises, or low-barrier entry requirements. That distinguish them from genuine opportunities.

### 📈 Key Findings:
> 1. The top phrases more common in fraudulent job ads are: `high school equivalent`, `fulltime entry level`, `high school diploma`
<img width="1030" height="589" alt="image" src="https://github.com/user-attachments/assets/9c518840-fd90-40ee-85b5-dc83b40066d9" />

## Results:
<img width="258" height="343" alt="image" src="https://github.com/user-attachments/assets/85d26ac7-e4a7-4d2b-8078-c66cae108a8f" />

These results reveal a **distinct pattern** in scam job postings:
- **Overuse of low-qualification language**: Phrases like `"high school equivalent"` and `"entry level"` suggest scammers target job seekers with minimal experience.
- **Repetition of basic job descriptors**: Terms like `"fulltime entry level"` appear disproportionately in scams, indicating the use of **generic templates**.
- **Focus on accessibility**: Scammers emphasize **low barriers to entry**, making roles appear open to a wide audience.

#### Notable Observations

- **No urgent or high-pay phrases in top results**: Surprisingly, commonly known red flags like `"work from home"`, `"earn money online"`, or `"apply now"` did not rank highly. This may indicate:
  - These phrases are also common in legitimate ads
  - Scammers are using more subtle language than expected (such as `'Full Time Not Applicable"` as a more subtle language than `"work from home"`)
- **Legitimate ads use more professional phrasing**: Phrases like `"excellent communication skills"` and `"written and verbal"` are more common in real jobs, suggesting scammers avoid detailed skill descriptions.
- **Negative differences** (e.g., `"have the ability"`, `"responsible for the"`) suggest these are strong indicators of legitimacy, not fraud.

#### Conclusion
While fraudulent job ads do not rely on flashy or exaggerated language, they consistently use simplified, low-barrier job descriptions,  particularly emphasizing entry-level roles and minimal education requirements.
This supports a detection strategy based on:
- **Template reuse** (e.g., repeated use of `"high school equivalent"`)
- **Missing specific skillsets** 


> 2. Fraudulent job advertisements tend to have higher probability of using these words: `earn`, `encouraged` , `oil`, `typing` , `houston` , `clerk`, `clerical`,`relocation`
This analysis identifies words that are significantly more common in fraudulent job advertisements compared to legitimate ones. By comparing the probability of word occurrence (`p_fraud` vs. `p_legit`), we uncover linguistic patterns that distinguish scams from genuine opportunities.
<img width="1030" height="589" alt="image" src="https://github.com/user-attachments/assets/efc7cd53-3238-4bb8-85ec-3372563ffde2" />

#### Results

<img width="323" height="332" alt="image" src="https://github.com/user-attachments/assets/a1449a82-cf70-4123-8078-0e39fe7d62db" />


#### Notable Patterns
1. **Low-Skill, Entry-Level Roles**  
   Words like `"clerk"`, `"clerical"`, `"typing"`, and `"administrative"` suggest scammers target entry-level job seekers with promises of easy work and fast income.

2. **Urgency and Persuasion**  
   High-ratio words like `"encouraged"`, `"participate"`, and `"apply"` possibly indicate emotional manipulation. This could be common in scams that pressure applicants to act quickly.

3. **Geographic Targeting**  
   `"Houston"` appears disproportionately in fraud ads, possibly indicating localized scam campaigns or fake postings targeting specific regions.

4. **Misuse of Corporate Jargon**  
   Terms like `"corporate"`, `"leveraging"`, `"staffing"`, and `"relocation"` are used in a vague, buzzword-like manner which mimicks professional language without substance.
   
5. **Income Promises**  
   `"Earn"`, `"bonuses"`, and `"benefits"` are more frequent in scams, often paired with overpromising.

6. **Suspicious Application Methods**  
   `"Email"` and `"via"` may indicate **non-standard application processes** (e.g., “email your resume to...”).

#### Surprising Observations

- `"Benefits"` and `"payroll"` are more common in scams. This is possibly used to create false legitimacy.
- `"Oil"`, `"energy"`, `"gas"`. these reflect template reuse from real job ads in the energy sector, but are likely copy-pasted into scam postings.
- `"Per"`, `"needed"`, `"duties"`. These are generic words with high frequency in scams, suggesting template-based writing.

#### Conclusion

While no single word is a definitive scam signal, the **combination** of:
- Low-skill terms (`clerk`, `typing`)
- Emotional cues (`encouraged`, `participate`)
- Income promises (`earn`, `bonuses`)
- Vague corporate language (`leveraging`, `staffing`)
forms a **distinct linguistic fingerprint** of job scams.

As part of identifying scam job advertisements,  it is recommended to use these high-difference, high-ratio words as red flags in a rule-based or scoring system. Prioritize words with both high `diff` (>0.08) and high `ratio` (>5), such as `"clerk"`, `"typing"`, `"earn"`, and `"encouraged"`.

---

## 🚀 How to Run  

📂 Repository Structure
├── data/               # Dataset (not uploaded due to size, add link instead)
├── Jobs_Fraud_Detection_ML_Model.ipynb
├── requirements.txt
├── README.Rmd

### Clone the repository  
   ```bash
   git clone https://github.com/yourusername/job-fraud-detection.git
   cd job-fraud-detection
  ```

### Install dependencies  
```bash
pip install -r requirements.txt
```

Run the notebook or script
jupyter notebook Jobs_Fraud_Detection_ML_Model.ipynb

## Data Preparation for Model Training

To build a reliable fraud detection model, the dataset was first cleaned using the `clean_df` function.

After cleaning, the features (`X`) and target (`y`) were separated:
- `X`: All job ad features **except** the `fraudulent` label (to prevent data leakage)
- `y`: Converted binary target where `'t'` => 1 (fraudulent), `'f'` => 0 (legitimate)

The data was then split into training and testing sets:
- **80% training**, **20% testing**
- **Stratified sampling** was used to preserve the proportion of fraudulent jobs (~4.8%) in both sets
- A fixed `random_state=42` ensures reproducibility

Finally, the cleaned text (`text_clean`) was extracted for use in text vectorization:
- `X_train_text`: Cleaned job ads for model training
- `X_test_text`: Cleaned job ads for evaluation

This process ensures the model learns from realistic patterns in the data and is evaluated fairly on unseen examples.

## Feature Engineering

To build a robust fraud detection model, we combined **text-based features** with **domain-driven signals** , both derived from exploratory analysis.

Earlier, we identified **red_flag** features such as 
- **Missing Fields**. Having missing `significant_fields` correlates to fraudulent scam job ads
- **Top 30 Words & Phrases** most commonly used in scam job advertisements.
Therefore, as part of feature engineering, we will be introducing the `create_custom_features()` function which will incorporate these red flags into our dataset for training.

#### **Methodology**
We used `TfidfVectorizer` to convert job ad text into numerical features:
- **N-grams (1–2 words)**. Captures both individual terms and common phrases (e.g., "work from home")
- **Stop words removed**. Improves signal-to-noise ratio

The vectorizer was **fitted only on training data** (`X_train_text`) to prevent data leakage

## 📈 Key Model Performance Results

<img width="360" height="311" alt="image" src="https://github.com/user-attachments/assets/32717246-78e1-4d34-b2ca-2caa0933a044" />


To assess the performance of the trained Logistic Regression model, we evaluated its predictions on the unseen test set (`X_test_combined`). The evaluation focused on metrics that are meaningful for fraud detection, where the goal is to **catch as many scams as possible** while minimizing false alarms.

#### Key Metrics
- **Precision (Fraudulent)**: **56.6%**  
When the model flags a job ad as fraudulent, it is correct about 57% of the time. This means roughly 4 in 10 flagged jobs are legitimate (false positives).
  
- **Recall (Fraudulent)**: **89.6%**  
The model successfully identifies nearly 90% of actual scam job ads, which is critical for protecting job seekers.

- **F1-Score (Fraudulent)**: **0.69**  
A balanced measure of precision and recall, indicating strong overall performance despite class imbalance.

- **ROC-AUC**: **0.987** 
Receiver Operating Characteristic - Area Under Curve
Exceptional ability to distinguish between fraudulent and legitimate job ads.
i.e. If 1 fraudulent job and 1 normal job is randomly picked, there’s a 98% chance the model assigns a higher fraud probability to the fraudulent job.

## Additional Findings
We also find out that these are the top features driving fraud prediction: <br>

<img width="403" height="460" alt="image" src="https://github.com/user-attachments/assets/fcb6960e-1040-41f2-8f65-6ac8c4aa0d6d" />


## 💡 Key Learnings

Tackled imbalanced classification problems.
Gained experience in text preprocessing for NLP tasks.
Understood the trade-offs between recall, precision, and accuracy in fraud detection.
Enhanced ability to communicate machine learning results through visualization.

📌 Future Work

Experiment with deep learning models (LSTM, BERT) for text-based features.

Deploy the model as a web app (Streamlit/Flask) for real-time predictions.

Implement explainable AI (SHAP/LIME) for model interpretability.
