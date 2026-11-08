🛍️ Market Basket Analysis Using Apriori

Unlocking Hidden Purchase Patterns Through Association Rule Mining

<p align="center">Turning Transaction Data into Meaningful Business Insights

</p>---

📌 Overview

What products do customers tend to purchase together?

Market Basket Analysis is a data mining technique that uncovers relationships between products by analyzing customer transaction patterns.

This project explores an electronics e-commerce transaction dataset containing 500 orders and applies the Apriori Algorithm to discover frequent product combinations and meaningful association rules.

The goal is to transform raw transaction data into actionable insights that can support product recommendations, cross-selling strategies, and better business decisions.

🎯 Project Goals

- Discover frequently purchased product combinations.
- Identify hidden relationships between products.
- Generate meaningful association rules.
- Evaluate purchasing patterns using Support, Confidence, and Lift.
- Visualize findings through data-driven charts.
- Explore practical applications of market basket analysis in e-commerce.

⚙️ Technology Stack

Technology| Purpose
Python| Core programming
Pandas| Data manipulation
Matplotlib| Data visualization
Mlxtend| Apriori and association rules
Google Colab| Development and experimentation
VS Code| Local development

🔍 Analytical Workflow

1. Data Exploration – Inspect transaction records, columns, missing values, and duplicates.
2. Data Preprocessing – Parse shopping cart information and prepare product transactions.
3. Product Analysis – Examine product popularity and basket size distribution.
4. Transaction Encoding – Convert transactions into a suitable binary format.
5. Frequent Itemset Mining – Apply the Apriori algorithm to identify frequent combinations.
6. Association Rule Generation – Discover relationships between purchased products.
7. Visualization & Interpretation – Present and interpret the discovered patterns.

📊 Key Findings

<details>
<summary><strong>Dataset & Analysis</strong></summary>- Total Transactions: 500
- Unique Orders: 500
- Unique Customers: 494
- Minimum Support: 2%
- Minimum Confidence: 30%
- Minimum Lift: 1.10
- Frequent Itemsets: 77

</details><details>
<summary><strong>Strongest Reported Association Rule</strong></summary>iStream + pearTV → Alcon 10

Metric| Result
Support| 2.4%
Confidence| 41.38%
Lift| 1.52

This rule indicates a positive purchasing association within the analyzed transactions.

</details>📈 Visual Insights

The analysis generates three visualizations:

- Product Popularity: Identifies frequently purchased products.
- Basket Size Distribution: Examines the number of products per order.
- Top Association Rules: Highlights stronger product associations using Lift.



🚀 Getting Started

1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/Market-Basket-Analysis-Using-Apriori.git

2. Navigate to the Project

cd Market-Basket-Analysis-Using-Apriori

3. Install Dependencies

pip install -r requirements.txt

4. Add the Dataset

Place an authorized copy of the transaction dataset named "dirty_data.csv" inside the "data/" directory.

5. Run the Analysis

python srcs/market_basket_analysis.py

⚠️ Limitations & Considerations

- The analysis uses a relatively small dataset of 500 transactions.
- Product quantities are not considered during association rule mining.
- Inconsistent product capitalization affects itemset generation.
- Some discovered rules have relatively low support.
- Association does not imply causation.

💡 Real-World Applications

Market Basket Analysis can help businesses explore:

- Product recommendation systems.
- Cross-selling opportunities.
- Product bundling strategies.
- Customer purchasing behavior.
- Inventory and merchandising decisions.

🔮 Future Improvements

- Analyze larger transaction datasets.
- Standardize inconsistent product names.
- Compare Apriori with alternative association rule mining algorithms.
- Develop an interactive dashboard.
- Explore personalized product recommendation strategies.

---

<p align="center">Turning Data into Discoveries. One Transaction at a Time.

</p>
