"""
Market Basket Analysis: A Real-World Case Study
-----------------------------------------------
Question : What do electronics customers buy together?
Data     : 500 orders, 10 products (dirty_data.csv)
Technique: Apriori association rule mining (mlxtend)

Install requirements once:
    pip install pandas matplotlib mlxtend

Run:
    python market_basket_analysis.py
(Change DATA_PATH below if your CSV is stored somewhere else.
 In Google Colab it was "/content/dirty_data.csv".)
"""

import ast
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dirty_data.csv"
FIRST_SUPPORT = 0.05           # first try: 5% minimum support
FINAL_SUPPORT = 0.02           # lowered to 2% to get richer rules
MIN_CONFIDENCE = 0.30          # keep rules with confidence >= 30%
MIN_LIFT = 1.10                # keep rules with lift > 1.1

# The raw data has product names in different capitalisation ("Alcon 10",
# "ALCON 10", "alcon 10"), which gives 29 columns instead of 10.
#   CLEAN_NAMES = False -> same as the original notebook and the report
#                          (29 columns; results: 77 itemsets, top lift 1.52)
#   CLEAN_NAMES = True  -> merge the spelling variants into the 10 real
#                          products (results change slightly, e.g. top lift 1.42)
CLEAN_NAMES = False

# The 10 real products, used only when CLEAN_NAMES = True.
CANONICAL_PRODUCTS = [
    "Lucent 330S", "Thunder line", "Candle Inferno", "Olivia x460",
    "Universe Note", "pearTV", "Toshika 750", "iStream",
    "Alcon 10", "iAssist Line",
]
NAME_MAP = {name.lower().strip(): name for name in CANONICAL_PRODUCTS}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# STEP 1: INSPECT THE DATA
# ----------------------------------------------------------------------
section("STEP 1: Load and inspect the data")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Duplicate rows ---")
print("Number of duplicate rows:", df.duplicated().sum())

print("\nUnique orders   :", df["order_id"].nunique())
print("Unique customers:", df["customer_id"].nunique())

print("\nShopping cart examples (raw text):")
print(df["shopping_cart"].head(5).to_string(index=False))

# ----------------------------------------------------------------------
# STEP 2: PARSE THE SHOPPING CART
# ----------------------------------------------------------------------
section("STEP 2: Parse the shopping_cart column")

# shopping_cart is stored as TEXT that looks like a list of (product, qty)
# tuples. ast.literal_eval safely turns the text into a real Python list.
df["cart"] = df["shopping_cart"].apply(ast.literal_eval)

# Number of different products in each order
df["basket_size"] = df["cart"].apply(len)

print("Average products per order:", round(df["basket_size"].mean(), 3))
print("Minimum products in an order:", df["basket_size"].min())
print("Maximum products in an order:", df["basket_size"].max())
print("\nBasket size distribution:")
print(df["basket_size"].value_counts().sort_index())

# Keep only the product NAMES (association rules ignore quantity).
# If CLEAN_NAMES is True, capitalisation is also standardised.
def clean_name(name):
    if not CLEAN_NAMES:
        return str(name).strip()
    key = str(name).strip().lower()
    return NAME_MAP.get(key, str(name).strip())


df["products"] = df["cart"].apply(
    lambda cart: sorted({clean_name(item[0]) for item in cart})
)

print("\nExample processed baskets:")
for i in range(5):
    print(f"Order {df.loc[i, 'order_id']}: {df.loc[i, 'products']}")

# ----------------------------------------------------------------------
# STEP 3: PRODUCT POPULARITY
# ----------------------------------------------------------------------
section("STEP 3: Product popularity")

# One row per product per order, then count how many orders contain it.
product_counts = df.explode("products")["products"].value_counts()
product_share = (product_counts / len(df) * 100).round(1)

print("Number of orders containing each product:")
print(product_counts)
print("\nShare of orders (%):")
print(product_share)

plt.figure(figsize=(10, 5))
product_counts.sort_values().plot(kind="barh", color="#7C3AED")
plt.title("Top 10 Most Purchased Products")
plt.xlabel("Number of orders")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("chart_product_popularity.png", dpi=200)
plt.show()

plt.figure(figsize=(7, 5))
df["basket_size"].value_counts().sort_index().plot(
    kind="bar", color=["#7C3AED", "#06B6D4", "#F59E0B"]
)
plt.xlabel("Number of Products in Order")
plt.ylabel("Number of Orders")
plt.title("Distribution of Basket Size")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart_basket_size.png", dpi=200)
plt.show()

# ----------------------------------------------------------------------
# STEP 4: ONE-HOT ENCODING (order x product True/False matrix)
# ----------------------------------------------------------------------
section("STEP 4: Encode baskets as a True/False matrix")

te = TransactionEncoder()
te_array = te.fit(df["products"]).transform(df["products"])
basket = pd.DataFrame(te_array, columns=te.columns_)

print("Basket matrix shape:", basket.shape)
print(basket.head())

# Save the encoded matrix (1 row per order, 1 column per product)
basket.astype(int).to_csv("basket_matrix.csv", index=False)
print("Saved: basket_matrix.csv")

# ----------------------------------------------------------------------
# STEP 5: FREQUENT ITEMSETS AND RULES AT 5% SUPPORT
# ----------------------------------------------------------------------
section("STEP 5: Apriori at 5% minimum support")

frequent_itemsets = apriori(basket, min_support=FIRST_SUPPORT, use_colnames=True)
frequent_itemsets["itemset_size"] = frequent_itemsets["itemsets"].apply(len)

print("Number of frequent itemsets:", len(frequent_itemsets))
print(frequent_itemsets.sort_values(by="support", ascending=False).head(20))

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\nNumber of association rules:", len(rules))

if len(rules) > 0:
    strong_rules = rules.sort_values(by=["lift", "confidence"], ascending=False)
    print(strong_rules[["antecedents", "consequents", "support",
                        "confidence", "lift"]].head(15))
print("\nOnly a couple of rules appear, so we lower the support threshold.")

# ----------------------------------------------------------------------
# STEP 6: LOWER SUPPORT TO 2%
# ----------------------------------------------------------------------
section("STEP 6: Apriori at 2% minimum support")

frequent_itemsets_02 = apriori(basket, min_support=FINAL_SUPPORT, use_colnames=True)
frequent_itemsets_02["itemset_size"] = frequent_itemsets_02["itemsets"].apply(len)

print("Number of frequent itemsets:", len(frequent_itemsets_02))
print("\nItemsets by size:")
print(frequent_itemsets_02["itemset_size"].value_counts().sort_index())

rules_02 = association_rules(frequent_itemsets_02, metric="lift", min_threshold=1.0)
print("\nNumber of rules at 2% support:", len(rules_02))

strong_rules_02 = rules_02.sort_values(by=["lift", "confidence"], ascending=False)
print(strong_rules_02[["antecedents", "consequents", "support",
                       "confidence", "lift"]].head(20))


def to_text(x):
    """frozenset -> readable text, e.g. 'iStream, pearTV'."""
    return ", ".join(sorted(x))


# Save the full itemset and rule tables so they can be attached to the report
fi_out = frequent_itemsets_02.copy()
fi_out["itemsets"] = fi_out["itemsets"].apply(to_text)
fi_out.sort_values("support", ascending=False).to_csv(
    "frequent_itemsets_2pct.csv", index=False)

ar_out = strong_rules_02[["antecedents", "consequents", "support",
                          "confidence", "lift"]].copy()
ar_out["antecedents"] = ar_out["antecedents"].apply(to_text)
ar_out["consequents"] = ar_out["consequents"].apply(to_text)
ar_out.to_csv("all_association_rules_2pct.csv", index=False)
print("\nSaved: frequent_itemsets_2pct.csv, all_association_rules_2pct.csv")

# ----------------------------------------------------------------------
# STEP 7: SELECT USEFUL RULES
# ----------------------------------------------------------------------
section("STEP 7: Select useful rules (support, confidence, lift filters)")

selected_rules = rules_02[
    (rules_02["support"] >= FINAL_SUPPORT)
    & (rules_02["confidence"] >= MIN_CONFIDENCE)
    & (rules_02["lift"] > MIN_LIFT)
].copy()

selected_rules = selected_rules.sort_values(by=["lift", "confidence"], ascending=False)

# Convert frozensets into readable text, e.g. "iStream, pearTV -> Alcon 10"
selected_rules["Rule"] = (
    selected_rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    + " \u2192 "
    + selected_rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
)

result_table = selected_rules[["Rule", "support", "confidence", "lift"]].head(10).copy()
result_table["support"] = (result_table["support"] * 100).round(2)
result_table["confidence"] = (result_table["confidence"] * 100).round(2)
result_table["lift"] = result_table["lift"].round(2)
result_table.columns = ["Association Rule", "Support (%)", "Confidence (%)", "Lift"]

print(f"Rules kept after filtering: {len(selected_rules)}")
print(result_table.to_string(index=False))

result_table.to_csv("top_association_rules.csv", index=False)
print("\nSaved: top_association_rules.csv")

# ----------------------------------------------------------------------
# STEP 8: CHART OF TOP RULES BY LIFT
# ----------------------------------------------------------------------
section("STEP 8: Plot the top rules by lift")

plot_data = result_table.head(8).sort_values("Lift")

plt.figure(figsize=(10, 6))
plt.barh(plot_data["Association Rule"], plot_data["Lift"], color="#EC4899")
plt.axvline(1, color="gray", linestyle="--", linewidth=1)  # lift = 1 means chance
plt.xlabel("Lift")
plt.ylabel("Association Rule")
plt.title("Top Association Rules by Lift")
plt.tight_layout()
plt.savefig("chart_top_rules_lift.png", dpi=200)
plt.show()

# ----------------------------------------------------------------------
# STEP 9: BUSINESS INSIGHTS
# ----------------------------------------------------------------------
section("STEP 9: Business insights")

print("""
1. Demand is evenly spread: each product appears in about 26-32% of orders.
2. Pairs are weak (lift about 1.16); triples are stronger (lift up to 1.52).
3. Strongest rule: iStream + pearTV -> Alcon 10 (lift 1.52, confidence 41%).
4. Alcon 10 and pearTV are anchor products and appear in many top rules.

Recommendations:
  - Bundle iStream + pearTV with Alcon 10.
  - Suggest Toshika 750 when Alcon 10 and Candle Inferno / Thunder line
    are in the cart.
  - Use Alcon 10 and pearTV as anchor products in promotions.

Limitations:
  - Rules cover only about 2-3% of orders (roughly 10-16 orders).
  - Lift of 1.2 to 1.5 is moderate, not strong.
  - Association does not prove causation.
""")
