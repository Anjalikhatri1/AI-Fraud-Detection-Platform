import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

def _random_date(n):
    start = date(2024, 10, 1)
    return [start + timedelta(days=int(x)) for x in np.random.randint(0, 180, n)]

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_transactions(n=10000):
    """
    Generate synthetic Indian financial transaction data.
    Simulates UPI, NEFT, RTGS, IMPS payment flows typical in
    the Indian banking ecosystem.
    """
    payment_methods = ["UPI", "NEFT", "RTGS", "IMPS", "Debit Card", "Credit Card"]
    devices = ["Mobile", "Desktop", "Tablet", "ATM"]
    states = [
        "Maharashtra", "Karnataka", "Delhi", "Tamil Nadu",
        "Telangana", "Gujarat", "Rajasthan", "West Bengal",
        "Uttar Pradesh", "Punjab"
    ]

    records = []
    dates = _random_date(n)
    for i in range(n):
        is_fraud = 1 if np.random.random() < 0.08 else 0  # ~8% fraud rate

        amount = (
            np.random.exponential(scale=5000) + 50000
            if is_fraud
            else np.random.exponential(scale=2000) + 100
        )
        amount = round(min(amount, 500000), 2)

        hour = np.random.choice(range(24), p=_hour_weights(is_fraud))
        age = np.random.randint(18, 75)

        records.append({
            "transaction_id": f"TXN{1000000 + i}",
            "user_id": f"USR{np.random.randint(1000, 9999)}",
            "age": age,
            "amount": amount,
            "payment_method": np.random.choice(payment_methods),
            "device": np.random.choice(devices, p=[0.55, 0.25, 0.10, 0.10]),
            "state": np.random.choice(states),
            "hour_of_day": hour,
            "transaction_date": dates[i],
            "is_new_payee": np.random.choice([0, 1], p=[0.7, 0.3]),
            "login_attempts": np.random.randint(1, 6) if is_fraud else np.random.randint(1, 3),
            "velocity_1h": np.random.randint(1, 15) if is_fraud else np.random.randint(1, 5),
            "is_fraud": is_fraud,
        })

    return pd.DataFrame(records)


def _hour_weights(is_fraud):
    """Fraud peaks at late night; legit traffic peaks at business hours."""
    weights = np.ones(24)
    if is_fraud:
        weights[0:5] = 4.0
        weights[22:24] = 3.0
    else:
        weights[9:18] = 2.5
    return weights / weights.sum()


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    """Encode categorical variables and create ML-ready feature matrix."""
    df = df.copy()
    le = LabelEncoder()
    for col in ["payment_method", "device", "state"]:
        df[col + "_enc"] = le.fit_transform(df[col])

    df["is_night_txn"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df["high_amount_flag"] = (df["amount"] > 20000).astype(int)
    df["risk_indicator"] = (
        df["is_night_txn"] + df["is_new_payee"] + df["high_amount_flag"]
    )

    feature_cols = [
        "age", "amount", "payment_method_enc", "device_enc", "state_enc",
        "hour_of_day", "is_new_payee", "login_attempts", "velocity_1h",
        "is_night_txn", "high_amount_flag", "risk_indicator",
    ]
    return df, feature_cols


# ─────────────────────────────────────────────
# 3. ANOMALY DETECTION — ISOLATION FOREST
# ─────────────────────────────────────────────

def run_isolation_forest(df, feature_cols):
    """
    Unsupervised anomaly detection.
    Flags unusual transaction patterns without needing labelled fraud data.
    Mirrors the AI scoring API integration described in the Power Platform architecture.
    """
    print("\n[1/4] Running Isolation Forest Anomaly Detection...")
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
    df["anomaly_flag"] = iso.fit_predict(X_scaled)
    df["anomaly_score"] = -iso.score_samples(X_scaled)          # higher = more anomalous
    df["anomaly_flag"] = (df["anomaly_flag"] == -1).astype(int)

    anomaly_precision = (
        df[df["anomaly_flag"] == 1]["is_fraud"].sum()
        / df["anomaly_flag"].sum() * 100
    )
    print(f"   Anomalies detected  : {df['anomaly_flag'].sum():,}")
    print(f"   Fraud precision     : {anomaly_precision:.1f}%")
    return df


# ─────────────────────────────────────────────
# 4. FRAUD PROBABILITY — LOGISTIC REGRESSION
# ─────────────────────────────────────────────

def run_logistic_regression(df, feature_cols):
    """
    Supervised fraud probability model.
    Outputs a 0-1 probability per transaction — this is the 'Risk Score'
    fed back into Power Apps via REST API.
    """
    print("\n[2/4] Training Logistic Regression Fraud Classifier...")
    X = df[feature_cols].fillna(0)
    y = df["is_fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    df["fraud_probability"] = model.predict_proba(X_scaled)[:, 1]

    # Evaluation
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n   Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"   AUC-ROC Score: {auc:.4f}")

    return df, model


# ─────────────────────────────────────────────
# 5. RISK SCORE COMPUTATION
# ─────────────────────────────────────────────

def compute_risk_score(df):
    """
    Composite risk score (0–100) combining:
    - Fraud probability from ML model
    - Anomaly score from Isolation Forest
    - Rule-based risk indicators (velocity, night, new payee)

    Maps to the Risk Score field in the Dataverse Transaction table.
    """
    print("\n[3/4] Computing composite risk scores...")
    df["risk_score"] = (
        (df["fraud_probability"] * 60)
        + (df["anomaly_score"] / df["anomaly_score"].max() * 25)
        + (df["risk_indicator"] / 3 * 15)
    ).clip(0, 100).round(2)

    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[0, 30, 60, 80, 100],
        labels=["Low", "Medium", "High", "Critical"],
    )
    cat_dist = df["risk_category"].value_counts()
    print("   Risk Category Distribution:")
    for cat, cnt in cat_dist.items():
        print(f"     {cat:10s}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    return df


# ─────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_all(df, output_dir="screenshots"):
    """Generate all output charts and dashboard screenshot."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[4/4] Generating visualizations → /{output_dir}/")

    palette = {
        "primary":   "#1A56DB",
        "danger":    "#E02424",
        "warning":   "#FF8000",
        "success":   "#057A55",
        "neutral":   "#6B7280",
        "bg":        "#F9FAFB",
        "card":      "#FFFFFF",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": palette["bg"],
        "axes.facecolor": palette["card"],
    })

    # ── Chart 1: Distribution of Transaction Amounts ──────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[df["is_fraud"] == 0]["amount"], bins=60, alpha=0.7,
            color=palette["primary"], label="Legitimate", edgecolor="white")
    ax.hist(df[df["is_fraud"] == 1]["amount"], bins=60, alpha=0.75,
            color=palette["danger"], label="Fraudulent", edgecolor="white")
    ax.set_xlabel("Transaction Amount (₹)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Transaction Amounts", fontsize=14, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Distribution of Transaction Amounts.png", dpi=150)
    plt.close(fig)
    print("   ✓ Distribution of Transaction Amounts.png")

    # ── Chart 2: Daily Average Transaction Amount ─────────────────────
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    daily = df.groupby("transaction_date")["amount"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["transaction_date"], daily["amount"],
            color=palette["primary"], linewidth=2)
    ax.fill_between(daily["transaction_date"], daily["amount"],
                    alpha=0.15, color=palette["primary"])
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Avg Transaction Amount (₹)", fontsize=12)
    ax.set_title("Daily Average Transaction Amount", fontsize=14, fontweight="bold", pad=15)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Daily Average Transaction Amount.png", dpi=150)
    plt.close(fig)
    print("   ✓ Daily Average Transaction Amount.png")

    # ── Chart 3: Risk Score Distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_map = {"Low": palette["success"], "Medium": palette["warning"],
                  "High": "#F05252", "Critical": palette["danger"]}
    for cat, color in colors_map.items():
        subset = df[df["risk_category"] == cat]["risk_score"]
        ax.hist(subset, bins=30, alpha=0.8, color=color, label=cat, edgecolor="white")
    ax.set_xlabel("Risk Score (0–100)", fontsize=12)
    ax.set_ylabel("Number of Transactions", fontsize=12)
    ax.set_title("Risk Score Distribution by Category", fontsize=14, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Risk Score Distribution.png", dpi=150)
    plt.close(fig)
    print("   ✓ Risk Score Distribution.png")

    # ── Chart 4: Fraud by Payment Method ─────────────────────────────
    fraud_pm = df.groupby("payment_method")["is_fraud"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(fraud_pm.index, fraud_pm.values, color=palette["primary"],
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, fraud_pm.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Payment Method", fontsize=12)
    ax.set_ylabel("Fraud Rate (%)", fontsize=12)
    ax.set_title("Fraud Rate by Payment Method", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Fraud Rate by Payment Method.png", dpi=150)
    plt.close(fig)
    print("   ✓ Fraud Rate by Payment Method.png")

    # ── Dashboard Screenshot ──────────────────────────────────────────
    _generate_dashboard(df, palette, output_dir)


def _generate_dashboard(df, palette, output_dir):
    """Compose a multi-panel Power BI-style dashboard screenshot."""
    fig = plt.figure(figsize=(18, 12), facecolor="#1E293B")
    fig.suptitle(
        "AI-Powered Financial Fraud Detection & Risk Scoring Platform",
        fontsize=18, fontweight="bold", color="white", y=0.97
    )

    # KPI row
    total = len(df)
    fraud_count = df["is_fraud"].sum()
    high_risk = (df["risk_category"].isin(["High", "Critical"])).sum()
    avg_risk = df["risk_score"].mean()

    kpis = [
        ("Total Transactions", f"{total:,}", palette["primary"]),
        ("Fraud Detected",     f"{fraud_count:,}", palette["danger"]),
        ("High / Critical Risk", f"{high_risk:,}", "#FF8000"),
        ("Avg Risk Score",    f"{avg_risk:.1f} / 100", "#A855F7"),
    ]

    for i, (label, value, color) in enumerate(kpis):
        ax_kpi = fig.add_axes([0.02 + i * 0.245, 0.84, 0.22, 0.10])
        ax_kpi.set_facecolor(color)
        ax_kpi.set_xticks([]); ax_kpi.set_yticks([])
        for sp in ax_kpi.spines.values(): sp.set_visible(False)
        ax_kpi.text(0.5, 0.65, value, ha="center", va="center",
                    fontsize=20, fontweight="bold", color="white",
                    transform=ax_kpi.transAxes)
        ax_kpi.text(0.5, 0.20, label, ha="center", va="center",
                    fontsize=10, color="white", alpha=0.85,
                    transform=ax_kpi.transAxes)

    # Panel 1: Fraud vs Legit pie
    ax1 = fig.add_axes([0.02, 0.44, 0.22, 0.36], facecolor="#1E293B")
    sizes = [total - fraud_count, fraud_count]
    ax1.pie(sizes, labels=["Legitimate", "Fraud"],
            colors=[palette["primary"], palette["danger"]],
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "white", "fontsize": 10})
    ax1.set_title("Fraud vs Legitimate", color="white", fontsize=11, pad=8)

    # Panel 2: Risk category bar
    ax2 = fig.add_axes([0.27, 0.44, 0.22, 0.36], facecolor="#1E293B")
    cats = ["Low", "Medium", "High", "Critical"]
    counts = [len(df[df["risk_category"] == c]) for c in cats]
    colors_bar = [palette["success"], "#F59E0B", "#F05252", palette["danger"]]
    ax2.barh(cats, counts, color=colors_bar, edgecolor="none", height=0.55)
    for val, y in zip(counts, range(len(cats))):
        ax2.text(val + 50, y, str(val), va="center", color="white", fontsize=9)
    ax2.set_facecolor("#1E293B")
    ax2.tick_params(colors="white", labelsize=9)
    ax2.set_xlabel("Count", color="white", fontsize=9)
    ax2.set_title("Transactions by Risk Level", color="white", fontsize=11, pad=8)
    for sp in ax2.spines.values(): sp.set_color("#374151")

    # Panel 3: Fraud by device
    ax3 = fig.add_axes([0.52, 0.44, 0.22, 0.36], facecolor="#1E293B")
    dev_fraud = df.groupby("device")["is_fraud"].mean() * 100
    ax3.bar(dev_fraud.index, dev_fraud.values,
            color=[palette["primary"], "#6366F1", "#06B6D4", "#8B5CF6"],
            edgecolor="none", width=0.55)
    ax3.set_facecolor("#1E293B")
    ax3.tick_params(colors="white", labelsize=9)
    ax3.set_ylabel("Fraud Rate (%)", color="white", fontsize=9)
    ax3.set_title("Fraud Rate by Device", color="white", fontsize=11, pad=8)
    for sp in ax3.spines.values(): sp.set_color("#374151")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: scatter fraud_prob vs risk_score
    ax4 = fig.add_axes([0.77, 0.44, 0.22, 0.36], facecolor="#1E293B")
    sample = df.sample(1500, random_state=42)
    colors_sc = [palette["danger"] if f else palette["primary"] for f in sample["is_fraud"]]
    ax4.scatter(sample["fraud_probability"], sample["risk_score"],
                c=colors_sc, alpha=0.4, s=8, edgecolors="none")
    ax4.set_facecolor("#1E293B")
    ax4.tick_params(colors="white", labelsize=8)
    ax4.set_xlabel("Fraud Probability", color="white", fontsize=9)
    ax4.set_ylabel("Risk Score", color="white", fontsize=9)
    ax4.set_title("Fraud Prob. vs Risk Score", color="white", fontsize=11, pad=8)
    for sp in ax4.spines.values(): sp.set_color("#374151")
    red_p = mpatches.Patch(color=palette["danger"], label="Fraud")
    blue_p = mpatches.Patch(color=palette["primary"], label="Legit")
    ax4.legend(handles=[red_p, blue_p], facecolor="#1E293B",
               labelcolor="white", fontsize=8, framealpha=0.5)

    # Panel 5: Daily trend
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    daily = df.groupby("transaction_date")["amount"].mean()
    ax5 = fig.add_axes([0.02, 0.05, 0.46, 0.32], facecolor="#1E293B")
    ax5.plot(daily.index, daily.values, color="#38BDF8", linewidth=1.5)
    ax5.fill_between(daily.index, daily.values, alpha=0.2, color="#38BDF8")
    ax5.set_facecolor("#1E293B")
    ax5.tick_params(colors="white", labelsize=8)
    ax5.set_title("Daily Average Transaction Amount (₹)", color="white", fontsize=11, pad=8)
    for sp in ax5.spines.values(): sp.set_color("#374151")
    fig.autofmt_xdate()

    # Panel 6: Top states by fraud count
    ax6 = fig.add_axes([0.52, 0.05, 0.46, 0.32], facecolor="#1E293B")
    state_fraud = df[df["is_fraud"] == 1]["state"].value_counts().head(10)
    ax6.barh(state_fraud.index[::-1], state_fraud.values[::-1],
             color="#F87171", edgecolor="none", height=0.6)
    ax6.set_facecolor("#1E293B")
    ax6.tick_params(colors="white", labelsize=9)
    ax6.set_xlabel("Fraud Cases", color="white", fontsize=9)
    ax6.set_title("Top States by Fraud Cases", color="white", fontsize=11, pad=8)
    for sp in ax6.spines.values(): sp.set_color("#374151")

    fig.savefig(f"{output_dir}/DashBoard.png", dpi=150, bbox_inches="tight",
                facecolor="#1E293B")
    plt.close(fig)
    print("   ✓ DashBoard.png")


# ─────────────────────────────────────────────
# 7. EXPORT FOR POWER BI / DATAVERSE
# ─────────────────────────────────────────────

def export_dataset(df, output_dir="data"):
    """Export processed dataset for Power BI import and Dataverse upload."""
    os.makedirs(output_dir, exist_ok=True)
    export_cols = [
        "transaction_id", "user_id", "age", "amount", "payment_method",
        "device", "state", "hour_of_day", "transaction_date", "is_new_payee",
        "login_attempts", "velocity_1h", "fraud_probability", "risk_score",
        "risk_category", "anomaly_flag", "is_fraud",
    ]
    out_path = f"{output_dir}/fraud_dashboard_dataset.csv"
    df[export_cols].to_csv(out_path, index=False)
    print(f"\n   ✓ Dataset exported → {out_path}  ({len(df):,} rows)")
    return out_path


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AI-Powered Fraud Detection & Risk Scoring Platform")
    print("  Author: Anjali Khatri | O23BCA110060 | CU")
    print("=" * 60)

    # Step 1 — Generate data
    print("\n[0/4] Generating synthetic transaction data...")
    df = generate_transactions(n=10000)
    print(f"   Generated {len(df):,} transactions  |  Fraud rate: {df['is_fraud'].mean()*100:.1f}%")

    # Step 2 — Features
    df, feature_cols = engineer_features(df)

    # Step 3 — Models
    df = run_isolation_forest(df, feature_cols)
    df, model = run_logistic_regression(df, feature_cols)

    # Step 4 — Risk scoring
    df = compute_risk_score(df)

    # Step 5 — Visualize
    plot_all(df, output_dir="screenshots")

    # Step 6 — Export
    export_dataset(df, output_dir="data")

    print("\n✅ Pipeline complete!")
    print("   → Open screenshots/DashBoard.png to preview the dashboard")
    print("   → Import data/fraud_dashboard_dataset.csv into Power BI")
