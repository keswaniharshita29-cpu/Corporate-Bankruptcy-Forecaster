import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.io import arff


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Corporate Bankruptcy Forecaster",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():

    model = joblib.load("bankruptcy_model.pkl")
    imputer = joblib.load("imputer.pkl")
    feature_names = joblib.load("feature_names.pkl")
    threshold = joblib.load("optimal_threshold.pkl")

    return model, imputer, feature_names, threshold


model, imputer, feature_names, threshold = load_model_files()


# ============================================================
# LOAD DATASET FOR DEMO COMPANIES
# ============================================================

@st.cache_data
def load_demo_data():

    data, meta = arff.loadarff("1year.arff")

    demo_df = pd.DataFrame(data)

    for col in demo_df.columns:

        if demo_df[col].dtype == object:

            demo_df[col] = demo_df[col].apply(
                lambda x:
                x.decode("utf-8")
                if isinstance(x, bytes)
                else x
            )

    demo_df = demo_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    return demo_df


demo_df = load_demo_data()


# ============================================================
# FEATURE DESCRIPTIONS
# ============================================================

feature_labels = {

    "Attr1": "Net Profit / Total Assets",
    "Attr2": "Total Liabilities / Total Assets",
    "Attr3": "Working Capital / Total Assets",
    "Attr4": "Current Assets / Short-Term Liabilities",
    "Attr5": "Cash + Securities + Receivables - ST Liabilities / Operating Expenses",
    "Attr6": "Retained Earnings / Total Assets",
    "Attr7": "EBIT / Total Assets",
    "Attr8": "Book Value of Equity / Total Liabilities",
    "Attr9": "Sales / Total Assets",
    "Attr10": "Equity / Total Assets",

    "Attr11": "(Gross Profit + Extraordinary Items + Financial Expenses) / Total Assets",
    "Attr12": "Gross Profit / Short-Term Liabilities",
    "Attr13": "(Gross Profit + Depreciation) / Sales",
    "Attr14": "(Gross Profit + Interest) / Total Assets",
    "Attr15": "(Total Liabilities × 365) / (Gross Profit + Depreciation)",
    "Attr16": "(Gross Profit + Depreciation) / Total Liabilities",
    "Attr17": "Total Assets / Total Liabilities",
    "Attr18": "Gross Profit / Total Assets",
    "Attr19": "Gross Profit / Sales",
    "Attr20": "Inventory × 365 / Sales",

    "Attr21": "Sales (Current Year) / Sales (Previous Year)",
    "Attr22": "Operating Profit / Total Assets",
    "Attr23": "Net Profit / Sales",
    "Attr24": "Gross Profit (3 Years) / Total Assets",
    "Attr25": "(Equity - Share Capital) / Total Assets",
    "Attr26": "(Net Profit + Depreciation) / Total Liabilities",
    "Attr27": "Operating Profit / Financial Expenses",
    "Attr28": "Working Capital / Fixed Assets",
    "Attr29": "Logarithm of Total Assets",
    "Attr30": "(Total Liabilities - Cash) / Sales",

    "Attr31": "(Gross Profit + Interest) / Sales",
    "Attr32": "Current Liabilities × 365 / Cost of Products Sold",
    "Attr33": "Operating Expenses / Short-Term Liabilities",
    "Attr34": "Operating Expenses / Total Liabilities",
    "Attr35": "Profit on Sales / Total Assets",
    "Attr36": "Total Sales / Total Assets",
    "Attr37": "(Current Assets - Inventory) / Long-Term Liabilities",
    "Attr38": "Constant Capital / Total Assets",
    "Attr39": "Profit on Sales / Sales",
    "Attr40": "(Current Assets - Inventory - Receivables) / Short-Term Liabilities",

    "Attr41": "Total Liabilities / (Operating Profit + Depreciation)",
    "Attr42": "Operating Profit / Sales",
    "Attr43": "Receivables + Inventory Turnover in Days",
    "Attr44": "Receivables × 365 / Sales",
    "Attr45": "Net Profit / Inventory",
    "Attr46": "(Current Assets - Inventory) / Short-Term Liabilities",
    "Attr47": "Inventory × 365 / Cost of Products Sold",
    "Attr48": "EBITDA / Total Assets",
    "Attr49": "EBITDA / Sales",
    "Attr50": "Current Assets / Total Liabilities",

    "Attr51": "Short-Term Liabilities / Total Assets",
    "Attr52": "Short-Term Liabilities × 365 / Cost of Products Sold",
    "Attr53": "Equity / Fixed Assets",
    "Attr54": "Constant Capital / Fixed Assets",
    "Attr55": "Working Capital",
    "Attr56": "(Sales - Cost of Products Sold) / Sales",
    "Attr57": "(Current Assets - Inventory - ST Liabilities) / Operating Cash Flow",
    "Attr58": "Total Costs / Total Sales",
    "Attr59": "Long-Term Liabilities / Equity",
    "Attr60": "Sales / Inventory",

    "Attr61": "Sales / Receivables",
    "Attr62": "Short-Term Liabilities × 365 / Sales",
    "Attr63": "Sales / Short-Term Liabilities",
    "Attr64": "Sales / Fixed Assets"
}


# ============================================================
# FEATURE CATEGORIES
# ============================================================

categories = {

    "💰 Profitability": [
        "Attr1", "Attr6", "Attr7", "Attr11",
        "Attr13", "Attr14", "Attr18", "Attr19",
        "Attr22", "Attr23", "Attr35", "Attr39"
    ],

    "💧 Liquidity": [
        "Attr3", "Attr4", "Attr5", "Attr12",
        "Attr28", "Attr37", "Attr40", "Attr46",
        "Attr50"
    ],

    "🏦 Solvency & Leverage": [
        "Attr2", "Attr8", "Attr10", "Attr15",
        "Attr16", "Attr17", "Attr25", "Attr26",
        "Attr30", "Attr34", "Attr38", "Attr51",
        "Attr53", "Attr54", "Attr59"
    ],

    "⚙️ Efficiency & Operations": [
        "Attr20", "Attr21", "Attr27", "Attr32",
        "Attr33", "Attr41", "Attr43", "Attr44",
        "Attr45", "Attr47", "Attr52", "Attr60",
        "Attr61", "Attr62", "Attr63", "Attr64"
    ],

    "📊 Asset Management": [
        "Attr9", "Attr24", "Attr29", "Attr36",
        "Attr48", "Attr49", "Attr55", "Attr56",
        "Attr57", "Attr58"
    ],

    "💼 Capital & Financial Structure": [
        "Attr31", "Attr42"
    ]
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    font-size: 19px;
    color: #94a3b8;
    margin-top: 5px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
}

.info-card {
    padding: 22px;
    border-radius: 15px;
    background: #111827;
    border: 1px solid #374151;
    margin-bottom: 20px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #374151;
    background: #111827;
}

.small-text {
    color: #94a3b8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Corporate Bankruptcy Forecaster</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered financial distress prediction using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🏦 Corporate Bankruptcy Forecaster")

    st.write(
        """
        This AI system analyzes financial ratios and predicts
        whether a company may be at risk of bankruptcy.
        """
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("Algorithm: Random Forest")
    st.write("Financial Features: 64")
    st.write(f"Decision Threshold: {threshold:.2f}")

    st.divider()

    st.subheader("📊 Dataset")

    st.write("Polish Companies Bankruptcy Dataset")
    st.write(f"Companies: {len(demo_df):,}")

    st.divider()

    st.warning(
        "This dashboard is an academic project and should not "
        "be used for real financial decisions."
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    """
    <div class="info-card">

    <b>🔍 How does the system work?</b><br><br>

    The company's financial ratios are provided to a trained
    Random Forest model. The model estimates a bankruptcy
    probability and compares it with the optimized decision
    threshold of <b>0.53</b>.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COMPANY FINANCIAL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🏢 Company Financial Information</div>',
    unsafe_allow_html=True
)

st.write(
    "You can test the system using a real company record from "
    "the dataset, enter financial ratios manually, or upload "
    "a company CSV/Excel file."
)


# ============================================================
# INPUT METHOD
# ============================================================

input_methods = [
    "Manual Financial Inputs",
    "Upload Company File (CSV / Excel)",
    "Demo: Non-Bankrupt Company",
    "Demo: Bankrupt Company"
]


demo_option = st.selectbox(
    "Choose input method",
    input_methods,
    key="input_method"
)


# ============================================================
# SESSION STATE
# ============================================================

if "manual_values" not in st.session_state:

    st.session_state.manual_values = {
        feature: 0.0
        for feature in feature_names
    }


# ============================================================
# UPLOAD FILE
# ============================================================

uploaded_df = None
selected_upload = None


if demo_option == "Upload Company File (CSV / Excel)":

    st.subheader("📁 Upload Company Financial Data")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"],
        help=(
            "Your file must contain Attr1 through Attr64. "
            "Each row represents one company."
        )
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):

                uploaded_df = pd.read_csv(uploaded_file)

            else:

                uploaded_df = pd.read_excel(uploaded_file)


            # Clean column names

            uploaded_df.columns = [
                str(column).strip()
                for column in uploaded_df.columns
            ]


            # Check required columns

            missing_columns = [
                feature
                for feature in feature_names
                if feature not in uploaded_df.columns
            ]


            if missing_columns:

                st.error(
                    "❌ Your file is missing these required columns: "
                    + ", ".join(missing_columns)
                )

                uploaded_df = None


            elif len(uploaded_df) == 0:

                st.error(
                    "❌ The uploaded file contains no company records."
                )

                uploaded_df = None


            else:

                st.success(
                    f"✅ File loaded successfully. "
                    f"{len(uploaded_df)} company record(s) found."
                )


                # If multiple companies are present

                if len(uploaded_df) > 1:

                    selected_row = st.selectbox(
                        "Select company record",
                        range(len(uploaded_df)),
                        format_func=lambda x:
                        f"Company record {x + 1}"
                    )

                    selected_upload = uploaded_df.iloc[selected_row]

                else:

                    selected_upload = uploaded_df.iloc[0]


        except Exception as error:

            st.error(
                f"❌ Could not read the uploaded file: {error}"
            )


# ============================================================
# DEMO COMPANY
# ============================================================

selected_demo = None


if demo_option == "Demo: Non-Bankrupt Company":

    non_bankrupt = demo_df[
        demo_df["class"] == 0
    ]

    selected_demo = non_bankrupt.iloc[0]

    st.success(
        "Demo company loaded from the dataset: "
        "actual non-bankrupt company record."
    )


elif demo_option == "Demo: Bankrupt Company":

    bankrupt = demo_df[
        demo_df["class"] == 1
    ]

    selected_demo = bankrupt.iloc[0]

    st.warning(
        "Demo company loaded from the dataset: "
        "actual bankrupt company record."
    )


# ============================================================
# APPLY DEMO / UPLOAD VALUES
# ============================================================

# ============================================================
# APPLY DEMO / UPLOAD VALUES
# ============================================================

if selected_demo is not None:

    for feature in feature_names:

        value = selected_demo[feature]

        if pd.isna(value):
            value = 0.0

        value = float(value)

        # Save both the persistent value AND the actual
        # Streamlit number_input widget value.
        st.session_state.manual_values[feature] = value

        widget_key = f"financial_input_{feature}"
        st.session_state[widget_key] = value


elif selected_upload is not None:

    for feature in feature_names:

        value = pd.to_numeric(
            pd.Series(
                [selected_upload[feature]]
            ),
            errors="coerce"
        ).iloc[0]

        if pd.isna(value):
            value = 0.0

        value = float(value)

        # Save both the persistent value AND the actual
        # Streamlit number_input widget value.
        st.session_state.manual_values[feature] = value

        widget_key = f"financial_input_{feature}"
        st.session_state[widget_key] = value


# ============================================================
# FINANCIAL INPUTS
# ============================================================

input_values = {}


for category, features in categories.items():

    with st.expander(
        category,
        expanded=False
    ):

        columns = st.columns(3)

        for i, feature in enumerate(features):

            with columns[i % 3]:

                widget_key = (
                    f"financial_input_{feature}"
                )


                if widget_key not in st.session_state:

                    st.session_state[
                        widget_key
                    ] = st.session_state[
                        "manual_values"
                    ].get(
                        feature,
                        0.0
                    )


                input_values[feature] = st.number_input(

                    feature_labels.get(
                        feature,
                        feature
                    ),

                    format="%.6f",

                    key=widget_key
                )


                # Save the value so it survives reruns

                st.session_state[
                    "manual_values"
                ][feature] = input_values[feature]


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")

st.info(
    "💡 Enter your financial ratios and click "
    "**ANALYZE COMPANY**. Your values will remain saved."
)


predict = st.button(
    "🔮 ANALYZE COMPANY",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # --------------------------------------------------------
    # Create dataframe in EXACT feature order
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            [
                input_values[feature]
                for feature in feature_names
            ]
        ],
        columns=feature_names
    )


    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    input_data = input_data.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # Apply trained imputer
    # --------------------------------------------------------

    try:

        input_imputed = imputer.transform(
            input_data
        )

    except Exception as error:

        st.error(
            "❌ The financial input could not be processed. "
            "Please make sure all 64 values are numeric."
        )

        st.exception(error)

        st.stop()


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_imputed
    )[0][1]


    probability_percent = (
        probability * 100
    )


    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    prediction = (
        1
        if probability >= threshold
        else 0
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🤖 AI Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if probability < 0.30:

        risk_level = "LOW"

    elif probability < threshold:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ BANKRUPTCY RISK DETECTED"
        )

        st.write(
            "The AI model classifies this company "
            "as potentially financially distressed."
        )

    else:

        st.success(
            "✅ NO BANKRUPTCY RISK DETECTED"
        )

        st.write(
            "The AI model does not classify this company "
            "as bankrupt at the selected threshold."
        )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Bankruptcy Probability",
            f"{probability_percent:.2f}%"
        )


    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with col3:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.write("")

    st.subheader(
        "📊 Bankruptcy Risk Score"
    )


    st.progress(
        float(
            min(
                probability,
                1.0
            )
        )
    )


    st.caption(
        f"Model-estimated bankruptcy probability: "
        f"{probability_percent:.2f}%"
    )


    # ========================================================
    # THRESHOLD EXPLANATION
    # ========================================================

    if probability >= threshold:

        st.warning(
            f"The predicted probability "
            f"({probability:.3f}) is above the "
            f"model threshold ({threshold:.2f}), "
            "so the company is classified as "
            "having bankruptcy risk."
        )

    else:

        st.info(
            f"The predicted probability "
            f"({probability:.3f}) is below the "
            f"model threshold ({threshold:.2f}), "
            "so the company is not classified "
            "as bankrupt."
        )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Most Important Financial Factors"
    )


    importance = model.feature_importances_


    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Financial Ratio": [
            feature_labels.get(
                feature,
                feature
            )
            for feature in feature_names
        ],

        "Importance": importance
    })


    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).head(10)


    chart_df = importance_df[
        [
            "Financial Ratio",
            "Importance"
        ]
    ].set_index(
        "Financial Ratio"
    )


    st.bar_chart(
        chart_df
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Financial Input Summary"
    )


    summary_df = pd.DataFrame({

        "Feature": feature_names,

        "Financial Ratio": [
            feature_labels.get(
                feature,
                feature
            )
            for feature in feature_names
        ],

        "Value": [
            input_values[feature]
            for feature in feature_names
        ]

    })


    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Corporate Bankruptcy Forecaster • "
    "Random Forest Machine Learning • "
    "Academic Project"
)
