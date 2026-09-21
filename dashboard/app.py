import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_engineered_housing_prices.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "dashboard",
    "models"
)

CLASSIFICATION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_classification_model.pkl"
)

REGRESSION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_regression_model.pkl"
)

MODEL_INFO_PATH = os.path.join(
    MODEL_DIR,
    "model_info.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not os.path.exists(
        CLASSIFICATION_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Classification model not found:\n"
            f"{CLASSIFICATION_MODEL_PATH}"
        )

    if not os.path.exists(
        REGRESSION_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Regression model not found:\n"
            f"{REGRESSION_MODEL_PATH}"
        )

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    if os.path.exists(MODEL_INFO_PATH):

        model_info = joblib.load(
            MODEL_INFO_PATH
        )

    else:

        model_info = {}

    return (
        classification_model,
        regression_model,
        model_info
    )


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features(model):

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )

    if hasattr(
        model,
        "named_steps"
    ):

        preprocessor = (
            model.named_steps.get(
                "preprocessor"
            )
        )

        if preprocessor is not None:

            if hasattr(
                preprocessor,
                "feature_names_in_"
            ):

                return list(
                    preprocessor.feature_names_in_
                )

    return []


# ============================================================
# PREPARE USER INPUT
# ============================================================

def prepare_input(
    input_data,
    model
):

    input_df = pd.DataFrame(
        [input_data]
    )

    expected_features = get_model_features(
        model
    )

    if expected_features:

        for column in expected_features:

            if column not in input_df.columns:

                input_df[column] = np.nan

        input_df = input_df[
            expected_features
        ]

    return input_df


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    top_n=15
):

    try:

        if not hasattr(
            model,
            "named_steps"
        ):

            return pd.DataFrame()

        preprocessor = (
            model.named_steps.get(
                "preprocessor"
            )
        )

        estimator = (
            model.named_steps.get(
                "model"
            )
        )

        if (
            preprocessor is None
            or estimator is None
        ):

            return pd.DataFrame()

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            estimator,
            "feature_importances_"
        ):

            importance = (
                estimator.feature_importances_
            )

        elif hasattr(
            estimator,
            "coef_"
        ):

            coefficients = estimator.coef_

            if coefficients.ndim > 1:

                importance = np.abs(
                    coefficients[0]
                )

            else:

                importance = np.abs(
                    coefficients
                )

        else:

            return pd.DataFrame()

        result = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance
            }
        )

        return (
            result
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(top_n)
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD APPLICATION
# ============================================================

try:

    df = load_data()

    (
        classification_model,
        regression_model,
        model_info
    ) = load_models()

except Exception as e:

    st.error(
        "❌ Error loading data or models"
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🏠 Real Estate Advisor"
)

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Prediction",
        "🔎 Property Filter",
        "📊 Market Insights",
        "💰 Price Analysis",
        "🤖 Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
**Project**

Real Estate Investment Advisor

**Machine Learning**

• Good Investment Classification
• Future Price Prediction

**Technology**

Python  
Pandas  
Scikit-learn  
Streamlit
"""
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🏠 Real Estate Investment Advisor"
)

st.markdown(
    """
### Predicting Property Profitability & Future Value

Analyze property characteristics, investment potential,
market insights and estimated future property value.
"""
)

st.markdown("---")


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "🏠 Prediction":

    st.header(
        "🔮 Property Investment Prediction"
    )

    st.write(
        "Enter the property details below to predict "
        "whether the property is a good investment "
        "and estimate its future value."
    )

    # ========================================================
    # LOCATION
    # ========================================================

    st.subheader(
        "📍 Location Details"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        states = sorted(
            df["State"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        state = st.selectbox(
            "State",
            states
        )

    with col2:

        city_data = df[
            df["State"].astype(str)
            == str(state)
        ]

        cities = sorted(
            city_data["City"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        city = st.selectbox(
            "City",
            cities
        )

    with col3:

        locality_data = df[
            (
                df["State"].astype(str)
                == str(state)
            )
            &
            (
                df["City"].astype(str)
                == str(city)
            )
        ]

        localities = sorted(
            locality_data["Locality"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if localities:

            locality = st.selectbox(
                "Locality",
                localities
            )

        else:

            locality = st.text_input(
                "Locality"
            )


    # ========================================================
    # PROPERTY DETAILS
    # ========================================================

    st.subheader(
        "🏢 Property Details"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        property_types = sorted(
            df["Property_Type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        property_type = st.selectbox(
            "Property Type",
            property_types
        )

    with col2:

        bhk_values = pd.to_numeric(
            df["BHK"],
            errors="coerce"
        ).dropna()

        bhk_min = int(
            bhk_values.min()
        )

        bhk_max = int(
            bhk_values.max()
        )

        bhk = st.number_input(
            "BHK",
            min_value=max(
                1,
                bhk_min
            ),
            max_value=max(
                1,
                bhk_max
            ),
            value=max(
                1,
                bhk_min
            ),
            step=1
        )

    with col3:

        size_values = pd.to_numeric(
            df["Size_in_SqFt"],
            errors="coerce"
        ).dropna()

        size_min = int(
            size_values.min()
        )

        size_max = int(
            size_values.max()
        )

        default_size = int(
            np.clip(
                1000,
                size_min,
                size_max
            )
        )

        size_sqft = st.number_input(
            "Size (Sq Ft)",
            min_value=max(
                1,
                size_min
            ),
            max_value=max(
                1,
                size_max
            ),
            value=default_size,
            step=50
        )

    with col4:

        year_values = pd.to_numeric(
            df["Year_Built"],
            errors="coerce"
        ).dropna()

        year_min = int(
            year_values.min()
        )

        year_max = int(
            year_values.max()
        )

        year_built = st.number_input(
            "Year Built",
            min_value=year_min,
            max_value=year_max,
            value=year_max,
            step=1
        )


    # ========================================================
    # PROPERTY CONDITION
    # ========================================================

    st.subheader(
        "🛋️ Property Condition"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        furnished_values = sorted(
            df["Furnished_Status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        furnished_status = st.selectbox(
            "Furnished Status",
            furnished_values
        )

    with col2:

        floor_no = st.number_input(
            "Floor Number",
            min_value=0,
            value=1,
            step=1
        )

    with col3:

        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            value=5,
            step=1
        )

    with col4:

        age_property = st.number_input(
            "Age of Property",
            min_value=0,
            value=5,
            step=1
        )


    # ========================================================
    # FACILITIES
    # ========================================================

    st.subheader(
        "🏫 Facilities & Infrastructure"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        nearby_schools = st.number_input(
            "Nearby Schools",
            min_value=0,
            value=5,
            step=1
        )

    with col2:

        nearby_hospitals = st.number_input(
            "Nearby Hospitals",
            min_value=0,
            value=3,
            step=1
        )

    with col3:

        transport_values = sorted(
            df[
                "Public_Transport_Accessibility"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if transport_values:

            transport = st.selectbox(
                "Public Transport Accessibility",
                transport_values
            )

        else:

            transport = st.selectbox(
                "Public Transport Accessibility",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )


    # ========================================================
    # OTHER FEATURES
    # ========================================================

    st.subheader(
        "🔐 Other Property Features"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        parking_values = sorted(
            df["Parking_Space"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        parking = st.selectbox(
            "Parking Space",
            parking_values
        )

    with col2:

        security_values = sorted(
            df["Security"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        security = st.selectbox(
            "Security",
            security_values
        )

    with col3:

        facing_values = sorted(
            df["Facing"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        facing = st.selectbox(
            "Facing",
            facing_values
        )

    with col4:

        owner_values = sorted(
            df["Owner_Type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        owner_type = st.selectbox(
            "Owner Type",
            owner_values
        )


    # ========================================================
    # AVAILABILITY + AMENITIES
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        availability_values = sorted(
            df["Availability_Status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        availability_status = st.selectbox(
            "Availability Status",
            availability_values
        )

    with col2:

        amenities = st.text_area(
            "Amenities",
            placeholder=(
                "Gym, Swimming Pool, "
                "Garden, Parking..."
            )
        )


    # ========================================================
    # PREDICT
    # ========================================================

    st.markdown("---")

    predict_button = st.button(
        "🔮 Predict Investment",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        try:

            # ==================================================
            # CREATE INPUT
            # ==================================================

            input_data = {

                "State": state,

                "City": city,

                "Locality": locality,

                "Property_Type": property_type,

                "BHK": bhk,

                "Size_in_SqFt": size_sqft,

                "Year_Built": year_built,

                "Furnished_Status": furnished_status,

                "Floor_No": floor_no,

                "Total_Floors": total_floors,

                "Age_of_Property": age_property,

                "Nearby_Schools": nearby_schools,

                "Nearby_Hospitals": nearby_hospitals,

                "Public_Transport_Accessibility": transport,

                "Parking_Space": parking,

                "Security": security,

                "Amenities": amenities,

                "Facing": facing,

                "Owner_Type": owner_type,

                "Availability_Status": availability_status
            }


            # ==================================================
            # CLASSIFICATION
            # ==================================================

            classification_input = prepare_input(
                input_data,
                classification_model
            )

            investment_prediction = (
                classification_model
                .predict(
                    classification_input
                )[0]
            )

            investment_prediction = int(
                investment_prediction
            )


            # ==================================================
            # CLASSIFICATION CONFIDENCE
            # ==================================================

            confidence = None

            if hasattr(
                classification_model,
                "predict_proba"
            ):

                probabilities = (
                    classification_model
                    .predict_proba(
                        classification_input
                    )
                )

                confidence = (
                    float(
                        np.max(
                            probabilities[0]
                        )
                    )
                    * 100
                )


            # ==================================================
            # REGRESSION
            # ==================================================

            regression_input = prepare_input(
                input_data,
                regression_model
            )

            future_price = (
                regression_model
                .predict(
                    regression_input
                )[0]
            )

            future_price = float(
                future_price
            )


            # ==================================================
            # RESULTS
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📊 Prediction Results"
            )

            col1, col2 = st.columns(2)

            with col1:

                if investment_prediction == 1:

                    st.success(
                        "✅ GOOD INVESTMENT"
                    )

                    st.write(
                        "The model predicts that "
                        "this property is a good "
                        "investment."
                    )

                else:

                    st.warning(
                        "⚠️ NOT A GOOD INVESTMENT"
                    )

                    st.write(
                        "The model predicts that "
                        "this property is not a "
                        "good investment."
                    )

                if confidence is not None:

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

            with col2:

                st.metric(
                    "Estimated Future Price (5 Years)",
                    f"₹ {future_price:,.2f} Lakhs"
                )


            # ==================================================
            # PROPERTY SUMMARY
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📋 Property Summary"
            )

            summary = pd.DataFrame(
                {
                    "Feature": [
                        "State",
                        "City",
                        "Locality",
                        "Property Type",
                        "BHK",
                        "Size",
                        "Year Built",
                        "Furnished Status",
                        "Floor Number",
                        "Total Floors",
                        "Age of Property",
                        "Nearby Schools",
                        "Nearby Hospitals",
                        "Public Transport",
                        "Parking",
                        "Security",
                        "Facing",
                        "Owner Type",
                        "Availability Status"
                    ],
                    "Value": [
                        state,
                        city,
                        locality,
                        property_type,
                        bhk,
                        f"{size_sqft:,} Sq Ft",
                        year_built,
                        furnished_status,
                        floor_no,
                        total_floors,
                        age_property,
                        nearby_schools,
                        nearby_hospitals,
                        transport,
                        parking,
                        security,
                        facing,
                        owner_type,
                        availability_status
                    ]
                }
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )


        except Exception as error:

            st.error(
                "❌ Prediction error"
            )

            st.exception(error)


# ============================================================
# PAGE 2 — PROPERTY FILTER
# ============================================================

elif page == "🔎 Property Filter":

    st.header(
        "🔎 Property Filter"
    )

    st.write(
        "Filter properties based on location, "
        "BHK, price and size."
    )


    # ========================================================
    # FILTER OPTIONS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        state_filter = st.selectbox(
            "State",
            [
                "All"
            ]
            +
            sorted(
                df["State"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    with col2:

        city_filter = st.selectbox(
            "City",
            [
                "All"
            ]
            +
            sorted(
                df["City"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

    with col3:

        bhk_filter = st.selectbox(
            "BHK",
            [
                "All"
            ]
            +
            sorted(
                df["BHK"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    # ========================================================
    # PRICE + SIZE
    # ========================================================

    col1, col2 = st.columns(2)

    price_values = pd.to_numeric(
        df["Price_in_Lakhs"],
        errors="coerce"
    ).dropna()

    size_values = pd.to_numeric(
        df["Size_in_SqFt"],
        errors="coerce"
    ).dropna()

    with col1:

        price_range = st.slider(
            "Price Range (Lakhs)",
            min_value=float(
                price_values.min()
            ),
            max_value=float(
                price_values.max()
            ),
            value=(
                float(
                    price_values.min()
                ),
                float(
                    price_values.max()
                )
            )
        )

    with col2:

        size_range = st.slider(
            "Size Range (Sq Ft)",
            min_value=float(
                size_values.min()
            ),
            max_value=float(
                size_values.max()
            ),
            value=(
                float(
                    size_values.min()
                ),
                float(
                    size_values.max()
                )
            )
        )


    # ========================================================
    # APPLY FILTER
    # ========================================================

    filtered_df = df.copy()

    if state_filter != "All":

        filtered_df = filtered_df[
            filtered_df["State"].astype(str)
            == str(state_filter)
        ]

    if city_filter != "All":

        filtered_df = filtered_df[
            filtered_df["City"].astype(str)
            == str(city_filter)
        ]

    if bhk_filter != "All":

        filtered_df = filtered_df[
            filtered_df["BHK"]
            == bhk_filter
        ]


    filtered_price = pd.to_numeric(
        filtered_df["Price_in_Lakhs"],
        errors="coerce"
    )

    filtered_size = pd.to_numeric(
        filtered_df["Size_in_SqFt"],
        errors="coerce"
    )

    filtered_df = filtered_df[
        filtered_price.between(
            price_range[0],
            price_range[1]
        )
        &
        filtered_size.between(
            size_range[0],
            size_range[1]
        )
    ]


    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown("---")

    st.metric(
        "Properties Found",
        f"{len(filtered_df):,}"
    )

    display_columns = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Year_Built",
        "Furnished_Status"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[
            display_columns
        ].head(500),
        use_container_width=True,
        hide_index=True
    )

    if len(filtered_df) > 500:

        st.info(
            "Showing the first 500 matching properties."
        )


# ============================================================
# PAGE 3 — MARKET INSIGHTS
# ============================================================

elif page == "📊 Market Insights":

    st.header(
        "📊 Market Insights"
    )

    st.write(
        "Explore overall property characteristics "
        "and market patterns."
    )


    # ========================================================
    # KEY METRICS
    # ========================================================

    total_properties = len(df)

    average_price = pd.to_numeric(
        df["Price_in_Lakhs"],
        errors="coerce"
    ).mean()

    average_size = pd.to_numeric(
        df["Size_in_SqFt"],
        errors="coerce"
    ).mean()

    average_bhk = pd.to_numeric(
        df["BHK"],
        errors="coerce"
    ).mean()


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Properties",
            f"{total_properties:,}"
        )

    with col2:

        st.metric(
            "Average Price",
            f"₹ {average_price:,.2f} L"
        )

    with col3:

        st.metric(
            "Average Size",
            f"{average_size:,.0f} Sq Ft"
        )

    with col4:

        st.metric(
            "Average BHK",
            f"{average_bhk:.2f}"
        )


    st.markdown("---")


    # ========================================================
    # CITY PRICE
    # ========================================================

    st.subheader(
        "🏙️ Average Price by City"
    )

    city_price = (
        df.groupby("City")[
            "Price_in_Lakhs"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    city_price.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Property Price by City"
    )

    ax.set_xlabel(
        "City"
    )

    ax.set_ylabel(
        "Average Price (Lakhs)"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # PROPERTY TYPE
    # ========================================================

    st.subheader(
        "🏢 Property Type Distribution"
    )

    property_counts = (
        df["Property_Type"]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    property_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Number of Properties by Property Type"
    )

    ax.set_xlabel(
        "Property Type"
    )

    ax.set_ylabel(
        "Number of Properties"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # BHK DISTRIBUTION
    # ========================================================

    st.subheader(
        "🛏️ BHK Distribution"
    )

    bhk_counts = (
        df["BHK"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    bhk_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "BHK Distribution"
    )

    ax.set_xlabel(
        "BHK"
    )

    ax.set_ylabel(
        "Number of Properties"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # TRANSPORT VS PRICE
    # ========================================================

    if (
        "Public_Transport_Accessibility"
        in df.columns
    ):

        st.subheader(
            "🚇 Transport Accessibility vs Average Price"
        )

        transport_price = (
            df.groupby(
                "Public_Transport_Accessibility"
            )["Price_in_Lakhs"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        transport_price.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Average Price by Public Transport Accessibility"
        )

        ax.set_xlabel(
            "Transport Accessibility"
        )

        ax.set_ylabel(
            "Average Price (Lakhs)"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# PAGE 4 — PRICE ANALYSIS
# ============================================================

elif page == "💰 Price Analysis":

    st.header(
        "💰 Price Analysis"
    )

    st.write(
        "Analyze property prices, size and "
        "construction characteristics."
    )


    # ========================================================
    # PRICE DISTRIBUTION
    # ========================================================

    st.subheader(
        "📈 Property Price Distribution"
    )

    price_data = pd.to_numeric(
        df["Price_in_Lakhs"],
        errors="coerce"
    ).dropna()

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        price_data,
        bins=50
    )

    ax.set_title(
        "Property Price Distribution"
    )

    ax.set_xlabel(
        "Price (Lakhs)"
    )

    ax.set_ylabel(
        "Number of Properties"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # SIZE VS PRICE
    # ========================================================

    st.subheader(
        "📐 Property Size vs Price"
    )

    scatter_df = df[
        [
            "Size_in_SqFt",
            "Price_in_Lakhs"
        ]
    ].copy()

    scatter_df[
        "Size_in_SqFt"
    ] = pd.to_numeric(
        scatter_df["Size_in_SqFt"],
        errors="coerce"
    )

    scatter_df[
        "Price_in_Lakhs"
    ] = pd.to_numeric(
        scatter_df["Price_in_Lakhs"],
        errors="coerce"
    )

    scatter_df = scatter_df.dropna()

    if len(scatter_df) > 5000:

        scatter_df = scatter_df.sample(
            5000,
            random_state=42
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        scatter_df["Size_in_SqFt"],
        scatter_df["Price_in_Lakhs"],
        alpha=0.5
    )

    ax.set_title(
        "Property Size vs Price"
    )

    ax.set_xlabel(
        "Size (Sq Ft)"
    )

    ax.set_ylabel(
        "Price (Lakhs)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # PRICE PER SQ FT
    # ========================================================

    if "Price_per_SqFt" in df.columns:

        st.subheader(
            "💵 Price per Sq Ft by Property Type"
        )

        price_sqft = (
            df.groupby(
                "Property_Type"
            )["Price_per_SqFt"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        price_sqft.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Average Price per Sq Ft by Property Type"
        )

        ax.set_xlabel(
            "Property Type"
        )

        ax.set_ylabel(
            "Price per Sq Ft"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # ========================================================
    # YEAR BUILT VS PRICE
    # ========================================================

    st.subheader(
        "🏗️ Average Price by Year Built"
    )

    year_price = (
        df.groupby(
            "Year_Built"
        )["Price_in_Lakhs"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    year_price.plot(
        kind="line",
        ax=ax
    )

    ax.set_title(
        "Average Property Price by Year Built"
    )

    ax.set_xlabel(
        "Year Built"
    )

    ax.set_ylabel(
        "Average Price (Lakhs)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "Note: This chart compares average property "
        "prices by construction year. It is not a "
        "historical market-price trend."
    )


# ============================================================
# PAGE 5 — MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.header(
        "🤖 Model Performance"
    )

    st.write(
        "Performance metrics and important features "
        "of the selected machine learning models."
    )


    # ========================================================
    # MODEL NAMES
    # ========================================================

    best_classification_name = model_info.get(
        "best_classification_model",
        "Classification Model"
    )

    best_regression_name = model_info.get(
        "best_regression_model",
        "Regression Model"
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    st.subheader(
        "🎯 Classification Model"
    )

    st.write(
        f"**Selected Model:** "
        f"{best_classification_name}"
    )

    classification_metrics = model_info.get(
        "best_classification_metrics",
        {}
    )

    if classification_metrics:

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "Accuracy",
                f"{classification_metrics.get('Accuracy', 0):.4f}"
            )

        with col2:

            st.metric(
                "Precision",
                f"{classification_metrics.get('Precision', 0):.4f}"
            )

        with col3:

            st.metric(
                "Recall",
                f"{classification_metrics.get('Recall', 0):.4f}"
            )

        with col4:

            st.metric(
                "F1 Score",
                f"{classification_metrics.get('F1 Score', 0):.4f}"
            )

        with col5:

            st.metric(
                "ROC-AUC",
                f"{classification_metrics.get('ROC-AUC', 0):.4f}"
            )

    else:

        st.info(
            "Classification metrics are not available "
            "in model_info.pkl."
        )


    # ========================================================
    # CLASSIFICATION FEATURES
    # ========================================================

    st.subheader(
        "📌 Important Classification Features"
    )

    classification_importance = (
        get_feature_importance(
            classification_model
        )
    )

    if not classification_importance.empty:

        st.dataframe(
            classification_importance,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        plot_data = (
            classification_importance
            .sort_values(
                "Importance"
            )
        )

        ax.barh(
            plot_data["Feature"],
            plot_data["Importance"]
        )

        ax.set_title(
            "Top Classification Features"
        )

        ax.set_xlabel(
            "Importance"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "Feature importance is not available "
            "for the classification model."
        )


    # ========================================================
    # REGRESSION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📈 Regression Model"
    )

    st.write(
        f"**Selected Model:** "
        f"{best_regression_name}"
    )

    regression_metrics = model_info.get(
        "best_regression_metrics",
        {}
    )

    if regression_metrics:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "MAE",
                f"{regression_metrics.get('MAE', 0):.6f}"
            )

        with col2:

            st.metric(
                "RMSE",
                f"{regression_metrics.get('RMSE', 0):.6f}"
            )

        with col3:

            st.metric(
                "R² Score",
                f"{regression_metrics.get('R2 Score', 0):.6f}"
            )

    else:

        st.info(
            "Regression metrics are not available "
            "in model_info.pkl."
        )


    # ========================================================
    # REGRESSION FEATURES
    # ========================================================

    st.subheader(
        "📌 Important Regression Features"
    )

    regression_importance = (
        get_feature_importance(
            regression_model
        )
    )

    if not regression_importance.empty:

        st.dataframe(
            regression_importance,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        plot_data = (
            regression_importance
            .sort_values(
                "Importance"
            )
        )

        ax.barh(
            plot_data["Feature"],
            plot_data["Importance"]
        )

        ax.set_title(
            "Top Regression Features"
        )

        ax.set_xlabel(
            "Importance"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "Feature importance is not available "
            "for the regression model."
        )


    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🎯 Target Variables"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"""
**Classification Target**

{model_info.get(
    "classification_target",
    "Good_Investment"
)}

The model predicts whether a property
is classified as a good investment.
"""
        )

    with col2:

        st.info(
            f"""
**Regression Target**

{model_info.get(
    "regression_target",
    "Future_Price_5Y"
)}

The model estimates the property value
after five years.
"""
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Real Estate Investment Advisor | "
    "Machine Learning Project"
)
