import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


# =====================================================
# CONFIG PAGE
# =====================================================
st.set_page_config(
    page_title="Credit Scoring AI",
    page_icon="💳",
    layout="wide"
)


# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    import os

    for f in ["modele_final.pkl", "model.pkl", "model_final.pkl"]:
        if os.path.exists(f):
            with open(f, "rb") as file:
                return pickle.load(file)

    return None


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():

    import os

    for f in ["credit_risk_dataset (1).csv", "credit_risk_dataset.csv"]:
        if os.path.exists(f):
            df = pd.read_csv(f)
            return df

    # fallback demo
    np.random.seed(42)
    return pd.DataFrame({
        "age_client": np.random.randint(18, 70, 1000),
        "revenu_annuel": np.random.randint(20000, 150000, 1000),
        "montant_pret": np.random.randint(1000, 40000, 1000),
        "taux_interet": np.random.uniform(5, 25, 1000),
        "statut_pret": np.random.choice([0, 1], 1000)
    })


# =====================================================
# RENAME COLUMNS (CRUCIAL FIX)
# =====================================================
def clean_columns(df):

    mapping = {
        "person_age": "age_client",
        "person_income": "revenu_annuel",
        "person_home_ownership": "type_logement",
        "person_emp_length": "anciennete_emploi",
        "loan_intent": "motif_pret",
        "loan_grade": "note_credit",
        "loan_amnt": "montant_pret",
        "loan_int_rate": "taux_interet",
        "loan_status": "statut_pret",
        "loan_percent_income": "ratio_revenu_pret",
        "cb_person_default_on_file": "defaut_dans_historique",
        "cb_person_cred_hist_length": "anciennete_historique_credit"
    }

    return df.rename(columns=mapping)


# =====================================================
# HOME
# =====================================================
def show_home():
    st.title("💳 Credit Scoring AI")
    st.write("Application IA de prédiction du risque de crédit")


# =====================================================
# PREDICTION
# =====================================================
def show_prediction():

    st.title("🎯 Prédiction de Crédit")

    model = load_model()

    if model is None:
        st.warning("Ajoute ton modèle .pkl")
        return

    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Revenu", 10000, 200000, 50000)
    amount = st.number_input("Montant prêt", 1000, 50000, 10000)
    rate = st.slider("Taux intérêt", 5.0, 25.0, 12.0)

    ratio = amount / income

    if st.button("Prédire"):
        X = pd.DataFrame({
            "age_client":[age],
            "revenu_annuel":[income],
            "montant_pret":[amount],
            "taux_interet":[rate],
            "ratio_revenu_pret":[ratio]
        })

        pred = model.predict(X)[0]

        if pred == 0:
            st.success("✅ Crédit accepté")
        else:
            st.error("❌ Crédit refusé")


# =====================================================
# ANALYTICS
def show_analytics():

    st.title("📊 Advanced Credit Risk Dashboard")

    df = load_data()
    df = clean_columns(df)

    # =====================================================
    # SIDEBAR FILTERS
    # =====================================================
    st.sidebar.header("🔍 Filtres dynamiques")

    age_range = st.sidebar.slider(
        "Age",
        int(df.age_client.min()),
        int(df.age_client.max()),
        (25, 60)
    )

    income_range = st.sidebar.slider(
        "Revenu",
        int(df.revenu_annuel.min()),
        int(df.revenu_annuel.max()),
        (20000, 120000)
    )

    grades = st.sidebar.multiselect(
        "Note crédit",
        df.note_credit.unique(),
        default=list(df.note_credit.unique())
    )

    logements = st.sidebar.multiselect(
        "Logement",
        df.type_logement.unique(),
        default=list(df.type_logement.unique())
    )

    # =====================================================
    # FILTER DATA
    # =====================================================
    df = df[
        (df.age_client.between(*age_range)) &
        (df.revenu_annuel.between(*income_range)) &
        (df.note_credit.isin(grades)) &
        (df.type_logement.isin(logements))
    ]

    # =====================================================
    # KPI CARDS
    # =====================================================
    col1, col2, col3, col4, col5 = st.columns(5)

    total = len(df)
    default_rate = df.statut_pret.mean() * 100
    avg_amount = df.montant_pret.mean()
    avg_rate = df.taux_interet.mean()
    expected_loss = (df.montant_pret * df.statut_pret).sum()

    col1.metric("Total prêts", total)
    col2.metric("Taux défaut", f"{default_rate:.1f}%")
    col3.metric("Montant moyen", f"{avg_amount:,.0f}€")
    col4.metric("Taux moyen", f"{avg_rate:.1f}%")
    col5.metric("Perte estimée", f"{expected_loss:,.0f}€")

    st.divider()

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distributions",
        "🔗 Corrélations",
        "🎯 Risque",
        "🧬 Segmentation"
    ])

    # =====================================================
    # TAB 1 DISTRIBUTIONS
    # =====================================================
    with tab1:

        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(df, x="age_client", color="statut_pret",
                               marginal="box")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.violin(df, x="note_credit", y="revenu_annuel",
                            color="statut_pret", box=True)
            st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            df,
            x="montant_pret",
            y="taux_interet",
            size="revenu_annuel",
            color="statut_pret",
            hover_data=["age_client"]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # TAB 2 CORRELATION
    # =====================================================
    with tab2:

        numeric_cols = df.select_dtypes(include=np.number)

        corr = numeric_cols.corr()

        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter_matrix(
            df,
            dimensions=["age_client", "revenu_annuel",
                        "montant_pret", "taux_interet"],
            color="statut_pret"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # TAB 3 RISK INTELLIGENCE
    # =====================================================
    with tab3:

        risk_by_grade = df.groupby("note_credit")["statut_pret"].mean() * 100
        fig = px.bar(x=risk_by_grade.index, y=risk_by_grade.values)
        st.plotly_chart(fig, use_container_width=True)

        top_risk = df.sort_values("ratio_revenu_pret", ascending=False).head(10)
        st.subheader("Top 10 clients risqués")
        st.dataframe(top_risk)

    # =====================================================
    # TAB 4 SEGMENTATION KMEANS
    # =====================================================
    with tab4:

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = df[["age_client", "revenu_annuel",
                "montant_pret", "taux_interet"]]

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=0)
        df["cluster"] = kmeans.fit_predict(Xs)

        fig = px.scatter(
            df,
            x="revenu_annuel",
            y="montant_pret",
            color="cluster",
            size="taux_interet"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("Clusters clients détectés automatiquement (Low / Medium / High risk)")

    st.divider()
    st.dataframe(df.head(200))



# =====================================================
# MAIN
# =====================================================
def main():

    page = st.sidebar.radio(
        "Navigation",
        ["Accueil", "Prédiction", "Analyse"]
    )

    if page == "Accueil":
        show_home()

    elif page == "Prédiction":
        show_prediction()

    else:
        show_analytics()


if __name__ == "__main__":
    main()

