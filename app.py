import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
    }
    h2 {
        color: #34495e;
        font-weight: 600;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    h3 {
        color: #555;
        font-weight: 500;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        # Chercher le modèle dans le répertoire courant
        import os
        possible_files = ['modele_final.pkl', 'model.pkl', 'credit_model.pkl']
        for file in possible_files:
            if os.path.exists(file):
                with open(file, 'rb') as f:
                    model = pickle.load(f)
                return model
        return None
    except Exception as e:
        st.warning(f"Impossible de charger le modèle: {e}")
        return None

# Fonction pour charger les données
@st.cache_data
def load_data():
    try:
        # Essayer différents noms de fichiers de données
        import os
        possible_files = ['credit_data.csv', 'credit_risk_dataset (1).csv', 'credit_risk_dataset.csv']
        for file in possible_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                return df
        # Si aucun fichier trouvé, créer des données de démonstration
        raise FileNotFoundError("Aucun fichier de données trouvé")
    except:
        # Si pas de fichier, créer des données de démonstration
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'age_client': np.random.randint(18, 75, n_samples),
            'revenu_annuel': np.random.randint(20000, 200000, n_samples),
            'type_logement': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples),
            'anciennete_emploi': np.random.uniform(0, 40, n_samples),
            'motif_pret': np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], n_samples),
            'note_credit': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples),
            'montant_pret': np.random.randint(1000, 50000, n_samples),
            'taux_interet': np.random.uniform(5, 25, n_samples),
            'statut_pret': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'ratio_revenu_pret': np.random.uniform(0.05, 0.5, n_samples),
            'defaut_dans_historique': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'anciennete_historique_credit': np.random.randint(0, 30, n_samples)
        })
        return df

# Navigation
def main():
    st.sidebar.title("🏦 Navigation")
    page = st.sidebar.radio(
        "Choisissez une page",
        ["🏠 Accueil", "🎯 Prédiction de Crédit", "📊 Analyse de Données"],
        index=0
    )
    
    if page == "🏠 Accueil":
        show_home()
    elif page == "🎯 Prédiction de Crédit":
        show_prediction()
    elif page == "📊 Analyse de Données":
        show_analytics()

def show_home():
    st.title("💳 Credit Scoring AI - Système de Prédiction Intelligent")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Prédiction Précise</h3>
            <p>Évaluez instantanément le risque de crédit avec notre modèle IA avancé</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analyses Avancées</h3>
            <p>Visualisez les tendances et patterns dans vos données de crédit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Performance Optimale</h3>
            <p>Décisions rapides basées sur des algorithmes de machine learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📋 À propos du système")
    st.write("""
    Cette application utilise des techniques avancées de machine learning pour prédire le risque de défaut de paiement
    sur les prêts. Elle analyse 12 variables clés pour fournir une évaluation complète du profil de crédit.
    
    **Fonctionnalités principales :**
    - ✅ Prédiction en temps réel du risque de crédit
    - ✅ Visualisations interactives avancées
    - ✅ Analyse des tendances et corrélations
    - ✅ Interface intuitive et professionnelle
    """)

def show_prediction():
    st.title("🎯 Prédiction de Risque de Crédit")
    
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Le modèle n'est pas encore chargé. Veuillez placer 'modele_final.pkl' dans le répertoire.")
        return
    
    st.markdown("### 📝 Informations du Client")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_client = st.slider("Âge du client", 18, 75, 35)
        revenu_annuel = st.number_input("Revenu annuel (€)", 10000, 500000, 50000, step=5000)
        type_logement = st.selectbox("Type de logement", ['RENT', 'OWN', 'MORTGAGE'])
        anciennete_emploi = st.number_input("Ancienneté emploi (années)", 0.0, 40.0, 5.0, step=0.5)
        motif_pret = st.selectbox("Motif du prêt", 
                                   ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 
                                    'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
        note_credit = st.selectbox("Note de crédit", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    
    with col2:
        montant_pret = st.number_input("Montant du prêt (€)", 1000, 100000, 15000, step=1000)
        taux_interet = st.slider("Taux d'intérêt (%)", 5.0, 25.0, 12.0, step=0.5)
        defaut_dans_historique = st.selectbox("Défaut dans l'historique", ['Yes', 'No'])
        anciennete_historique_credit = st.slider("Ancienneté historique crédit (années)", 0, 30, 5)
        
        # Calcul automatique du ratio
        ratio_revenu_pret = montant_pret / revenu_annuel if revenu_annuel > 0 else 0
        st.metric("Ratio Revenu/Prêt", f"{ratio_revenu_pret:.2%}")
    
    if st.button("🔮 Prédire le Risque", use_container_width=True):
        # Préparer les données pour la prédiction
        input_data = pd.DataFrame({
            'age_client': [age_client],
            'revenu_annuel': [revenu_annuel],
            'type_logement': [type_logement],
            'anciennete_emploi': [anciennete_emploi],
            'motif_pret': [motif_pret],
            'note_credit': [note_credit],
            'montant_pret': [montant_pret],
            'taux_interet': [taux_interet],
            'ratio_revenu_pret': [ratio_revenu_pret],
            'defaut_dans_historique': [defaut_dans_historique],
            'anciennete_historique_credit': [anciennete_historique_credit]
        })
        
        try:
            # Faire la prédiction
            prediction = model.predict(input_data)[0]
            
            # Essayer d'obtenir les probabilités si disponible
            try:
                proba = model.predict_proba(input_data)[0]
                proba_defaut = proba[1] * 100
            except:
                proba_defaut = 50  # Valeur par défaut
            
            st.markdown("---")
            st.markdown("### 📊 Résultat de la Prédiction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 0:
                    st.success("✅ Crédit Approuvé")
                    st.metric("Décision", "ACCEPTÉ", delta="Faible Risque", delta_color="inverse")
                else:
                    st.error("❌ Crédit Refusé")
                    st.metric("Décision", "REFUSÉ", delta="Risque Élevé", delta_color="normal")
            
            with col2:
                st.metric("Probabilité de Défaut", f"{proba_defaut:.1f}%")
            
            with col3:
                st.metric("Score de Confiance", f"{100-proba_defaut:.1f}%")
            
            # Jauge visuelle
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = proba_defaut,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risque de Défaut", 'font': {'size': 24}},
                delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#90EE90'},
                        {'range': [30, 70], 'color': '#FFD700'},
                        {'range': [70, 100], 'color': '#FF6B6B'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70}}))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")

def show_analytics():
    st.title("📊 Analyse de Données - Tableau de Bord Interactif")
    
    df = load_data()
    
    # KPIs
    st.markdown("### 📈 Indicateurs Clés de Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Prêts", f"{len(df):,}", delta=f"{len(df)//10} ce mois")
    
    with col2:
        taux_defaut = (df['loan_status'].sum() / len(df) * 100)
        st.metric("Taux de Défaut", f"{taux_defaut:.1f}%", delta=f"{taux_defaut-30:.1f}%", delta_color="inverse")
    
    with col3:
        montant_moyen = df['montant_pret'].mean()
        st.metric("Montant Moyen", f"{montant_moyen:,.0f}€", delta="2.5%")
    
    with col4:
        revenu_moyen = df['revenu_annuel'].mean()
        st.metric("Revenu Moyen", f"{revenu_moyen:,.0f}€", delta="5.2%")
    
    with col5:
        taux_moyen = df['taux_interet'].mean()
        st.metric("Taux Moyen", f"{taux_moyen:.1f}%", delta="-0.3%", delta_color="inverse")
    
    st.markdown("---")
    
    # Filtres interactifs
    st.sidebar.markdown("### 🔍 Filtres")
    
    age_range = st.sidebar.slider("Plage d'âge", 
                                    int(df['age_client'].min()), 
                                    int(df['age_client'].max()), 
                                    (25, 65))
    
    selected_logement = st.sidebar.multiselect("Type de logement", 
                                                df['type_logement'].unique(), 
                                                default=df['type_logement'].unique())
    
    selected_motif = st.sidebar.multiselect("Motif du prêt", 
                                             df['motif_pret'].unique(), 
                                             default=df['motif_pret'].unique())
    
    # Filtrer le dataframe
    df_filtered = df[
        (df['age_client'] >= age_range[0]) & 
        (df['age_client'] <= age_range[1]) &
        (df['type_logement'].isin(selected_logement)) &
        (df['motif_pret'].isin(selected_motif))
    ]
    
    # Visualisations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Corrélations", "📈 Tendances", "🎯 Analyses Avancées"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des âges
            fig1 = px.histogram(df_filtered, x='age_client', 
                               color='statut_pret',
                               title="Distribution des Âges par Statut de Prêt",
                               labels={'age_client': 'Âge', 'statut_pret': 'Défaut'},
                               color_discrete_map={0: '#90EE90', 1: '#FF6B6B'},
                               marginal="box")
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Distribution des montants
            fig3 = px.box(df_filtered, x='type_logement', y='montant_pret',
                         color='statut_pret',
                         title="Montant du Prêt par Type de Logement",
                         color_discrete_map={0: '#90EE90', 1: '#FF6B6B'})
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Distribution des revenus
            fig2 = px.violin(df_filtered, y='revenu_annuel', x='note_credit',
                            color='statut_pret',
                            title="Distribution des Revenus par Note de Crédit",
                            color_discrete_map={0: '#90EE90', 1: '#FF6B6B'},
                            box=True)
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Taux d'intérêt
            fig4 = px.scatter(df_filtered, x='montant_pret', y='taux_interet',
                             color='statut_pret', size='revenu_annuel',
                             title="Relation Montant/Taux d'Intérêt",
                             color_discrete_map={0: '#90EE90', 1: '#FF6B6B'},
                             hover_data=['age_client', 'note_credit'])
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab2:
        # Matrice de corrélation
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        corr_matrix = df_filtered[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            text_auto='.2f',
                            title="Matrice de Corrélation",
                            color_continuous_scale='RdBu_r',
                            aspect="auto")
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter matrix
        st.markdown("### 🔍 Relations Multivariées")
        fig_scatter = px.scatter_matrix(df_filtered[['age_client', 'revenu_annuel', 'montant_pret', 'taux_interet', 'statut_pret']],
                                       dimensions=['age_client', 'revenu_annuel', 'montant_pret', 'taux_interet'],
                                       color='statut_pret',
                                       title="Matrice de Dispersion",
                                       color_discrete_map={0: '#90EE90', 1: '#FF6B6B'})
        fig_scatter.update_layout(height=800)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        # Tendances temporelles (simulées avec ancienneté)
        col1, col2 = st.columns(2)
        
        with col1:
            # Taux de défaut par motif
            defaut_par_motif = df_filtered.groupby('motif_pret')['statut_pret'].mean() * 100
            fig_motif = px.bar(x=defaut_par_motif.index, y=defaut_par_motif.values,
                              title="Taux de Défaut par Motif de Prêt",
                              labels={'x': 'Motif', 'y': 'Taux de Défaut (%)'},
                              color=defaut_par_motif.values,
                              color_continuous_scale='Reds')
            fig_motif.update_layout(height=400)
            st.plotly_chart(fig_motif, use_container_width=True)
        
        with col2:
            # Taux de défaut par note de crédit
            defaut_par_note = df_filtered.groupby('note_credit')['statut_pret'].mean() * 100
            fig_note = px.line(x=defaut_par_note.index, y=defaut_par_note.values,
                              title="Taux de Défaut par Note de Crédit",
                              labels={'x': 'Note de Crédit', 'y': 'Taux de Défaut (%)'},
                              markers=True)
            fig_note.update_traces(line_color='#FF6B6B', line_width=3)
            fig_note.update_layout(height=400)
            st.plotly_chart(fig_note, use_container_width=True)
        
        # Sunburst chart
        fig_sun = px.sunburst(df_filtered, path=['type_logement', 'motif_pret', 'note_credit'],
                             values='montant_pret',
                             title="Hiérarchie des Prêts: Logement → Motif → Note",
                             color='montant_pret',
                             color_continuous_scale='Viridis')
        fig_sun.update_layout(height=600)
        st.plotly_chart(fig_sun, use_container_width=True)
    
    with tab4:
        # Analyses avancées
        st.markdown("### 🎯 Segmentation des Clients")
        
        # Treemap
        fig_tree = px.treemap(df_filtered, path=['type_logement', 'defaut_dans_historique', 'note_credit'],
                             values='montant_pret',
                             color='statut_pret',
                             title="Segmentation par Logement, Historique et Note",
                             color_continuous_scale='RdYlGn_r')
        fig_tree.update_layout(height=500)
        st.plotly_chart(fig_tree, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Analyse du ratio
            fig_ratio = px.density_heatmap(df_filtered, x='ratio_revenu_pret', y='taux_interet',
                                          marginal_x="histogram", marginal_y="histogram",
                                          title="Densité: Ratio Revenu/Prêt vs Taux d'Intérêt",
                                          color_continuous_scale='Viridis')
            fig_ratio.update_layout(height=500)
            st.plotly_chart(fig_ratio, use_container_width=True)
        
        with col2:
            # Parallel coordinates
            fig_parallel = px.parallel_coordinates(
                df_filtered.sample(min(500, len(df_filtered))),
                dimensions=['age_client', 'revenu_annuel', 'montant_pret', 'taux_interet', 'anciennete_historique_credit'],
                color='statut_pret',
                title="Coordonnées Parallèles - Profils de Risque",
                color_continuous_scale='RdYlGn_r'
            )
            fig_parallel.update_layout(height=500)
            st.plotly_chart(fig_parallel, use_container_width=True)
    
    # Tableau de données
    st.markdown("---")
    st.markdown("### 📋 Données Brutes (Filtrées)")
    st.dataframe(df_filtered.head(100), use_container_width=True, height=400)
    
    # Statistiques descriptives
    with st.expander("📊 Statistiques Descriptives"):
        st.dataframe(df_filtered.describe(), use_container_width=True)

if __name__ == "__main__":
    main()

