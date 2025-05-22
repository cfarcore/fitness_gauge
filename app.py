import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
from io import BytesIO
import os
from matplotlib.colors import LinearSegmentedColormap
import altair as alt  # Importa Altair per il grafico

# === Percorsi file
DB_FILE = "db.csv"
EXERCISES_FILE = "esercizi.csv"
BENCHMARK_FILE = "benchmark_finale_completo.csv"
USERS_FILE = "utenti.csv"
COACH_CREDENTIALS = {"username": "coach", "password": "supercoach"}

# === Verify DB_FILE Path ===
DB_FILE = "db.csv"  # Ensure this points to the correct file path

# === Utility functions ===
def calcola_eta(data_nascita: date) -> int:
    oggi = date.today()
    return oggi.year - data_nascita.year - ((oggi.month, oggi.day) <
                                            (data_nascita.month, data_nascita.day))

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.error(f"File not found: {path}")
        return pd.DataFrame()

def save_csv(path: str, df: pd.DataFrame):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(path, index=False)
        st.info(f"File saved successfully: {path}")
    except Exception as e:
        st.error(f"Error saving file: {e}")

def export_excel(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Risultati", index=False)
    return buf.getvalue()

def parse_valore(val: str) -> float | None:
    """
    Parses a value string into a float. Handles formats like "10", "10:30", or invalid inputs.
    """
    try:
        # Handle simple numeric values
        return float(val)
    except ValueError:
        # Handle time format "MM:SS"
        if isinstance(val, str) and ":" in val:
            try:
                m, s = val.split(":")
                return int(m) + int(s) / 60
            except ValueError:
                return None  # Invalid time format
        # Handle other invalid formats
        return None

def normalizza_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """lowercase+strip colonne e valori stringa"""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]  # Normalizza i nomi delle colonne
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def rinomina_colonne_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Rinomina le colonne del benchmark DataFrame per corrispondere ai nomi attesi."""
    mapping = {
        "categoria": "categoria",
        "esercizio": "esercizio",
        "genere": "genere",
        "etamin": "etamin",
        "etamax": "etamax",
        "pesomin": "pesomin",
        "pesomax": "pesomax",
        "tipovalore": "tipovalore",
        "valoremin": "valoremin",
        "valoremax": "valoremax",
        "etichetta": "etichetta",
        # Gestione di nomi con maiuscole o accenti
        "EtaMin": "etamin",
        "EtaMax": "etamax",
        "PesoMin": "pesomin",
        "PesoMax": "pesomax",
        "TipoValore": "tipovalore",
        "ValoreMin": "valoremin",
        "ValoreMax": "valoremax",
        "Etichetta": "etichetta"
    }
    # Debug: stampa i nomi delle colonne originali
    print("Colonne originali nel benchmark_df:", df.columns.tolist())
    df = df.rename(columns=mapping)
    # Debug: stampa i nomi delle colonne rinominate
    print("Colonne rinominate nel benchmark_df:", df.columns.tolist())
    return df

def valuta_benchmark(categoria, esercizio, sesso, eta, peso, valore_raw,
                     benchmark_df: pd.DataFrame):
    valore = parse_valore(valore_raw)
    if valore is None:
        return None, None

    # Debug: stampa i nomi delle colonne per verificare
    print("Colonne disponibili in benchmark_df:", benchmark_df.columns.tolist())

    # normalize inputs
    cat = categoria.strip().lower()
    es = esercizio.strip().lower()
    sx = sesso.strip().lower()
    dfb = benchmark_df

    # Assicurati che i nomi delle colonne siano coerenti
    if "etamin" not in dfb.columns or "etamax" not in dfb.columns:
        raise KeyError("Le colonne 'etamin' o 'etamax' non sono presenti nel DataFrame. Verifica i nomi delle colonne.")

    # Filtra i benchmark in base alla categoria
    if cat == "forza":
        filtrati = dfb[
            (dfb["categoria"] == cat) &
            (dfb["esercizio"] == es) &
            (dfb["genere"] == sx) &
            (dfb["etamin"] <= eta) &
            (dfb["etamax"] >= eta) &
            (dfb["pesomin"] <= peso) &
            (dfb["pesomax"] >= peso)
        ]
    else:
        filtrati = dfb[
            (dfb["categoria"] == cat) &
            (dfb["esercizio"] == es) &
            (dfb["genere"] == sx) &
            (dfb["etamin"] <= eta) &
            (dfb["etamax"] >= eta)
        ]

    for _, r in filtrati.iterrows():
        mn, mx = r["valoremin"], r["valoremax"]
        lab, tipo = r["etichetta"], r["tipovalore"]
        if tipo == "ratio":
            rap = valore / peso if peso > 0 else 0  # Avoid division by zero
            if mn <= rap <= mx:
                return lab, min(rap / mx, 1.0)
        elif tipo == "valore":
            if mn <= valore <= mx:
                return lab, min((valore - mn) / (mx - mn), 1.0)
        elif tipo == "tempo":
            if mn <= valore <= mx:
                return lab, min((mx - valore) / (mx - mn), 1.0)
    return None, None

def crea_barra_orizzontale(progresso, titolo=""):
    fig, ax = plt.subplots(figsize=(6, 0.4))
    cmap = plt.get_cmap("RdYlGn")
    col = cmap(0.1 if progresso < 0.5 else (0.5 if progresso < 0.8 else 0.9))
    ax.barh(0, progresso, color=col, height=0.3)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["Base", "Medio", "Buono", "Avanzato", "Elite"], fontsize=8)
    ax.set_title(titolo, fontsize=10, loc="left", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

def crea_tubo_led(progresso: float, tube_height: float = 4, tube_width: float = 0.5, etichetta: str = ""):
    """
    Disegna un tubo LED verticale con altezza proporzionale al progresso e colori gradienti.
    - `progresso`: fra 0 e 1, percentuale di riempimento dal basso.
    - `tube_height`, `tube_width`: dimensione del canvas in pollici.
    - `etichetta`: testo da mostrare accanto al tubo.
    """
    # 1) Definizione dei colori e livelli
    colori = LinearSegmentedColormap.from_list("gradient", ["red", "orange", "yellow", "limegreen", "green"])
    livelli = ["Principiante", "Base", "Intermedio", "Buono", "Elite"]
    sezioni = len(livelli)
    altezza_sezione = 1 / sezioni  # Ogni livello occupa una frazione uguale

    # 2) Creo figura e asse
    fig, ax = plt.subplots(figsize=(tube_width, tube_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 3) Disegno il tubo con gradienti
    for i in range(sezioni):
        start = i * altezza_sezione
        end = start + altezza_sezione
        ax.add_patch(Rectangle((0, start), 1, altezza_sezione, color=colori(i / (sezioni - 1)), zorder=1))
        ax.text(1.1, (start + end) / 2, livelli[i], fontsize=8, va="center", ha="left", color="black")

    # 4) Disegno il riempimento proporzionale al progresso
    ax.add_patch(Rectangle((0, 0), 1, progresso, color=colori(progresso), zorder=2))

    # 5) Tolgo assi e aggiungo etichetta
    ax.axis("off")
    ax.text(0.5, progresso, f"{int(progresso * 100)}%", fontsize=12, va="center", ha="center", color="black", zorder=3)
    ax.text(0.5, -0.05, etichetta, fontsize=10, va="center", ha="center", color="black", zorder=3)

    return fig

def crea_grafico_radar(data, categorie, titolo=""):
    """
    Crea un grafico radar per visualizzare i progressi.
    - `data`: lista di valori normalizzati (0-1) per ogni categoria.
    - `categorie`: lista di nomi delle categorie.
    - `titolo`: titolo del grafico.
    """
    from math import pi

    # Ensure data and categories are valid
    if not data or not categorie or len(data) != len(categorie):
        raise ValueError("I dati e le categorie devono essere non vuoti e della stessa lunghezza.")

    # Normalize data to ensure values are between 0 and 1
    data = [min(max(d, 0), 1) for d in data]

    N = len(categorie)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Chiude il cerchio

    data += data[:1]  # Chiude il cerchio

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color="blue", alpha=0.25)
    ax.plot(angles, data, color="blue", linewidth=2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="gray", fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorie, fontsize=10)
    ax.set_title(titolo, fontsize=12, pad=20)

    # Add gridlines for better readability
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    return fig

# === Remove Safe Rerun Function ===
# Removed the `safe_rerun()` function to avoid using `st.experimental_rerun()`.

# === Streamlit page config & CSS
if __name__ == "__main__":
    st.set_page_config(page_title="Fitness Gauge", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Imposta il layout per dispositivi mobili */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #2c3e50; color: #ecf0f1;
        margin: 0; padding: 0; overflow-x: hidden;
    }
    .css-1d391kg, .css-1d3z3hw { background-color: #34495e; color: #ecf0f1; }
    h1, h2, h3 { color: #ecf0f1; text-align: center; }
    .stButton>button {
        background-color: #3498db; color: white; border-radius: 10px;
        padding: 0.6em 1.5em; font-weight: bold; border: none; transition: 0.3s;
        font-size: 1rem;
    }
    .stButton>button:hover { background-color: #2980b9; transform: scale(1.05); }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 10px; border: 1px solid #ccc; padding: 0.8em;
        background-color: #ffffff; color: #2c3e50; font-size: 1rem;
    }
    .stTextInput>div>div>input::placeholder, .stNumberInput>div>div>input::placeholder {
        color: #95a5a6;
    }
    .stPlotlyChart, .stAltairChart, .stVegaLiteChart {
        border-radius: 10px; padding: 1em; background-color: #34495e;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    .stDataFrame { font-size: 0.9rem; }
    .stDownloadButton>button {
        background-color: #27ae60; color: white; border-radius: 10px;
        padding: 0.6em 1.5em; font-weight: bold; border: none; transition: 0.3s;
        font-size: 1rem;
    }
    .stDownloadButton>button:hover { background-color: #1e8449; transform: scale(1.05); }
    @media (max-width: 768px) {
        .stButton>button, .stDownloadButton>button {
            width: 100%; font-size: 1rem; padding: 0.8em;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            font-size: 1rem; padding: 0.8em;
        }
        h1, h2, h3 { font-size: 1.5rem; }
    }
    </style>
""", unsafe_allow_html=True)

# === Sidebar Login/Logout
st.sidebar.title("Login")
session = st.session_state
if "logged" not in session:
    session.logged = False
    session.user = None
    session.is_coach = False

mode = st.sidebar.radio("Accesso come:", ["Utente", "Coach"])
if not session.logged:
    u = st.sidebar.text_input("Nome utente")
    p = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Accedi"):
        if mode == "Coach":
            if u == COACH_CREDENTIALS["username"] and p == COACH_CREDENTIALS["password"]:
                session.logged = True
                session.user = "Coach"
                session.is_coach = True
            else:
                st.sidebar.error("Credenziali coach errate.")
        else:
            users_df = load_csv(USERS_FILE)
            user = users_df[(users_df["Nome"] == u) & (users_df["Password"] == p)]
            if not user.empty:
                session.logged = True
                session.user = u
                session.is_coach = False
            else:
                st.sidebar.error("Credenziali utente errate.")

if session.logged:
    if st.sidebar.button("🔓 Esci"):
        session.logged = False
        session.user = None
        session.is_coach = False
        

# === Area Coach
if session.logged and session.is_coach:
    st.title("🧑‍🏫 Area Coach")
    tabs = st.tabs(["📋 Test", "🏋️ Esercizi", "🎯 Benchmark", "👥 Utenti"])

    with tabs[0]:
        st.subheader("📋 Tutti i Test")
        df = load_csv(DB_FILE)
        st.dataframe(df)
        st.download_button("📥 Scarica DB", export_excel(df), file_name="db_completo.xlsx")

    with tabs[1]:
        st.subheader("🏋️ Gestione Esercizi")
        ex_df = load_csv(EXERCISES_FILE)
        st.dataframe(ex_df)
        st.markdown("### ➕ Aggiungi nuovo esercizio")
        cat_nuovo = st.selectbox("Categoria", ["Forza", "Ginnastica", "Metabolico", "Mobilità"])
        es_nuovo = st.text_input("Nome esercizio")
        if st.button("💾 Aggiungi esercizio"):
            if not es_nuovo.strip():
                st.warning("Inserisci un nome valido.")
            elif ((ex_df["Categoria"] == cat_nuovo) & (ex_df["Esercizio"] == es_nuovo)).any():
                st.warning("Questo esercizio esiste già.")
            else:
                nuovo = pd.DataFrame([[cat_nuovo, es_nuovo]], columns=["Categoria", "Esercizio"])
                ex_df = pd.concat([ex_df, nuovo], ignore_index=True)
                save_csv(EXERCISES_FILE, ex_df)
                st.success(f"Esercizio '{es_nuovo}' aggiunto!")
                + st.query_params.refresh = "true"
        st.markdown("### ❌ Elimina esercizio")
        if not ex_df.empty:
            to_del = st.selectbox("Seleziona esercizio da eliminare", ex_df["Esercizio"].unique())
            if "confirm_delete_exercise" not in st.session_state:
                st.session_state.confirm_delete_exercise = None

            if st.button("Elimina selezionato", key="delete_exercise"):
                st.session_state.confirm_delete_exercise = to_del

            if st.session_state.confirm_delete_exercise == to_del:
                st.warning(f"Sei sicuro di voler eliminare l'esercizio '{to_del}' e i relativi test?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Conferma eliminazione", key="confirm_exercise"):
                        # Remove the exercise from the exercises file
                        ex_df = ex_df[ex_df["Esercizio"] != to_del]
                        save_csv(EXERCISES_FILE, ex_df)

                        # Remove related entries from the database file
                        db_df = load_csv(DB_FILE)
                        db_df = db_df[db_df["Esercizio"] != to_del]
                        save_csv(DB_FILE, db_df)

                        st.success(f"Esercizio '{to_del}' e i relativi test sono stati eliminati!")
                        st.session_state.confirm_delete_exercise = None
                        try:
                            st.experimental_rerun()  # Trigger a page refresh
                        except RuntimeError:
                            st.warning("Impossibile eseguire il refresh automatico. Ricarica manualmente la pagina.")
                with col2:
                    if st.button("Annulla", key="cancel_exercise"):
                        st.session_state.confirm_delete_exercise = None

    with tabs[2]:
        st.subheader("🎯 Benchmark")
        b_df = load_csv(BENCHMARK_FILE)
        st.dataframe(b_df)
        st.markdown("⚠️ Modifica diretta su CSV consigliata.")

        # === Inserimento manuale benchmark ===
        st.markdown("### ➕ Inserisci nuovo benchmark")
        with st.form("nuovo_benchmark"):
            col1, col2, col3 = st.columns(3)
            with col1:
                categoria = st.selectbox("Categoria", ["Forza", "Ginnastica", "Metabolico", "mobilità"])
                esercizio = st.text_input("Esercizio")
                genere = st.selectbox("Genere", ["maschio", "femmina"])
                etamin = st.number_input("Età minima", min_value=0, value=18)
                etamax = st.number_input("Età massima", min_value=0, value=99)
            with col2:
                pesomin = st.number_input("Peso minimo", min_value=0.0, value=0.0)
                pesomax = st.number_input("Peso massimo", min_value=0.0, value=200.0)
                valoremin = st.number_input("Valore minimo", value=0.0)
                valoremax = st.number_input("Valore massimo", value=1000.0)
            with col3:
                etichetta = st.text_input("Etichetta (es: Base, Buono, Elite)")
                tipovalore = st.selectbox("Tipo valore", ["valore", "ratio", "tempo"])
            submitted = st.form_submit_button("💾 Aggiungi benchmark")
        if submitted:
            if not esercizio.strip() or not etichetta.strip():
                st.warning("Compila tutti i campi obbligatori.")
            else:
                nuovo_bench = {
                    "categoria": categoria.strip().lower(),
                    "esercizio": esercizio.strip().lower(),
                    "genere": genere.strip().lower(),
                    "etamin": etamin,
                    "etamax": etamax,
                    "pesomin": pesomin,
                    "pesomax": pesomax,
                    "valoremin": valoremin,
                    "valoremax": valoremax,
                    "etichetta": etichetta.strip(),
                    "tipovalore": tipovalore.strip().lower()
                }
                b_df = pd.concat([b_df, pd.DataFrame([nuovo_bench])], ignore_index=True)
                save_csv(BENCHMARK_FILE, b_df)
                st.success("Benchmark aggiunto con successo!")
                + st.query_params.refresh = "true"

        # === Tab 3: Gestione Utenti ===
    with tabs[3]:
        st.subheader("👥 Gestione Utenti")
        users_df = load_csv(USERS_FILE)
        st.dataframe(users_df)

        st.markdown("### ➕ Aggiungi Nuovo Utente")
        nuovo_nome     = st.text_input("Nome", key="nuovo_nome")
        nuovo_cognome  = st.text_input("Cognome", key="nuovo_cognome")  # Aggiunto campo cognome
        nuova_password = st.text_input("Password", type="password", key="nuova_pw")
        nuovo_genere   = st.selectbox("Genere", ["Maschio", "Femmina"], key="nuovo_genere")
        nuovo_peso     = st.number_input("Peso", min_value=0.0, key="nuovo_peso")
        nuova_data     = st.date_input("Data di nascita", key="nuova_data")

        if st.button("💾 Aggiungi Utente"):
            if not nuovo_nome.strip() or not nuovo_cognome.strip() or not nuova_password:
                st.warning("Compila nome, cognome e password.")
            elif (users_df["Nome"] == nuovo_nome).any() and (users_df["Cognome"] == nuovo_cognome).any():
                st.warning("Esiste già un utente con questo nome e cognome.")
            else:
                r = {
                    "Nome": nuovo_nome,
                    "Cognome": nuovo_cognome,
                    "Password": nuova_password,
                    "Genere": nuovo_genere,
                    "Peso": nuovo_peso,
                    "DataNascita": str(nuova_data)
                }
                users_df = pd.concat([users_df, pd.DataFrame([r])], ignore_index=True)
                save_csv(USERS_FILE, users_df)
                st.success(f"Utente '{nuovo_nome} {nuovo_cognome}' aggiunto.")
                users_df = load_csv(USERS_FILE)

        st.markdown("### ❌ Elimina Utente")
        if not users_df.empty:
            da_cancellare = st.selectbox(
                "Seleziona utente da eliminare",
                users_df.apply(lambda x: f"{x['Nome']} {x['Cognome']}", axis=1),
                key="del_utente"
            )
            if st.button("Elimina Utente"):
                nome, cognome = da_cancellare.split(" ")
                users_df = users_df[~((users_df["Nome"] == nome) & (users_df["Cognome"] == cognome))]
                save_csv(USERS_FILE, users_df)
                st.success(f"Utente '{da_cancellare}' eliminato.")
        else:
            st.info("Nessun utente registrato.")
    

# === Area Utente
elif session.logged:
    # carico profilo utente
    users_df = load_csv(USERS_FILE)
    row = users_df[users_df["Nome"] == session.user]
    if row.empty:
        st.error("Utente non trovato."), st.stop()
    atleta = row.iloc[0]
    peso_def = atleta["Peso"]
    genere_def = atleta["Genere"]
    cognome_def = atleta["Cognome"]  # Aggiunto cognome
    dn = pd.to_datetime(atleta["DataNascita"]).date()

    st.sidebar.title("Links")
    pagina = st.sidebar.radio("Vai a", ["👤 Profilo", "📋 Inserisci Test", "📈 Analisi", "🗃 Storico"])  # Removed "📊 Termometri"

    if pagina == "👤 Profilo":
        st.header(f"Profilo di {session.user}")
        users_df = load_csv(USERS_FILE)
        row = users_df[users_df["Nome"] == session.user]
        if row.empty:
            st.error("Utente non trovato."), st.stop()
        atleta = row.iloc[0]
        peso_def = atleta["Peso"]
        genere_def = atleta["Genere"]
        cognome_def = atleta["Cognome"]  # Aggiunto cognome
        dn = pd.to_datetime(atleta["DataNascita"]).date()

        # Display user's full name
        st.write(f"**Nome Completo**: {session.user} {cognome_def}")
        st.write(f"**Data di nascita attuale**: {dn} — Età: {calcola_eta(dn)}")

        new_w = st.number_input("Peso", value=peso_def)
        new_g = st.selectbox("Genere", ["Maschio", "Femmina"], index=0 if genere_def == "Maschio" else 1)
        new_cognome = st.text_input("Cognome", value=cognome_def)  # Modifica cognome
        new_dn = st.date_input("Modifica data di nascita", value=dn)
        
        if st.button("💾 Salva"):
            users_df.loc[users_df["Nome"] == session.user, ["Peso", "Genere", "Cognome", "DataNascita"]] = [new_w, new_g, new_cognome, str(new_dn)]
            save_csv(USERS_FILE, users_df)
            st.success("Profilo aggiornato!")
            + st.query_params.refresh = "true"

    elif pagina == "📋 Inserisci Test":
        st.header("Inserisci Test")
        ex_df = load_csv(EXERCISES_FILE)
        gruppi = ex_df.groupby("Categoria")["Esercizio"].apply(list).to_dict()
        c_sel = st.selectbox("Categoria", list(gruppi.keys()))
        e_sel = st.selectbox("Esercizio", gruppi[c_sel])
        v = st.text_input("Valore")
        d = st.date_input("Data del test", value=date.today())
        if st.button("Salva test"):
            new_t = {
                "Nome": session.user,
                "Data": str(d),
                "Categoria": c_sel,
                "Esercizio": e_sel,
                "Valore": v,
                "Peso": peso_def,
                "Genere": genere_def,
                "DataNascita": str(dn)
            }
            df = load_csv(DB_FILE)
            df = pd.concat([df, pd.DataFrame([new_t])], ignore_index=True)
            save_csv(DB_FILE, df)
            st.success("Test salvato!")

    elif pagina == "📈 Analisi":
        st.header("Analisi dei Progressi")
        df_all = load_csv(DB_FILE)
        bdf = normalizza_dataframe(load_csv(BENCHMARK_FILE))
        df_u = df_all[df_all["Nome"] == session.user]
        if df_u.empty:
            st.info("Nessun test registrato.")
        else:
            # Create tabs for different analysis sections
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Macro Aree", "📅 Timeline", "📊 Confronto", "🎯 Benchmark"])

            with tab1:
                # Calcolo dei progressi per macro aree
                macro_aree = ["Forza", "Ginnastica", "Metabolico"]
                progressi = []
                for area in macro_aree:
                    sub = df_u[df_u["Categoria"] == area]
                    if sub.empty:
                        progressi.append(0)
                    else:
                        valori = []
                        for _, r in sub.iterrows():
                            eta = calcola_eta(datetime.strptime(r["DataNascita"], "%Y-%m-%d").date())
                            _, p = valuta_benchmark(r["Categoria"], r["Esercizio"], r["Genere"],
                                                    eta, r["Peso"], r["Valore"], bdf)
                            if p is not None:
                                valori.append(p)
                        progressi.append(np.mean(valori) if valori else 0)

                # Mostra il grafico radar
                fig = crea_grafico_radar(progressi, macro_aree, titolo="Progressi per Macro Aree")
                st.pyplot(fig)

            with tab2:
                # Timeline dei progressi
                st.subheader("📅 Timeline dei Progressi")
                esercizi = df_u["Esercizio"].unique().tolist()
                esercizio_selezionato = st.selectbox("Seleziona un esercizio", esercizi)

                if esercizio_selezionato:
                    sub_df = df_u[df_u["Esercizio"] == esercizio_selezionato].copy()
                    if sub_df.empty:
                        st.warning("Nessun test registrato per questo esercizio.")
                    else:
                        # Converti i valori in numerico
                        sub_df["ValoreNumerico"] = sub_df["Valore"].apply(parse_valore)
                        sub_df["Data"] = pd.to_datetime(sub_df["Data"])  # Assicurati che le date siano in formato datetime
                        sub_df = sub_df.sort_values("Data")  # Ordina per data

                        # Crea il grafico Altair
                        chart = alt.Chart(sub_df).mark_line(point=True).encode(
                            x=alt.X("Data:T", title="Data"),
                            y=alt.Y("ValoreNumerico:Q", title="Valore"),
                            tooltip=["Data:T", "ValoreNumerico:Q"]
                        ).properties(
                            title=f"Andamento dei Test per {esercizio_selezionato}",
                            width=450,
                            height=400
                        )

                        st.altair_chart(chart, use_container_width=True)

            with tab3:
                # Confronto tra esercizi
                st.subheader("📊 Confronto tra Esercizi")
                categoria_selezionata = st.selectbox("Filtra per categoria", ["Tutte"] + df_u["Categoria"].unique().tolist())

                # Filtra per categoria se selezionata
                if categoria_selezionata != "Tutte":
                    df_u = df_u[df_u["Categoria"] == categoria_selezionata]

                # Prendi solo il test più recente per ogni esercizio
                latest_tests = df_u.sort_values("Data", ascending=False).drop_duplicates(subset=["Esercizio"])
                if latest_tests.empty:
                    st.warning("Nessun test registrato per questa categoria.")
                else:
                    confronto_data = []
                    for _, row in latest_tests.iterrows():
                        eta = calcola_eta(datetime.strptime(row["DataNascita"], "%Y-%m-%d").date())
                        etichetta, percentuale = valuta_benchmark(
                            row["Categoria"], row["Esercizio"], row["Genere"],
                            eta, row["Peso"], row["Valore"], bdf
                        )
                        if percentuale is not None:
                            confronto_data.append({
                                "Esercizio": row["Esercizio"],
                                "Percentuale": percentuale * 100,
                                "Etichetta": etichetta,
                                "Valore Grezzo": row["Valore"],
                                "Data Test": row["Data"],
                                "Colore": "red" if percentuale < 0.5 else "orange" if percentuale < 0.8 else "green"
                            })

                    if confronto_data:
                        confronto_df = pd.DataFrame(confronto_data)

                        # Crea il grafico Altair
                        chart = alt.Chart(confronto_df).mark_bar().encode(
                            x=alt.X("Esercizio:N", title="Esercizio", sort="-y"),
                            y=alt.Y("Percentuale:Q", title="Percentuale Raggiunta (%)", scale=alt.Scale(domain=[0, 100])),
                            color=alt.Color("Colore:N", scale=None, legend=None),
                            tooltip=["Esercizio", "Percentuale", "Etichetta", "Valore Grezzo", "Data Test"]
                        ).properties(
                            title="Confronto Percentuale tra Esercizi",
                            width=450,
                            height=400
                        )

                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning("Nessun dato confrontabile per i test registrati.")

            with tab4:
                # Benchmark più vicini da raggiungere
                st.subheader("🎯 Benchmark più vicini da raggiungere")
                categorie = df_u["Categoria"].unique().tolist()
                benchmark_prossimi = []

                for categoria in categorie:
                    sub_df = df_u[df_u["Categoria"] == categoria]
                    if sub_df.empty:
                        continue

                    migliore_progresso = None
                    esercizio_selezionato = None
                    dettaglio = None

                    for _, row in sub_df.iterrows():
                        eta = calcola_eta(datetime.strptime(row["DataNascita"], "%Y-%m-%d").date())
                        etichetta_attuale, progresso_attuale = valuta_benchmark(
                            row["Categoria"], row["Esercizio"], row["Genere"],
                            eta, row["Peso"], row["Valore"], bdf
                        )

                        if progresso_attuale is not None and progresso_attuale < 1.0:
                            # Trova il prossimo livello
                            benchmark_esercizio = bdf[
                                (bdf["categoria"] == row["Categoria"].lower()) &
                                (bdf["esercizio"] == row["Esercizio"].lower()) &
                                (bdf["genere"] == row["Genere"].lower()) &
                                (bdf["etamin"] <= eta) &
                                (bdf["etamax"] >= eta) &
                                (bdf["pesomin"] <= row["Peso"]) &
                                (bdf["pesomax"] >= row["Peso"])
                            ]

                            benchmark_successivo = benchmark_esercizio[
                                benchmark_esercizio["valoremin"] > parse_valore(row["Valore"])
                            ].sort_values("valoremin").head(1)

                            if not benchmark_successivo.empty:
                                valore_richiesto = benchmark_successivo.iloc[0]["valoremin"]
                                etichetta_successiva = benchmark_successivo.iloc[0]["etichetta"]
                                differenza = valore_richiesto - parse_valore(row["Valore"])

                                if migliore_progresso is None or differenza < migliore_progresso:
                                    migliore_progresso = differenza
                                    esercizio_selezionato = row["Esercizio"]
                                    dettaglio = {
                                        "Esercizio": esercizio_selezionato,
                                        "Valore Attuale": row["Valore"],
                                        "Etichetta Attuale": etichetta_attuale,
                                        "Prossimo Livello": etichetta_successiva,
                                        "Valore Richiesto": valore_richiesto,
                                        "Differenza Mancante": differenza
                                    }

                if dettaglio:
                    benchmark_prossimi.append((categoria, dettaglio))

                if benchmark_prossimi:
                    for categoria, dettaglio in benchmark_prossimi:
                        st.markdown(f"### {categoria}")
                        st.markdown(f"- **Esercizio**: {dettaglio['Esercizio']}")
                        st.markdown(f"- **Valore Attuale**: {dettaglio['Valore Attuale']}")
                        st.markdown(f"- **Etichetta Attuale**: {dettaglio['Etichetta Attuale']}")
                        st.markdown(f"- **Prossimo Livello**: {dettaglio['Prossimo Livello']}")
                        st.markdown(f"- **Valore Richiesto**: {dettaglio['Valore Richiesto']}")
                        st.markdown(f"- **Differenza Mancante**: {dettaglio['Differenza Mancante']:.2f}")
                else:
                    st.info("Nessun benchmark prossimo da raggiungere trovato.")

    elif pagina == "🗃 Storico":
        st.header("Storico Test")
        df = load_csv(DB_FILE)  # Ensure the correct file is loaded
        his = df[df["Nome"] == session.user].sort_values("Data", ascending=False)
        st.dataframe(his)

        # —— Elimina Test ——
        st.markdown("### ❌ Elimina Test")
        choices = his.index.to_list()
        descr = {idx: f"{his.loc[idx,'Data']} – {his.loc[idx,'Esercizio']} – {his.loc[idx,'Valore']}"
                 for idx in choices}
        to_del = st.selectbox("Seleziona test da eliminare:", choices, format_func=lambda i: descr[i])
        if to_del is not None:
            if "confirm_delete_test" not in st.session_state:
                st.session_state.confirm_delete_test = None

            if st.button("Elimina test selezionato", key="delete_test"):
                st.session_state.confirm_delete_test = to_del

            if st.session_state.confirm_delete_test == to_del:
                st.markdown(
                    f"<span style='color:red; font-weight:bold;'>⚠️ Sei sicuro di voler eliminare il test '{descr[to_del]}'?</span>",
                    unsafe_allow_html=True
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Conferma eliminazione", key="confirm_test"):
                        if to_del in his.index:
                            # Remove the selected test from the DataFrame
                            df = df.drop(index=to_del)
                            save_csv(DB_FILE, df)  # Save changes to the correct file
                            st.success("Test eliminato con successo!")
                            st.session_state.confirm_delete_test = None
                            st.session_state["refresh"] = True  # Set refresh flag
                            # Reload the updated DataFrame to ensure changes are reflected
                            his = load_csv(DB_FILE)
                        else:
                            st.error("Errore: Test selezionato non valido. Assicurati di selezionare un test valido dall'elenco.")
                            st.session_state.confirm_delete_test = None
                with col2:
                    if st.button("Annulla", key="cancel_test"):
                        st.session_state.confirm_delete_test = None

        # —— Modifica Test ——
        st.markdown("### ✏️ Modifica Test")
        to_mod = st.selectbox("Seleziona test da modificare:", choices, format_func=lambda i: descr[i], key="mod")
        if to_mod is not None:
            row = his.loc[to_mod]
            new_date = st.date_input("Data del test", value=pd.to_datetime(row["Data"]).date())
            new_val = st.text_input("Valore", value=row["Valore"])
            if st.button("Salva modifiche"):
                df.loc[to_mod, "Data"] = str(new_date)
                df.loc[to_mod, "Valore"] = new_val
                save_csv(DB_FILE, df)
                st.success("Test modificato correttamente!")
                st.session_state["refresh"] = True  # Set refresh flag

        # —— Download Excel ——
        st.download_button("📥 Scarica Excel", export_excel(df[df["Nome"] == session.user]), file_name=f"{session.user}_storico.xlsx")

# Handle Refresh Logic
if "refresh" in st.session_state and st.session_state["refresh"]:
    st.session_state["refresh"] = False
    st.query_params.clear()                   # cancella vecchi param
    st.query_params.refresh = "true"          # imposta il nuovo
