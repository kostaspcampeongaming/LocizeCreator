import streamlit as st
import pandas as pd
import psycopg
from psycopg.rows import dict_row
import io
from pathlib import Path
from datetime import datetime, timezone
import time

# =========================================================
# CONFIG: your vault locale order (deduped, preserved order)
# =========================================================
RAW_LOCALES = [
    "en", "en-CA", "en-NZ", "en-AU", "de", "fi", "no", "pt", "pt-BR",
    "es", "es-CL", "it", "fr", "fr-CA", "it", "pl", "az", "ru", "tr", "sv"
]

def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

LOCALES = dedupe_preserve_order(RAW_LOCALES)

BASE_COLS = ["key", "tags", "context", "maxCharacters", "namespace"]
EXPECTED_HEADERS = BASE_COLS + LOCALES


# =========================================================
# DB
# =========================================================
def utc_now():
    return datetime.now(timezone.utc).isoformat()

def now_utc_dt():
    return datetime.now(timezone.utc)

def get_conn():
    db_url = st.secrets.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing DATABASE_URL. Add it to Streamlit Secrets or .streamlit/secrets.toml")

    conn = psycopg.connect(db_url, row_factory=dict_row)
    # Prevent long implicit transactions during Streamlit reruns (multi-user safe)
    conn.autocommit = True
    return conn

def df_from_query(conn, sql: str, params=None) -> pd.DataFrame:
    params = params or ()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()  # list[dict] due to dict_row
    return pd.DataFrame(rows)

def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            key TEXT PRIMARY KEY,
            tags TEXT NOT NULL DEFAULT '',
            context TEXT NOT NULL DEFAULT '',
            maxCharacters TEXT NOT NULL DEFAULT '',
            namespace TEXT NOT NULL DEFAULT 'translation',
            created_at TIMESTAMPTZ NOT NULL
        );

        CREATE TABLE IF NOT EXISTS translations (
            key TEXT NOT NULL REFERENCES entries(key) ON DELETE CASCADE,
            locale TEXT NOT NULL,
            value TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL,
            updated_by TEXT NOT NULL DEFAULT 'system',
            PRIMARY KEY (key, locale)
        );

        CREATE TABLE IF NOT EXISTS conflicts (
            key TEXT NOT NULL REFERENCES entries(key) ON DELETE CASCADE,
            locale TEXT NOT NULL,
            a_value TEXT NOT NULL,
            a_source TEXT NOT NULL,
            b_value TEXT NOT NULL,
            b_source TEXT NOT NULL,
            resolved_value TEXT,
            resolved_by TEXT,
            resolved_at TIMESTAMPTZ,
            PRIMARY KEY (key, locale)
        );

        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );
        """)

def reset_db(conn):
    """
    Force clean sweep (data-only) WITHOUT dropping tables.
    Much safer under concurrency than DROP TABLE.
    """
    init_db(conn)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE conflicts, translations, entries, meta;")

def set_meta(conn, k, v):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO meta(k, v) VALUES (%s, %s)
            ON CONFLICT(k) DO UPDATE SET v = EXCLUDED.v
        """, (k, v))

def get_meta(conn, k, default=None):
    with conn.cursor() as cur:
        cur.execute("SELECT v FROM meta WHERE k=%s", (k,))
        row = cur.fetchone()
    return row["v"] if row else default

def upsert_entry(conn, key, tags="", context="", maxCharacters="", namespace="translation"):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO entries(key, tags, context, maxCharacters, namespace, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(key) DO NOTHING
        """, (key, tags or "", context or "", maxCharacters or "", namespace or "translation", now_utc_dt()))

def set_translation(conn, key, locale, value, who="system"):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO translations(key, locale, value, updated_at, updated_by)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT(key, locale) DO UPDATE SET
              value=EXCLUDED.value,
              updated_at=EXCLUDED.updated_at,
              updated_by=EXCLUDED.updated_by
        """, (key, locale, value if value is not None else "", now_utc_dt(), who))

def get_translation(conn, key, locale):
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM translations WHERE key=%s AND locale=%s", (key, locale))
        row = cur.fetchone()
    return row["value"] if row else ""

def list_keys(conn, contains=""):
    if contains.strip():
        df = df_from_query(conn, "SELECT key FROM entries WHERE key ILIKE %s ORDER BY key", (f"%{contains}%",))
    else:
        df = df_from_query(conn, "SELECT key FROM entries ORDER BY key")

    if df.empty:
        return []
    return df["key"].tolist()

def list_conflicts(conn):
    return df_from_query(conn, """
        SELECT key, locale, a_source, a_value, b_source, b_value, resolved_value, resolved_by, resolved_at
        FROM conflicts
        ORDER BY key, locale
    """)

def upsert_conflict(conn, key, locale, a_value, a_source, b_value, b_source):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO conflicts(key, locale, a_value, a_source, b_value, b_source)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(key, locale) DO NOTHING
        """, (key, locale, a_value, a_source, b_value, b_source))

def resolve_conflict(conn, key, locale, resolved_value, who):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE conflicts
            SET resolved_value=%s, resolved_by=%s, resolved_at=%s
            WHERE key=%s AND locale=%s
        """, (resolved_value, who, now_utc_dt(), key, locale))
    set_translation(conn, key, locale, resolved_value, who=who)

def read_locize_xlsx(file_bytes: bytes, source_name: str):
    df = pd.read_excel(io.BytesIO(file_bytes), dtype=str).fillna("")

    for c in ["key", "namespace"]:
        if c not in df.columns:
            raise ValueError(f"{source_name}: Missing required column '{c}'")

    df["namespace"] = "translation"

    for h in EXPECTED_HEADERS:
        if h not in df.columns:
            df[h] = ""

    df = df[EXPECTED_HEADERS]
    df["key"] = df["key"].astype(str).str.strip()
    df = df[df["key"] != ""]
    return df

def bootstrap_from_files(conn, dfs_with_source, who="bootstrap"):
    # Clear conflicts
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE conflicts;")

    total_cells_written = 0
    total_conflicts = 0

    for df, src in dfs_with_source:
        for _, row in df.iterrows():
            key = row["key"]
            upsert_entry(
                conn,
                key,
                tags=row.get("tags", ""),
                context=row.get("context", ""),
                maxCharacters=row.get("maxCharacters", ""),
                namespace="translation"
            )

            for loc in LOCALES:
                incoming = str(row.get(loc, "") or "")

                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 AS one FROM translations WHERE key=%s AND locale=%s",
                        (key, loc)
                    )
                    exists_row = cur.fetchone()

                if exists_row is None:
                    set_translation(conn, key, loc, incoming, who=who)
                    total_cells_written += 1
                else:
                    existing = get_translation(conn, key, loc)
                    if incoming.strip() != "" and existing.strip() != "" and incoming.strip() != existing.strip():
                        upsert_conflict(conn, key, loc, existing, "DB", incoming, src)
                        total_conflicts += 1

    set_meta(conn, "bootstrapped_at", utc_now())
    set_meta(conn, "bootstrapped_by", who)

    return {
        "rows_processed": sum(len(df) for df, _ in dfs_with_source),
        "cells_written_initial": total_cells_written,
        "conflicts_detected": total_conflicts
    }

def build_export_df(conn, selected_locales, fill_missing_with_en=False):
    keys = list_keys(conn)
    rows = []
    for key in keys:
        row = {
            "key": key,
            "tags": "",
            "context": "",
            "maxCharacters": "",
            "namespace": "translation"
        }
        for loc in selected_locales:
            v = get_translation(conn, key, loc)
            if fill_missing_with_en and (v.strip() == "") and loc != "en":
                v = get_translation(conn, key, "en")
            row[loc] = v if v is not None else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[BASE_COLS + selected_locales]
    return df

def export_replacements(df: pd.DataFrame, selected_locales, brand_value, support_value, custom_pairs):
    def repl(s):
        t = "" if s is None else str(s)
        t = t.replace("support@BRAND", support_value)
        t = t.replace("BRAND", brand_value)
        for ph, rv in custom_pairs:
            if ph and rv:
                t = t.replace(ph, rv)
        return t

    out = df.copy()
    for loc in selected_locales:
        out[loc] = out[loc].apply(repl)
    return out

def df_to_xlsx_bytes(df: pd.DataFrame):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    bio.seek(0)
    return bio.getvalue()

def keys_with_empty_en(conn):
    df = df_from_query(conn, """
        SELECT e.key
        FROM entries e
        LEFT JOIN translations t
          ON e.key = t.key AND t.locale='en'
        WHERE COALESCE(TRIM(t.value), '') = ''
        ORDER BY e.key
    """)
    if df.empty:
        return []
    return df["key"].tolist()


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Locize Vault Tool", layout="wide")
st.title("Locize Vault Tool (Default Non-Branded Vault)")

st.sidebar.markdown("### Database (Postgres)")
conn = get_conn()
init_db(conn)

# Health counters
with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) AS c FROM entries")
    entry_count = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) AS c FROM translations")
    tr_count = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) AS c FROM conflicts WHERE resolved_value IS NULL")
    conf_count = cur.fetchone()["c"]
st.sidebar.info(f"DB rows: entries={entry_count}, translations={tr_count}, open_conflicts={conf_count}")

bootstrapped_at = get_meta(conn, "bootstrapped_at", "")
bootstrapped_by = get_meta(conn, "bootstrapped_by", "")

if bootstrapped_at:
    st.sidebar.success(f"Bootstrapped\n\n{bootstrapped_at}\n\nby {bootstrapped_by or 'unknown'}")
else:
    st.sidebar.warning("Not bootstrapped yet")

tab_boot, tab_missing, tab_conf, tab_dict, tab_export = st.tabs(
    ["Bootstrap", "Missing EN", "Conflicts", "Dictionary", "Export"]
)

# -----------------------------
# Bootstrap
# -----------------------------
with tab_boot:
    st.subheader("Bootstrap database from 3 XLSX files")
    st.write("This resets DB tables (TRUNCATE) and rebuilds from uploaded XLSX. Empty translations are kept.")

    who = st.text_input("Your name (audit)", value="web-content")

    st.markdown("#### Option A: Upload the 3 XLSX (recommended)")
    up_files = st.file_uploader("Upload 3 XLSX files", type=["xlsx"], accept_multiple_files=True)

    st.markdown("#### Option B: Load from disk (same folder as app.py)")
    if "disk_files" not in st.session_state:
        st.session_state["disk_files"] = []

    if st.button("Load key-selection-1.xlsx + key-currencies.xlsx + key-tournaments.xlsx from disk"):
        folder = Path(__file__).parent
        names = ["key-selection-1.xlsx", "key-currencies.xlsx", "key-tournaments.xlsx"]
        missing = [n for n in names if not (folder / n).exists()]
        if missing:
            st.error(f"Missing on disk: {missing}. Put them next to app.py or use upload.")
            st.session_state["disk_files"] = []
        else:
            st.session_state["disk_files"] = [(n, (folder / n).read_bytes()) for n in names]
            st.success("Loaded 3 files from disk (saved in session).")

    st.markdown("---")
    st.warning("This action will RESET the DB content in Postgres (TRUNCATE).")

    if st.button("Build / Reset DB now"):
        t0 = time.time()
        status = st.status("Bootstrapping…", expanded=True)
        try:
            status.write("Step 1: DB ping")
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok;")
                ok = cur.fetchone()["ok"]
            status.write(f"DB ping ok: {ok} (t={time.time()-t0:.2f}s)")

            status.write("Step 2: Read XLSX files")
            dfs = []

            if up_files:
                for f in up_files:
                    status.write(f"Reading: {f.name}")
                    df = read_locize_xlsx(f.getvalue(), f.name)
                    status.write(f"  rows={len(df)} cols={len(df.columns)}")
                    dfs.append((df, f.name))

            if st.session_state.get("disk_files"):
                for n, b in st.session_state["disk_files"]:
                    status.write(f"Reading: {n}")
                    df = read_locize_xlsx(b, n)
                    status.write(f"  rows={len(df)} cols={len(df.columns)}")
                    dfs.append((df, n))

            if len(dfs) == 0:
                status.update(label="No files provided", state="error")
                st.error("No files provided. Upload the 3 XLSX or load from disk.")
                st.stop()

            status.write("Step 3: Force clean sweep (TRUNCATE)")
            reset_db(conn)
            status.write(f"Reset done (t={time.time()-t0:.2f}s)")

            status.write("Step 4: Bootstrap merge/write")
            stats = bootstrap_from_files(conn, dfs, who=who.strip() or "bootstrap")
            status.write(stats)

            status.update(label="Bootstrap complete ✅", state="complete")
            st.rerun()

        except Exception as e:
            status.update(label="Bootstrap failed ❌", state="error")
            st.exception(e)

# -----------------------------
# Missing EN
# -----------------------------
with tab_missing:
    st.subheader("Keys with empty EN")
    st.write("These keys have `en` empty (or missing). Fill them here before exporting if needed.")

    ks = keys_with_empty_en(conn)
    st.write(f"Keys with empty `en`: {len(ks)}")

    if len(ks) == 0:
        st.info("No missing EN values.")
    else:
        filter_text = st.text_input("Filter keys contains", value="")
        view = [k for k in ks if filter_text.lower() in k.lower()] if filter_text.strip() else ks

        st.dataframe(pd.DataFrame({"key": view}).head(500), width="stretch")

        pick_key = st.selectbox("Pick a key to edit EN", options=view[:1000])
        current_en = get_translation(conn, pick_key, "en")
        new_en = st.text_area("EN value", value=current_en, height=120)

        who2 = st.text_input("Edited by", value="web-content", key="missing_en_who")
        if st.button("Save EN"):
            set_translation(conn, pick_key, "en", new_en, who=who2.strip() or "web-content")
            st.success("Saved.")
            st.rerun()

# -----------------------------
# Conflicts
# -----------------------------
with tab_conf:
    st.subheader("Conflicts (optional)")
    st.write("Conflicts appear only if the same key+locale has different non-empty values across bootstrap XLSX files.")

    conf = list_conflicts(conn)
    if conf.empty:
        st.info("No conflicts detected.")
    else:
        open_conf = conf[conf["resolved_value"].isna()]
        st.write(f"Total conflicts: {len(conf)} | Open: {len(open_conf)}")

        if open_conf.empty:
            st.success("All conflicts resolved ✅")
        else:
            st.dataframe(open_conf.head(500), width="stretch")

            keys = open_conf["key"].unique().tolist()
            if "conf_pick_key" not in st.session_state or st.session_state["conf_pick_key"] not in keys:
                st.session_state["conf_pick_key"] = keys[0]

            pick_key = st.selectbox("Pick key", options=keys, key="conf_pick_key")

            locales_for_key = open_conf[open_conf["key"] == pick_key]["locale"].tolist()
            if not locales_for_key:
                st.warning("Stale selection (no locales left). Click refresh.")
                if st.button("Refresh selection"):
                    st.session_state.pop("conf_pick_key", None)
                    st.session_state.pop("conf_pick_locale", None)
                    st.rerun()
            else:
                if "conf_pick_locale" not in st.session_state or st.session_state["conf_pick_locale"] not in locales_for_key:
                    st.session_state["conf_pick_locale"] = locales_for_key[0]

                pick_locale = st.selectbox("Pick locale", options=locales_for_key, key="conf_pick_locale")
                match = open_conf[(open_conf["key"] == pick_key) & (open_conf["locale"] == pick_locale)]

                if match.empty:
                    st.warning("Already resolved or stale. Click refresh.")
                    if st.button("Refresh selection", key="refresh2"):
                        st.session_state.pop("conf_pick_key", None)
                        st.session_state.pop("conf_pick_locale", None)
                        st.rerun()
                else:
                    row = match.iloc[0]
                    a_val, a_src = row["a_value"], row["a_source"]
                    b_val, b_src = row["b_value"], row["b_source"]

                    st.markdown("#### Candidates")
                    st.write({"A source": a_src, "A value": a_val})
                    st.write({"B source": b_src, "B value": b_val})

                    choice = st.radio("Choose base", options=["Use A", "Use B", "Custom"], index=0)
                    initial = a_val if choice == "Use A" else b_val if choice == "Use B" else ""
                    final = st.text_area("Resolved value (editable)", value=initial, height=120)
                    who3 = st.text_input("Resolved by", value="web-content", key="conf_who")

                    if st.button("Save resolution"):
                        resolve_conflict(conn, pick_key, pick_locale, final, who=who3.strip() or "web-content")
                        st.success("Saved.")
                        st.rerun()

# -----------------------------
# Dictionary
# -----------------------------
with tab_dict:
    st.subheader("Dictionary (view/edit any key)")
    search = st.text_input("Search key", value="")
    keys = list_keys(conn, contains=search)
    st.write(f"Keys: {len(keys)}")

    if not keys:
        st.info("No keys yet. Bootstrap first.")
    else:
        selected_key = st.selectbox("Select key", options=keys[:5000])
        st.write(f"Key: `{selected_key}`")

        rows = [{"locale": loc, "value": get_translation(conn, selected_key, loc)} for loc in LOCALES]
        st.dataframe(pd.DataFrame(rows), width="stretch", height=420)

        st.markdown("#### Edit a locale")
        edit_locale = st.selectbox("Locale", options=LOCALES, key="dict_locale")
        cur_val = get_translation(conn, selected_key, edit_locale)
        new_val = st.text_area("Value", value=cur_val, height=120, key="dict_val")
        who4 = st.text_input("Edited by", value="web-content", key="dict_who")
        if st.button("Save value"):
            set_translation(conn, selected_key, edit_locale, new_val, who=who4.strip() or "web-content")
            st.success("Saved.")
            st.rerun()

# -----------------------------
# Export
# -----------------------------
with tab_export:
    st.subheader("Export Locize-ready XLSX (vault structure)")

    chosen = st.multiselect("Locales to export", options=LOCALES, default=["en", "de", "fi"])
    if "en" not in chosen:
        st.error("'en' is mandatory and must be selected.")
        chosen = ["en"] + chosen

    fill_missing_with_en = st.checkbox("Fill missing locale with EN (fallback)", value=False)

    st.markdown("### Replacements (optional at export time)")
    brand_value = st.text_input("Replace BRAND with", value="")
    support_value = st.text_input("Replace support@BRAND with", value="")

    st.markdown("### Custom replacements (optional, 3 pairs)")
    c1a = st.text_input("Placeholder #1", value="")
    c1b = st.text_input("Value #1", value="")
    c2a = st.text_input("Placeholder #2", value="")
    c2b = st.text_input("Value #2", value="")
    c3a = st.text_input("Placeholder #3", value="")
    c3b = st.text_input("Value #3", value="")
    custom_pairs = [(c1a, c1b), (c2a, c2b), (c3a, c3b)]

    st.markdown("---")

    if st.button("Build export preview"):
        df = build_export_df(conn, selected_locales=chosen, fill_missing_with_en=fill_missing_with_en)

        empty_en = keys_with_empty_en(conn)
        if empty_en:
            st.warning(f"There are {len(empty_en)} keys with empty EN. You can still export, but QA carefully.")
            st.dataframe(pd.DataFrame({"key": empty_en}).head(200), width="stretch")

        conf_df = list_conflicts(conn)
        open_conf = conf_df[conf_df["resolved_value"].isna()] if not conf_df.empty else conf_df
        if not open_conf.empty:
            st.warning(f"There are {len(open_conf)} unresolved conflicts.")
            st.dataframe(open_conf.head(50), width="stretch")

        st.dataframe(df.head(50), width="stretch")

        if brand_value.strip() and support_value.strip():
            final_df = export_replacements(df, chosen, brand_value.strip(), support_value.strip(), custom_pairs)
        else:
            final_df = df

        final_df = final_df[BASE_COLS + chosen]

        data = df_to_xlsx_bytes(final_df)
        fname = f"locize_vault_export_{datetime.now().date().isoformat()}.xlsx"
        st.download_button(
            "Download XLSX",
            data=data,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
