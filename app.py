import streamlit as st
import pandas as pd
import psycopg
from psycopg.rows import dict_row
import io
from pathlib import Path
from datetime import datetime, timezone
import time
import os

# =========================================================
# CONFIG
# =========================================================
RAW_LOCALES = [
    "en", "en-CA", "en-NZ", "en-AU", "de", "fi", "no", "pt", "pt-BR",
    "es", "es-CL", "it", "fr", "fr-CA", "pl", "az", "ru", "tr", "sv"
]

def dedupe(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

LOCALES = dedupe(RAW_LOCALES)
BASE_COLS = ["key", "tags", "context", "maxCharacters", "namespace"]
EXPECTED_HEADERS = BASE_COLS + LOCALES


# =========================================================
# DB helpers
# =========================================================
def now_utc():
    return datetime.now(timezone.utc)

def get_conn():
    db_url = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL missing in Streamlit Secrets")

    conn = psycopg.connect(db_url, row_factory=dict_row)
    conn.autocommit = True
    return conn

def df_from_query(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
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

def truncate_all(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE conflicts, translations, entries, meta;")

def set_meta(conn, k, v):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO meta(k, v) VALUES (%s, %s)
            ON CONFLICT(k) DO UPDATE SET v = EXCLUDED.v
        """, (k, v))

def get_meta(conn, k):
    df = df_from_query(conn, "SELECT v FROM meta WHERE k=%s", (k,))
    return df.iloc[0]["v"] if not df.empty else ""


# =========================================================
# XLSX + BULK BOOTSTRAP (ADMIN ONLY)
# =========================================================
def read_locize_xlsx(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes), dtype=str).fillna("")
    if "key" not in df.columns:
        raise ValueError("Missing column: key")

    for h in EXPECTED_HEADERS:
        if h not in df.columns:
            df[h] = ""

    df = df[EXPECTED_HEADERS]
    df["key"] = df["key"].str.strip()
    return df[df["key"] != ""]

def bulk_bootstrap(conn, dfs, who):
    t0 = time.time()

    merged = {}
    for df in dfs:
        for _, row in df.iterrows():
            merged[row["key"]] = row

    keys = list(merged.keys())
    now = now_utc()

    entries = []
    translations = []

    for k, r in merged.items():
        entries.append((k, r["tags"], r["context"], r["maxCharacters"], "translation", now))
        for loc in LOCALES:
            translations.append((k, loc, r.get(loc, ""), now, who))

    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO entries(key, tags, context, maxCharacters, namespace, created_at)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT(key) DO NOTHING
        """, entries)

        cur.executemany("""
            INSERT INTO translations(key, locale, value, updated_at, updated_by)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT(key, locale) DO UPDATE SET
              value=EXCLUDED.value,
              updated_at=EXCLUDED.updated_at,
              updated_by=EXCLUDED.updated_by
        """, translations)

    set_meta(conn, "bootstrapped_at", now.isoformat())
    set_meta(conn, "bootstrapped_by", who)

    return {
        "keys": len(entries),
        "translations": len(translations),
        "seconds": round(time.time() - t0, 2)
    }


# =========================================================
# App UI
# =========================================================
st.set_page_config(page_title="Locize Vault Tool", layout="wide")
st.title("Locize Vault Tool (Permanent Vault)")

conn = get_conn()
init_db(conn)

# --- Sidebar DB state
counts = df_from_query(conn, """
    SELECT
      (SELECT COUNT(*) FROM entries) AS entries,
      (SELECT COUNT(*) FROM translations) AS translations,
      (SELECT COUNT(*) FROM conflicts WHERE resolved_value IS NULL) AS open_conflicts
""").iloc[0]

st.sidebar.markdown("### Database (Postgres)")
st.sidebar.info(
    f"entries={counts['entries']}\n"
    f"translations={counts['translations']}\n"
    f"open_conflicts={counts['open_conflicts']}"
)

if counts["entries"] == 0:
    st.sidebar.warning("DB is empty — admin must import once.")
else:
    st.sidebar.success("DB ready ✓")

# --- Admin mode
is_admin = st.sidebar.checkbox("Admin mode")
ADMIN_PW = st.secrets.get("ADMIN_PASSWORD", "")

if is_admin:
    pw = st.sidebar.text_input("Admin password", type="password")
    is_admin = pw == ADMIN_PW

tabs = ["Dictionary", "Missing EN", "Conflicts", "Export"]
if is_admin:
    tabs = ["Admin Import"] + tabs

tab_objs = st.tabs(tabs)


# =========================================================
# ADMIN IMPORT
# =========================================================
if is_admin:
    with tab_objs[0]:
        st.subheader("Admin: One-time Import / Rebuild DB")

        files = st.file_uploader("Upload XLSX files", type=["xlsx"], accept_multiple_files=True)
        who = st.text_input("Imported by", value="admin")

        st.warning("This will ERASE and rebuild the database.")

        if st.button("RUN IMPORT"):
            if not files:
                st.error("No files uploaded.")
                st.stop()

            status = st.status("Importing…", expanded=True)
            try:
                dfs = [read_locize_xlsx(f.getvalue()) for f in files]
                status.write("Truncating DB")
                truncate_all(conn)
                status.write("Bulk inserting")
                stats = bulk_bootstrap(conn, dfs, who)
                status.write(stats)
                status.update(label="Import complete ✓", state="complete")
                st.rerun()
            except Exception as e:
                status.update(label="Import failed", state="error")
                st.exception(e)


# =========================================================
# DICTIONARY
# =========================================================
with tab_objs[-4 if is_admin else 0]:
    st.subheader("Dictionary")
    keys = df_from_query(conn, "SELECT key FROM entries ORDER BY key")
    if keys.empty:
        st.info("No keys yet.")
    else:
        selected = st.selectbox("Key", keys["key"].tolist())
        for loc in LOCALES:
            val = df_from_query(
                conn,
                "SELECT value FROM translations WHERE key=%s AND locale=%s",
                (selected, loc)
            )
            new = st.text_area(loc, value=val.iloc[0]["value"] if not val.empty else "")
            if st.button(f"Save {loc}"):
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO translations(key, locale, value, updated_at)
                        VALUES (%s,%s,%s,%s)
                        ON CONFLICT(key, locale) DO UPDATE SET
                          value=EXCLUDED.value,
                          updated_at=EXCLUDED.updated_at
                    """, (selected, loc, new, now_utc()))
                st.success("Saved")
                st.rerun()


# =========================================================
# EXPORT
# =========================================================
with tab_objs[-1]:
    st.subheader("Export")
    locales = st.multiselect("Locales", LOCALES, default=["en"])
    df = df_from_query(conn, """
        SELECT e.key, t.locale, t.value
        FROM entries e
        JOIN translations t ON e.key = t.key
    """)

    rows = []
    for k in df["key"].unique():
        r = {"key": k, "tags": "", "context": "", "maxCharacters": "", "namespace": "translation"}
        for loc in locales:
            v = df[(df.key == k) & (df.locale == loc)]
            r[loc] = v.iloc[0]["value"] if not v.empty else ""
        rows.append(r)

    out = pd.DataFrame(rows)[BASE_COLS + locales]
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        out.to_excel(w, index=False)
    bio.seek(0)

    st.download_button(
        "Download XLSX",
        data=bio.getvalue(),
        file_name="locize_export.xlsx"
    )
