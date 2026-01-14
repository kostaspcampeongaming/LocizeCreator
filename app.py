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
        raise RuntimeError("DATABASE_URL missing in Streamlit Secrets (or env var)")

    conn = psycopg.connect(db_url, row_factory=dict_row)
    conn.autocommit = True
    return conn

def df_from_query(conn, sql, params=(), columns=None) -> pd.DataFrame:
    """
    Always returns a DataFrame with columns present.
    This prevents KeyError when query returns 0 rows.
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()  # list[dict] due to dict_row

    if not rows:
        return pd.DataFrame(columns=columns or [])
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
    df = df_from_query(conn, "SELECT v FROM meta WHERE k=%s", (k,), columns=["v"])
    return df.iloc[0]["v"] if not df.empty else ""


# =========================================================
# XLSX + BULK BOOTSTRAP (ADMIN ONLY)
# =========================================================
def read_locize_xlsx(file_bytes: bytes, source_name: str):
    df = pd.read_excel(io.BytesIO(file_bytes), dtype=str).fillna("")
    if "key" not in df.columns:
        raise ValueError(f"{source_name}: Missing required column 'key'")

    # Locize structure fields must exist (even if empty)
    for h in EXPECTED_HEADERS:
        if h not in df.columns:
            df[h] = ""

    # Maintain exact order (and ignore extra columns)
    df = df[EXPECTED_HEADERS]

    # Normalize
    df["key"] = df["key"].astype(str).str.strip()
    df["namespace"] = "translation"

    # Keep only rows with key
    df = df[df["key"] != ""]
    return df

def bulk_bootstrap(conn, dfs_with_source, who: str):
    """
    Fast import:
    - Truncates all tables (admin action)
    - Merges rows by key (later file wins if duplicates)
    - Bulk inserts entries + translations
    """
    t0 = time.time()
    merged = {}

    for df, src in dfs_with_source:
        for _, row in df.iterrows():
            k = str(row["key"]).strip()
            if not k:
                continue
            merged[k] = row

    keys = list(merged.keys())
    now = now_utc()

    entry_rows = []
    tr_rows = []

    for k in keys:
        r = merged[k]
        entry_rows.append((
            k,
            str(r.get("tags", "") or ""),
            str(r.get("context", "") or ""),
            str(r.get("maxCharacters", "") or ""),
            "translation",
            now
        ))
        for loc in LOCALES:
            tr_rows.append((
                k,
                loc,
                str(r.get(loc, "") or ""),
                now,
                who
            ))

    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO entries(key, tags, context, maxCharacters, namespace, created_at)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT(key) DO NOTHING
        """, entry_rows)

        cur.executemany("""
            INSERT INTO translations(key, locale, value, updated_at, updated_by)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT(key, locale) DO UPDATE SET
              value=EXCLUDED.value,
              updated_at=EXCLUDED.updated_at,
              updated_by=EXCLUDED.updated_by
        """, tr_rows)

        # Conflicts cleared (DB is fresh)
        cur.execute("TRUNCATE TABLE conflicts;")

    set_meta(conn, "bootstrapped_at", now.isoformat())
    set_meta(conn, "bootstrapped_by", who)

    return {
        "keys_inserted": len(keys),
        "translations_written": len(tr_rows),
        "seconds": round(time.time() - t0, 2)
    }


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Locize Vault Tool", layout="wide")
st.title("Locize Vault Tool (Permanent Vault)")

conn = get_conn()
init_db(conn)

# Sidebar DB state
counts = df_from_query(conn, """
    SELECT
      (SELECT COUNT(*)::int FROM entries) AS entries,
      (SELECT COUNT(*)::int FROM translations) AS translations,
      (SELECT COUNT(*)::int FROM conflicts WHERE resolved_value IS NULL) AS open_conflicts
""", columns=["entries", "translations", "open_conflicts"]).iloc[0]

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

# Admin mode (password stored in Streamlit Secrets)
is_admin_ui = st.sidebar.checkbox("Admin mode")
ADMIN_PW = st.secrets.get("ADMIN_PASSWORD", "")

is_admin = False
if is_admin_ui:
    pw = st.sidebar.text_input("Admin password", type="password")
    is_admin = (pw == ADMIN_PW) and bool(ADMIN_PW)

tabs = ["Dictionary", "Missing EN", "Conflicts", "Export"]
if is_admin:
    tabs = ["Admin Import"] + tabs

tab_objs = st.tabs(tabs)

# -----------------------------
# Admin Import
# -----------------------------
if is_admin:
    with tab_objs[0]:
        st.subheader("Admin: Import / Rebuild DB")

        files = st.file_uploader("Upload XLSX files", type=["xlsx"], accept_multiple_files=True)
        who = st.text_input("Imported by", value="admin")

        st.warning("This will ERASE and rebuild the database.")

        if st.button("RUN IMPORT"):
            if not files:
                st.error("No files uploaded.")
                st.stop()

            status = st.status("Importing…", expanded=True)
            try:
                status.write("Reading XLSX…")
                dfs = [(read_locize_xlsx(f.getvalue(), f.name), f.name) for f in files]
                for df, src in dfs:
                    status.write(f"{src}: rows={len(df)} unique_keys={df['key'].nunique()}")

                status.write("Truncating DB…")
                truncate_all(conn)

                status.write("Bulk inserting…")
                stats = bulk_bootstrap(conn, dfs, who.strip() or "admin")
                status.write(stats)

                status.update(label="Import complete ✓", state="complete")
                st.rerun()
            except Exception as e:
                status.update(label="Import failed", state="error")
                st.exception(e)

# Helper: pick the correct tab indexes depending on admin
def tab_index(name: str) -> int:
    # if admin, "Admin Import" shifts everything by +1
    if is_admin:
        mapping = {
            "Dictionary": 1,
            "Missing EN": 2,
            "Conflicts": 3,
            "Export": 4
        }
    else:
        mapping = {
            "Dictionary": 0,
            "Missing EN": 1,
            "Conflicts": 2,
            "Export": 3
        }
    return mapping[name]

# -----------------------------
# Dictionary
# -----------------------------
with tab_objs[tab_index("Dictionary")]:
    st.subheader("Dictionary")

    # ---- Create New Key (Beta requirement) ----
    with st.expander("➕ Create New Key", expanded=False):
        with st.form("create_new_key_form", clear_on_submit=False):
            new_key = st.text_input("Key (required, unique)", value="").strip()
            created_by = st.text_input("Created by", value="web-content").strip() or "web-content"
            copy_en_all = st.checkbox("After creation: copy EN → all locales", value=False)
            initial_en = st.text_area("Initial EN value (optional)", value="", height=100)

            submitted = st.form_submit_button("Create key")

        if submitted:
            if not new_key:
                st.error("Key is required.")
            else:
                # uniqueness check
                exists_df = df_from_query(
                    conn,
                    "SELECT 1 AS exists FROM entries WHERE key=%s",
                    (new_key,),
                    columns=["exists"]
                )
                if not exists_df.empty:
                    st.error(f"Key already exists: {new_key}")
                else:
                    now = now_utc()
                    try:
                        with conn.cursor() as cur:
                            # create entry
                            cur.execute("""
                                INSERT INTO entries(key, tags, context, maxCharacters, namespace, created_at)
                                VALUES (%s, '', '', '', 'translation', %s)
                            """, (new_key, now))

                            # create translations (empty by default)
                            tr_rows = []
                            for loc in LOCALES:
                                val = ""
                                if loc == "en" and initial_en is not None:
                                    val = initial_en
                                tr_rows.append((new_key, loc, val, now, created_by))

                            cur.executemany("""
                                INSERT INTO translations(key, locale, value, updated_at, updated_by)
                                VALUES (%s,%s,%s,%s,%s)
                                ON CONFLICT(key, locale) DO NOTHING
                            """, tr_rows)

                            # optional: copy EN to all locales (only if EN non-empty)
                            if copy_en_all and (initial_en or "").strip():
                                cur.executemany("""
                                    UPDATE translations
                                    SET value=%s, updated_at=%s, updated_by=%s
                                    WHERE key=%s AND locale=%s
                                """, [
                                    (initial_en, now, created_by, new_key, loc)
                                    for loc in LOCALES if loc != "en"
                                ])

                        st.success(f"Created key: {new_key}")
                        st.rerun()
                    except Exception as e:
                        st.exception(e)

    # ---- Existing Dictionary view/edit ----
    keys_df = df_from_query(conn, "SELECT key FROM entries ORDER BY key", columns=["key"])
    if keys_df.empty:
        st.info("No keys yet. Create one above (or ask an admin to import).")
        st.stop()

    selected = st.selectbox("Key", keys_df["key"].tolist())

    # show values
    tr_df = df_from_query(conn,
        "SELECT locale, value FROM translations WHERE key=%s ORDER BY locale",
        (selected,),
        columns=["locale", "value"]
    )

    st.dataframe(tr_df, width="stretch", height=420)

    st.markdown("#### Edit a locale")
    edit_locale = st.selectbox("Locale", options=LOCALES)
    current_val = tr_df[tr_df["locale"] == edit_locale]["value"].iloc[0] if not tr_df[tr_df["locale"] == edit_locale].empty else ""
    new_val = st.text_area("Value", value=current_val, height=120)
    edited_by = st.text_input("Edited by", value="web-content")

    if st.button("Save"):
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO translations(key, locale, value, updated_at, updated_by)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT(key, locale) DO UPDATE SET
                  value=EXCLUDED.value,
                  updated_at=EXCLUDED.updated_at,
                  updated_by=EXCLUDED.updated_by
            """, (selected, edit_locale, new_val, now_utc(), edited_by.strip() or "web-content"))
        st.success("Saved.")
        st.rerun()


# -----------------------------
# Missing EN
# -----------------------------
with tab_objs[tab_index("Missing EN")]:
    st.subheader("Missing EN")
    df_missing = df_from_query(conn, """
        SELECT e.key
        FROM entries e
        LEFT JOIN translations t
          ON e.key = t.key AND t.locale='en'
        WHERE COALESCE(TRIM(t.value), '') = ''
        ORDER BY e.key
    """, columns=["key"])

    if df_missing.empty:
        st.info("No missing EN values.")
        st.stop()

    st.write(f"Keys with empty EN: {len(df_missing)}")
    pick_key = st.selectbox("Pick key", df_missing["key"].tolist()[:2000])

    cur_en = df_from_query(conn,
        "SELECT value FROM translations WHERE key=%s AND locale='en'",
        (pick_key,),
        columns=["value"]
    )
    cur_val = cur_en.iloc[0]["value"] if not cur_en.empty else ""
    new_val = st.text_area("EN value", value=cur_val, height=120)
    edited_by = st.text_input("Edited by", value="web-content", key="missing_en_by")

    if st.button("Save EN"):
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO translations(key, locale, value, updated_at, updated_by)
                VALUES (%s,'en',%s,%s,%s)
                ON CONFLICT(key, locale) DO UPDATE SET
                  value=EXCLUDED.value,
                  updated_at=EXCLUDED.updated_at,
                  updated_by=EXCLUDED.updated_by
            """, (pick_key, new_val, now_utc(), edited_by.strip() or "web-content"))
        st.success("Saved.")
        st.rerun()

# -----------------------------
# Conflicts (kept for future)
# -----------------------------
with tab_objs[tab_index("Conflicts")]:
    st.subheader("Conflicts")
    conf_df = df_from_query(conn, """
        SELECT key, locale, a_source, a_value, b_source, b_value, resolved_value
        FROM conflicts
        ORDER BY key, locale
    """, columns=["key", "locale", "a_source", "a_value", "b_source", "b_value", "resolved_value"])

    if conf_df.empty:
        st.info("No conflicts.")
        st.stop()

    st.dataframe(conf_df, width="stretch", height=520)

# -----------------------------
# Export
# -----------------------------
with tab_objs[tab_index("Export")]:
    st.subheader("Export Locize-ready XLSX")

    locales = st.multiselect("Locales", LOCALES, default=["en"])
    if "en" not in locales:
        locales = ["en"] + locales

    # Pull values
    df = df_from_query(conn, """
        SELECT e.key, t.locale, t.value
        FROM entries e
        JOIN translations t ON e.key = t.key
    """, columns=["key", "locale", "value"])

    # Guard: empty DB
    if df.empty or "key" not in df.columns:
        st.info("No data to export yet. Ask an admin to import once.")
        st.stop()

    rows = []
    for k in df["key"].unique():
        r = {"key": k, "tags": "", "context": "", "maxCharacters": "", "namespace": "translation"}
        for loc in locales:
            v = df[(df["key"] == k) & (df["locale"] == loc)]
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
        file_name="locize_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
