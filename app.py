import os
import time
import json
import sqlite3
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from io import StringIO
from datetime import datetime
from pprint import pprint
from pulp import (
    LpProblem,
    LpMaximize,
    LpVariable,
    LpBinary,
    lpSum,
    PULP_CBC_CMD,
)

# Optional ML model imports
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_RF = True
except Exception:
    HAS_RF = False

# -------------- Config --------------
CACHE_DB = "fpl_cache.db"
DEFAULT_BUDGET = 100.0
FORMATIONS = {
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
}
POS_FROM_ID = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# --------------- Robust Utilities --------------
def ensure_cache_schema():
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()

def cache_get(key, default=None):
    ensure_cache_schema()
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("SELECT value FROM cache WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except Exception:
        return default

def cache_set(key, value):
    ensure_cache_schema()
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("REPLACE INTO cache (key, value) VALUES (?, ?)", (key, json.dumps(value)))
    conn.commit()
    conn.close()

def safe_request(url, retries=3, backoff=2.0, timeout=20):
    """Simple retry wrapper with exponential backoff."""
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i < retries - 1:
                time.sleep(backoff ** i)
            else:
                raise e

# --------------- Data Layer --------------
@st.cache_data(show_spinner=False)
def fetch_bootstrap_static():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = safe_request(url)
    players = data["elements"]
    teams = {t["id"]: t["name"] for t in data["teams"]}

    records = []
    for p in players:
        pid = p["id"]
        name = p["web_name"]
        full_name = p["second_name"]
        team_id = p["team"]
        pos_id = p["element_type"]
        pos = POS_FROM_ID.get(pos_id, "UNK")
        now_cost = p["now_cost"] / 10.0  # price in millions
        team_name = teams.get(team_id, "Unknown")
        records.append({
            "id": pid,
            "name": name,
            "full_name": full_name,
            "team_id": team_id,
            "team_name": team_name,
            "position": pos,
            "price": round(now_cost, 2),
            "selected_by_percent": p.get("selected_by_percent", 0.0),
            "web_name": name
        })
    df = pd.DataFrame.from_records(records)
    return df

@st.cache_data(show_spinner=False)
def fetch_element_history(p_id):
    """Fetch the per-player history (current season) and return a list of points per GW."""
    url = f"https://fantasy.premierleague.com/api/element-summary/{p_id}/"
    try:
        data = safe_request(url)
        history = data.get("history", [])
        pts = [gw.get("points", 0) for gw in history]
        last5 = pts[-5:] if len(pts) >= 5 else pts
        return last5
    except Exception:
        return []

def predict_points_for_player_baseline(history_points):
    """Baseline predictor: weighted mean of last several games (simple MVP)."""
    if len(history_points) == 0:
        base = 1.0
    else:
        # more weight to recent games
        weights = np.array([1, 2, 3, 4, 5][-len(history_points):], dtype=float)
        w = weights / weights.sum()
        base = float(np.dot(history_points, w))
    return max(0.0, base * 0.9 + 0.6)

def load_trained_predictor():
    """Attempt to load a trained predictor if present."""
    paths = [
        "models/fpl_predictor.pkl",
        "models/fpl_predictor.joblib",
        "models/fpl_predictor.xgb"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                import joblib
                model = joblib.load(p)
                return ("model", model)
            except Exception:
                try:
                    import pickle
                    with open(p, "rb") as f:
                        model = pickle.load(f)
                    return ("model", model)
                except Exception:
                    continue
    return None

def compute_features_for_player(row, history_points, horizon_gw=1):
    """Create a small feature set for a player to feed into a predictor."""
    recent_pts = history_points[-5:]
    last5_mean = float(np.mean(recent_pts)) if recent_pts else 0.0
    last5_len = len(recent_pts)
    minutes_rough = max(10.0, min(540.0, 90 * last5_len + 30))  # rough proxy
    prev_price = float(row.get("price", 0.0))
    feat = {
        "price": prev_price,
        "minutes_proxy": minutes_rough,
        "last5_mean_pts": last5_mean,
        "last5_len": last5_len,
        "team_strength": 0.0,  # placeholder (could be derived from team form)
        "position": row.get("position", "UNK")
    }
    return feat

def build_features_df(players_df, horizon_gw=1):
    """Return a DataFrame of features for each player for ML inference."""
    feats = []
    for _, row in players_df.iterrows():
        his = fetch_element_history(row["id"])
        feat = compute_features_for_player(row, his, horizon_gw)
        feats.append(feat)
    df = pd.DataFrame(feats)
    df = df.fillna(0.0)
    return df

def predict_points_with_model(model_info, X_df, horizon_gw=1):
    mode, model = model_info
    if mode != "model":
        return None
    try:
        preds = model.predict(X_df)
        return preds
    except Exception:
        return None

def predict_points_for_player(row, history_points, horizon_gw=1, model_info=None):
    """Return a single predicted point value for a player."""
    if model_info:
        # Build features and predict
        feat = compute_features_for_player(row, history_points, horizon_gw)
        X = pd.DataFrame([feat])
        pred = predict_points_with_model(model_info, X, horizon_gw)
        if pred is not None:
            return float(pred[0])
    # Fallback to baseline if no model
    return predict_points_for_player_baseline(history_points)

# -------------- Optimization Helpers --------------
def optimize_squad(players_df, pred_points, horizon_gw=1, budget=DEFAULT_BUDGET):
    """
    Stage A: pick 15 players with fixed quotas; 2 GK, 5 DEF, 5 MID, 3 FWD.
    - max 3 players per real club
    - total price <= budget
    """
    players = players_df.copy()
    players["pred"] = pred_points

    ids = players["id"].tolist()
    price = {row["id"]: float(row["price"]) for idx, row in players.iterrows()}
    pos = {row["id"]: row["position"] for idx, row in players.iterrows()}
    team = {row["id"]: row["team_name"] for idx, row in players.iterrows()}

    # Stage A: MILP
    prob = LpProblem("FPL_Squad_StageA", LpMaximize)

    x = {pid: LpVariable(f"x_{pid}", cat=LpBinary) for pid in ids}

    prob += lpSum([players.loc[players["id"] == pid, "pred"].values[0] * x[pid] for pid in ids])

    # Constraints
    prob += lpSum([x[pid] for pid in ids]) == 15
    prob += lpSum([price[pid] * x[pid] for pid in ids]) <= budget

    # Club limits
    clubs = {}
    for idx, row in players.iterrows():
        c = row["team_name"]
        clubs.setdefault(c, []).append(row["id"])
    for club_name, pids in clubs.items():
        prob += lpSum([x[pid] for pid in pids]) <= 3

    # Position quotas
    pos_targets = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for ptype, target in pos_targets.items():
        pids = players[players["position"] == ptype]["id"].tolist()
        if pids:
            prob += lpSum([x[pid] for pid in pids]) == target

    solver = PULP_CBC_CMD(msg=False, timeLimit=20)
    prob.solve(solver)

    chosen_ids = []
    for pid in ids:
        if x[pid].value() is not None and x[pid].value() > 0.5:
            chosen_ids.append(pid)

    squad = players[players["id"].isin(chosen_ids)].copy()
    return squad

def optimize_starting_11(squad_df, formation="4-4-2", pred_map=None):
    """Stage B: Start 11 from the 15 based on a formation; returns start and bench ids and captain."""
    if pred_map is None:
        pred_map = {row["id"]: row.get("pred", 0.0) for idx, row in squad_df.iterrows()}

    formation_counts = FORMATIONS.get(formation, FORMATIONS["4-4-2"])
    ids = squad_df["id"].tolist()

    prob = LpProblem("FPL_Starting11", LpMaximize)

    s = {pid: LpVariable(f"s_{pid}", cat=LpBinary) for pid in ids}
    cap = {pid: LpVariable(f"cap_{pid}", cat=LpBinary) for pid in ids}

    pred = {row["id"]: float(row.get("pred", 0.0)) for idx, row in squad_df.iterrows()}
    prob += lpSum([pred[pid] * (s[pid] + cap[pid]) for pid in ids])

    prob += lpSum([s[pid] for pid in ids]) == 11
    prob += lpSum([cap[pid] for pid in ids]) == 1
    for pid in ids:
        prob += cap[pid] <= s[pid]

    for pos, count in formation_counts.items():
        pos_ids = squad_df[squad_df["position"] == pos]["id"].tolist()
        if pos_ids:
            prob += lpSum([s[pid] for pid in pos_ids]) == count

    solver = PULP_CBC_CMD(msg=False, timeLimit=20)
    prob.solve(solver)

    start_ids = [pid for pid in ids if s[pid].value() is not None and s[pid].value() > 0.5]
    captain_id = None
    for pid in ids:
        if cap[pid].value() is not None and cap[pid].value() > 0.5:
            captain_id = pid
            break

    bench_ids = [pid for pid in ids if pid not in start_ids]

    return start_ids, captain_id, bench_ids

# ------------- Chip Planning Module --------------
def plan_chips(squad_df, pred_series, horizon_weeks, formation="4-4-2", budget=DEFAULT_BUDGET):
    """
    Short-horizon chip planning (1-4 weeks).
    Chips:
      - Wildcard: reoptimize squad completely (within budget/formation)
      - Free Hit: like Wildcard but only for next GW; here we emulate across horizon by single GW
      - Bench Boost: bench points are doubled for the upcoming GW
      - Triple Captain: captain's points tripled
    Returns: a dict of chip_name -> uplift_points, rationale
    """
    # Current baseline
    baseline_points = float(np.sum(pred_series))

    # 1) Wildcard uplift: re-optimize 15 with same constraints
    pool_df = squad_df.copy()
    wildcard_squad = optimize_squad(pool_df, pool_df["pred"] if "pred" in pool_df.columns else pred_series, horizon_weeks, budget)
    wildcard_points = float(wildcard_squad["pred"].sum()) if "pred" in wildcard_squad.columns else 0.0
    uplift_wildcard = max(0.0, wildcard_points - baseline_points)

    # 2) Free Hit (emulated)
    free_hit_squad = optimize_squad(pool_df, pool_df["pred"] if "pred" in pool_df.columns else pred_series, horizon_weeks, budget)
    free_hit_points = float(free_hit_squad["pred"].sum()) if "pred" in free_hit_squad.columns else 0.0
    uplift_free_hit = max(0.0, free_hit_points - baseline_points)

    # 3) Bench Boost
    top11, captain_id, bench_ids = optimize_starting_11(squad_df, formation)
    bench_points = float(squad_df[squad_df["id"].isin(bench_ids)]["pred"].sum()) if len(bench_ids) > 0 else 0.0
    uplift_bench_boost = bench_points  # rough proxy

    # 4) Triple Captain
    if "pred" in squad_df.columns:
        captain_candidates = squad_df.sort_values(by="pred", ascending=False).head(11)
        best_captain_pred = captain_candidates["pred"].max() if not captain_candidates.empty else 0.0
    else:
        best_captain_pred = 0.0
    uplift_triple_captain = best_captain_pred

    chips = [
        {"chip": "Wildcard", "uplift": uplift_wildcard, "reason": "Full squad churn to maximize horizon points."},
        {"chip": "Free Hit", "uplift": uplift_free_hit, "reason": "One-week reset; same idea as wildcard but for a single GW."},
        {"chip": "Bench Boost", "uplift": uplift_bench_boost, "reason": "Doubles points for bench players in next GW."},
        {"chip": "Triple Captain", "uplift": uplift_triple_captain, "reason": "Triple points for one captain in next GW."},
    ]

    best = max(chips, key=lambda c: c["uplift"])
    return {
        "best_chip": best["chip"],
        "uplift": best["uplift"],
        "best_reason": best["reason"],
        "all_options": chips
    }

# ------------- Transfers Plan --------------
def parse_current_squad_csv(csv_text_or_file):
    """Parse a small CSV with fields: id,name,position,team_name,price,pred"""
    if isinstance(csv_text_or_file, str):
        df = pd.read_csv(StringIO(csv_text_or_file))
    else:
        df = pd.read_csv(csv_text_or_file)
    required = {"id","name","position","team_name","price","pred"}
    if not required.issubset(set(df.columns)):
        return None
    return df

def compute_transfers(current_squad_df, target_squad_df, budget=DEFAULT_BUDGET):
    """
    current_squad_df: columns id, name, position, team_name, price, pred
    target_squad_df: same shape (from optimizer)
    Returns a list of transfers: {sell_id, buy_id, delta_pred, delta_cost}
    """
    cur_ids = set(current_squad_df["id"].astype(int).tolist())
    tgt_ids = set(target_squad_df["id"].astype(int).tolist())

    to_buy = [row for _, row in target_squad_df.iterrows() if int(row["id"]) not in cur_ids]
    to_sell = [row for _, row in current_squad_df.iterrows() if int(row["id"]) not in tgt_ids]

    buys = {r["id"]: r for r in to_buy}
    sells = {r["id"]: r for r in to_sell}

    best_transfers = []
    current_cost = sum(current_squad_df["price"].tolist())

    from itertools import combinations, product

    candidates = list(buys.values())
    if not candidates:
        return []

    # 1-transfer options
    for b in candidates:
        cost_change = float(b["price"]) - (float(next(iter(sells.values())))["price"] if sells else 0.0)
        s_p = max(sells.values(), key=lambda x: x["pred"]) if sells else None
        pred_gain = float(b["pred"]) - (float(s_p["pred"]) if s_p else 0.0)
        new_cost = current_cost + cost_change
        if new_cost <= budget:
            best_transfers.append({
                "sell_id": int(s_p["id"]) if s_p else None,
                "buy_id": int(b["id"]),
                "delta_pred": pred_gain,
                "delta_cost": cost_change,
            })

    # 2-transfer options
    if len(candidates) >= 2:
        for b1, b2 in combinations(candidates, 2):
            total_buy_cost = float(b1["price"]) + float(b2["price"])
            total_pred = float(b1["pred"]) + float(b2["pred"])
            worst_sell = max(sells.values(), key=lambda x: x["pred"]) if sells else None
            pred_gain = total_pred - (float(worst_sell["pred"]) if worst_sell else 0.0)
            cost_change = total_buy_cost - (float(worst_sell["price"]) if worst_sell else 0.0)
            new_cost = current_cost + cost_change
            if new_cost <= budget:
                best_transfers.append({
                    "sell_id": int(worst_sell["id"]) if worst_sell else None,
                    "buy_id": int(b1["id"]),
                    "delta_pred": pred_gain,
                    "delta_cost": cost_change,
                })
                best_transfers.append({
                    "sell_id": int(worst_sell["id"]) if worst_sell else None,
                    "buy_id": int(b2["id"]),
                    "delta_pred": pred_gain,
                    "delta_cost": cost_change,
                })

    # Pick top 3 unique transfers by highest delta_pred
    unique = {}
    for t in best_transfers:
        key = (t["buy_id"], t["sell_id"])
        if key not in unique or unique[key]["delta_pred"] < t["delta_pred"]:
            unique[key] = t
    top = sorted(unique.values(), key=lambda x: x["delta_pred"], reverse=True)[:3]
    return top

# ------------- Generative AI Guidance --------------
def ai_guidance(prompt_text, max_tokens=500, model="gpt-5-nano-2025-08-07"):
    """Call OpenAI API if key is provided; otherwise return offline guidance."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return (
            "AI Coach (offline): Based on the current recommended squad, the top considerations are:\n"
            f"- Ensure budget elasticity: current budget ~{DEFAULT_BUDGET}, leave margin for price changes.\n"
            f"- The predicted points show players X,Y,Z as high-performers; double-check risk (rotation, injuries).\n"
            "Tip: Use chip planning results to time Wildcards/Bench Boost around double gameweeks."
        )

    try:
        import requests
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a data-driven FPL coach."},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": max_tokens,
        }
        resp = requests.post("https://api.euron.one/api/v1/euri/chat/completions", headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "AI guidance failed to fetch (API error). You can still rely on the model's numeric outputs and heuristics."

# ------------- UI + App Logic --------------
def main():
    st.set_page_config(page_title="FPL AI Predictor & Team Optimizer (MVP)", layout="wide")
    st.title("FPL AI Predictor & Team Optimizer (MVP)")
    # EPL logo header (local or online)
    st.image("https://upload.wikimedia.org/wikipedia/en/e/e1/Premier_League_Logo.png", width=200)

    # 1) Data fetch
    with st.spinner("Fetching latest FPL data (bootstrap-static)…"):
        try:
            players_df = fetch_bootstrap_static()
        except Exception as e:
            st.error(f"Data fetch failed: {e}. Using cached data if available.")
            cached = cache_get("players_bootstrap")
            if cached is not None:
                players_df = pd.DataFrame.from_records(cached)
            else:
                st.stop()

    # 2) History + predictions
    st.sidebar.header("Settings")
    horizon_gw = st.sidebar.number_input("Horizon (games ahead)", min_value=1, max_value=4, value=1, step=1)
    formation = st.sidebar.selectbox("Starting XI formation", list(FORMATIONS.keys()), index=0)
    budget = st.sidebar.number_input("Budget (million)", min_value=50.0, max_value=150.0, value=DEFAULT_BUDGET, step=0.5)

    # Optional: load trained predictor
    model_info = load_trained_predictor()

    # Build per-player history and predictions
    st.info("Predictor: using a simple baseline by default; a trained model can be loaded if present.")
    preds = []
    # Limit to a reasonable subset for MVP responsiveness
    if len(players_df) > 350:
        players_df = players_df.sort_values(by="price", ascending=False).head(350)

    # We'll fetch history for each visible player (could be slow; you can cache more aggressively)
    with st.spinner("Fetching per-player history (last 5 GWs)…"):
        for _, row in players_df.iterrows():
            his = fetch_element_history(int(row["id"]))
            pred = predict_points_for_player(row, his, horizon_gw, model_info)
            preds.append(pred)

    players_df["pred_next_gw"] = preds
    # Also create a friendly alias for display
    players_df["PredNextGW"] = players_df["pred_next_gw"]

    # Sort by predicted points
    players_df = players_df.sort_values(by="pred_next_gw", ascending=False).reset_index(drop=True)

    # 3) Stage A: 15-player squad
    squad = optimize_squad(players_df, players_df["pred_next_gw"], horizon_gw, budget)
    if squad.empty:
        st.error("Optimization failed to find a valid squad with the current constraints. Try relaxing budget or formation.")
        return

    # 4) Stage B: Starting XI
    start_ids, captain_id, bench_ids = optimize_starting_11(squad, formation)

    # Display Stage A (squad)
    squad_display = squad[["id","name","team_name","position","price","pred_next_gw"]].rename(columns={
        "id":"ID","name":"Player","team_name":"Club","position":"POS","price":"Price","pred_next_gw":"PredNextGW"
    })
    squad_display["Price"] = squad_display["Price"].round(2)
    squad_display["PredNextGW"] = squad_display["PredNextGW"].round(2)

    st.header("Stage A: 15-player Squad (MVP)")
    st.dataframe(squad_display)

    # Stage B: Starting XI (robust rename first, then display)
    starting11 = squad[squad["id"].isin(start_ids)].copy()
    display11 = starting11.rename(columns={
        "name": "Player",
        "team_name": "Club",
        "position": "POS",
        "price": "Price",
        "pred_next_gw": "PredNextGW"
    })
    starter_display = display11[["id","Player","Club","POS","Price","PredNextGW"]]
    starter_display["Price"] = starter_display["Price"].astype(float).round(2)
    starter_display["PredNextGW"] = starter_display["PredNextGW"].astype(float).round(2)

    st.markdown("---")
    st.header(f"Stage B: Starting XI ({formation})")
    if starter_display.empty:
        st.warning("Could not form a starting XI with current formation and squad. Check constraints.")
    else:
        st.dataframe(starter_display)

        captain_name = squad[squad["id"] == captain_id]["name"].values[0] if captain_id in squad["id"].values else "N/A"
        st.subheader("Captain and VC suggestion")
        st.write(f"Captain for this GW (current plan): {captain_name}")

    # 4) Chips Advisor (Chip Planning Module)
    st.header("Chips Advisor (Short-horizon planning)")
    chip_plan = plan_chips(squad, squad["pred_next_gw"] if "pred_next_gw" in squad.columns else squad["PredNextGW"], horizon_gw, formation, budget)
    st.write(f"Best chip: {chip_plan['best_chip']}")
    st.write(f"Expected uplift (points): {chip_plan['uplift']:.2f}")
    st.write(f"Reason: {chip_plan['best_reason']}")
    st.write("All options (uplifts):")
    opt_df = pd.DataFrame(chip_plan["all_options"])
    if not opt_df.empty:
        opt_df = opt_df[["chip","uplift","reason"]].rename(columns={
            "chip":"Chip","uplift":"UpliftPts","reason":"Rationale"
        })
        opt_df["UpliftPts"] = opt_df["UpliftPts"].round(2)
        st.dataframe(opt_df.style.format({"UpliftPts": "{:.2f}"}))

    # 5) Transfers Planner
    st.markdown("---")
    st.header("Transfers Planner (from current squad to MVP squad)")
    st.write("Provide your current squad as CSV (id,name,position,team_name,price,pred).")
    current_csv = st.text_area("Paste your current 15-player squad CSV here (or leave blank to skip).", height=200)
    if current_csv.strip():
        current_squad_df = parse_current_squad_csv(current_csv)
        if current_squad_df is None:
            st.error("Current squad CSV is invalid. Ensure it has columns: id,name,position,team_name,price,pred")
        else:
            # Build target (MVP) squad as minimal transfer target
            target_squad_df = squad[["id","name","position","team_name","price","pred_next_gw"]].copy()
            target_squad_df = target_squad_df.rename(columns={
                "name":"name","team_name":"team_name","price":"price","pred_next_gw":"pred","position":"position"
            })
            transfers = compute_transfers(current_squad_df, target_squad_df)
            if not transfers:
                st.info("No transfers needed based on current input.")
            else:
                t_df = pd.DataFrame(transfers)
                t_df = t_df.rename(columns={"sell_id":"Sell_ID","buy_id":"Buy_ID","delta_pred":"DeltaPred","delta_cost":"DeltaCost"})
                st.dataframe(t_df)
                st.write("Notes:")
                st.write("- Transfers shown are top options given a 100m budget and current squad prices.")
                st.write("- You can tweak the current squad or adjust budget to explore more options.")

    # 6) Generative AI Guidance
    st.header("Generative AI Coach (guidance)")
    ai_enabled = st.checkbox("Enable Generative AI guidance (OpenAI)" , value=False)
    if ai_enabled:
        prompt_text = (
            "Given the current MVP recommendations (Stage A 15-player squad, Stage B starting XI, "
            f"formation {formation}, horizon {horizon_gw} GW, budget {budget:.2f}m), "
            "provide constructive feedback on the squad and chips plan. Highlight rotation risk, "
            "injury concerns, and any potential better alternatives. "
            "Offer concise, actionable suggestions and explain reasoning briefly."
        )
        guidance = ai_guidance(prompt_text)
        st.info(guidance)

    # 7) Utilities
    st.markdown("---")
    st.write("Notes:")
    st.write("- This MVP enforces FPL squad rules and offers a clear end-to-end flow with warnings when infeasible.")
    st.write("- The predictor is pluggable: replace the baseline with a trained model (load via models/).")
    st.write("- The chip planner is a short-horizon helper; adapt horizons, rules, and penalties as needed.")
    st.write("- For robust production use, consider more advanced data caching, error handling, and rate-limiting.")

    # Visualization: top 10 players by PredNextGW
    top10 = players_df.nlargest(10, "PredNextGW")
    if not top10.empty:
        vis = top10.rename(columns={"name": "Player"})
        chart = alt.Chart(vis).mark_bar().encode(
            x=alt.X("Player:N", sort=None),
            y="PredNextGW:Q",
            color="position:N"
        ).properties(width=700, height=300)
        st.altair_chart(chart, use_container_width=True)

    # Optional: Clear cache button (for testing)
    if st.button("Reset Local Cache"):
        try:
            if os.path.exists(CACHE_DB):
                os.remove(CACHE_DB)
            st.success("Cache reset. Data will be re-fetched on next run.")
        except Exception as e:
            st.error(f"Failed to reset cache: {e}")

if __name__ == "__main__":
    main()