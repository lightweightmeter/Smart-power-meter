import threading
import time
import random
import datetime
import sqlite3
import pandas as pd
from flask import Flask, jsonify
import logging
import numpy as np
import warnings

# ===============================
# Prophet Import (with fallback)
# ===============================
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except Exception:
        PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
DB_FILE = "meter.db"

# ===============================
# Database Setup
# ===============================
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.row_factory = sqlite3.Row
with conn:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS readings
             (timestamp TEXT, voltage REAL, current REAL, power REAL, energy REAL)"""
    )


# ===============================
# Utility Functions
# ===============================
def insert_reading(r):
    try:
        with conn:
            conn.execute(
                "INSERT INTO readings VALUES (?, ?, ?, ?, ?)",
                (r["timestamp"], r["voltage"], r["current"], r["power"], r["energy"]),
            )
    except Exception:
        logging.exception("DB insert failed")


def fetch_recent_rows(limit=30):
    cur = conn.cursor()
    cur.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT ?", (limit,))
    return [dict(row) for row in cur.fetchall()]


# ===============================
# Data Simulator
# ===============================
def generate_reading(now=None):
    if now is None:
        now = datetime.datetime.now()
    hour = now.hour
    voltage = round(random.uniform(216, 230), 2)

    if 6 <= hour <= 9:
        base_current = random.uniform(1.0, 6.0)
    elif 18 <= hour <= 23:
        base_current = random.uniform(2.0, 10.0)
    else:
        base_current = random.uniform(0.2, 3.0)

    # occasional spikes
    if random.random() < 0.05:
        current = base_current + random.uniform(5, 12)
    else:
        current = base_current + random.uniform(-0.2, 0.5)

    current = max(0.01, round(current, 3))
    pf = random.uniform(0.85, 0.99)
    power = round(voltage * current * pf, 2)
    energy = round(power / 1000 / 60, 6)

    return {
        # ✅ Always store timezone-naive ISO timestamp
        "timestamp": datetime.datetime.now().replace(tzinfo=None).isoformat(timespec="microseconds"),
        "voltage": voltage,
        "current": current,
        "power": power,
        "energy": energy,
    }


def simulator_worker(interval_seconds=10):
    logging.info("Simulator started (interval=%ss)", interval_seconds)
    while True:
        try:
            r = generate_reading()
            insert_reading(r)
        except Exception:
            logging.exception("Simulator error")
        time.sleep(interval_seconds)


# Start simulator in background
sim_thread = threading.Thread(target=simulator_worker, args=(10,), daemon=True)
sim_thread.start()


# ===============================
# API Endpoints
# ===============================
@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/get_data")
def get_data():
    return jsonify(fetch_recent_rows(60))


@app.route("/get_billing")
def get_billing():
    cur = conn.cursor()
    cur.execute("SELECT SUM(energy) as total FROM readings")
    row = cur.fetchone()
    total_kwh = float(row["total"]) if row and row["total"] is not None else 0.0
    rate_per_kwh = 6.5
    cost = round(total_kwh * rate_per_kwh, 2)
    return jsonify({"total_kwh": round(total_kwh, 6), "bill_estimate": cost})


@app.route("/get_alerts")
def get_alerts():
    rows = fetch_recent_rows(30)
    alerts = []
    if not rows:
        return jsonify(alerts)

    powers = [float(r["power"]) for r in rows if r.get("power") is not None]
    if not powers:
        return jsonify(alerts)

    mean_p = float(np.mean(powers))
    std_p = float(np.std(powers))

    for i, r in enumerate(rows):
        p = float(r["power"]) if r.get("power") is not None else 0.0
        if p > 2000:
            alerts.append({"level": "critical", "msg": f"High load at {r['timestamp']}: {p} W"})
        if p < 100 and float(r.get("current", 0)) > 5:
            alerts.append(
                {"level": "warning", "msg": f"Strange current at {r['timestamp']} (I={r['current']} A, P={p} W)"}
            )
        if i < len(rows) - 1:
            prev = rows[i + 1]
            prev_p = float(prev.get("power", 0))
            if abs(p - prev_p) > max(800, 3 * std_p):
                alerts.append(
                    {"level": "warning", "msg": f"Sudden power change at {r['timestamp']}: {prev_p} W → {p} W"}
                )

    seen = set()
    unique = []
    for a in alerts:
        if a["msg"] not in seen:
            seen.add(a["msg"])
            unique.append(a)
    return jsonify(unique)


# ===============================
# Forecast Endpoint (Timezone-safe)
# ===============================
@app.route("/get_forecast")
def get_forecast():
    try:
        df = pd.read_sql_query("SELECT * FROM readings ORDER BY timestamp ASC", conn)
        if df.empty:
            return jsonify([])

        df["power"] = pd.to_numeric(df["power"], errors="coerce")
        df = df.dropna(subset=["power"])
        if df.empty:
            return jsonify([])

        # ✅ Normalize all timestamps to naive datetime
        df["ds"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["ds"] = df["ds"].dt.tz_localize(None)
        df = df.dropna(subset=["ds"])
        if df.empty:
            logging.warning("Forecast: no valid timestamps after parsing")
            return jsonify([])

        df = df.set_index("ds")["power"].resample("1T").mean().ffill().reset_index()
        df = df.rename(columns={"power": "y"})

        if len(df) < 6:
            avg = float(df["y"].mean())
            out = []
            for i in range(1, 6):
                ts = (datetime.datetime.now() + datetime.timedelta(minutes=i)).isoformat()
                out.append({"timestamp": ts, "predicted_power": round(avg + random.uniform(-50, 50), 2)})
            return jsonify(out)

        if PROPHET_AVAILABLE:
            try:
                m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
                m.fit(df)
                future = m.make_future_dataframe(periods=5, freq="min")
                forecast = m.predict(future).tail(5)
                out = []
                for _, row in forecast.iterrows():
                    out.append({
                        "timestamp": pd.to_datetime(row["ds"]).isoformat(),
                        "predicted_power": round(float(row["yhat"]), 2)
                    })
                return jsonify(out)
            except Exception as e:
                logging.exception(f"Prophet forecast failed: {e}")

        avg = float(df["y"].tail(10).mean())
        out = []
        for i in range(1, 6):
            ts = (datetime.datetime.now() + datetime.timedelta(minutes=i)).isoformat()
            out.append({"timestamp": ts, "predicted_power": round(avg + random.uniform(-50, 50), 2)})
        return jsonify(out)

    except Exception as e:
        logging.exception(f"Forecast endpoint failed: {e}")
        return jsonify({"error": "forecast_failed", "details": str(e)}), 500


# ===============================
# Summary Endpoint (Timezone-safe)
# ===============================
@app.route("/get_summary")
def get_summary():
    try:
        df = pd.read_sql_query("SELECT * FROM readings", conn)
        if df.empty:
            return jsonify([])

        df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
        df = df.dropna(subset=["energy"])
        if df.empty:
            return jsonify([])

        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["ts"] = df["ts"].dt.tz_localize(None)
        df = df.dropna(subset=["ts"])

        df["date"] = df["ts"].dt.date
        summary = df.groupby("date")["energy"].sum().reset_index()
        summary["bill"] = (summary["energy"] * 6.5).round(2)

        out = []
        for _, row in summary.iterrows():
            out.append({
                "date": str(row["date"]),
                "energy": round(float(row["energy"]), 6),
                "bill": float(row["bill"])
            })
        return jsonify(out)

    except Exception as e:
        logging.exception(f"Summary endpoint failed: {e}")
        return jsonify({"error": "summary_failed", "details": str(e)}), 500


# ===============================
# Reinforcement Learning Agent
# ===============================
class SmartMeterEnv:
    def __init__(self):
        self.state = [500.0, 6.5]
        self.time = 0
        self.max_time = 24
        self.done = False

    def reset(self):
        self.state = [float(random.randint(200, 2000)), float(random.choice([5, 6.5, 8]))]
        self.time = 0
        self.done = False
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        power, tariff = float(self.state[0]), float(self.state[1])
        if action == 1:
            power *= 0.8
        elif action == 2:
            tariff = 5.0
        cost = (power / 1000.0) * tariff
        reward = -cost
        if power > 1800:
            reward -= 5
        self.time += 1
        if self.time >= self.max_time:
            self.done = True
        self.state = [power, tariff]
        return np.array(self.state, dtype=np.float32), reward, self.done, {}


class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.9, epsilon=1.0, decay=0.99):
        self.q_table = {}
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay

    def get_state_key(self, state):
        return (int(state[0] // 50 * 50), round(float(state[1]), 1))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_key = self.get_state_key(state)
        return int(np.argmax(self.q_table.get(state_key, np.zeros(self.action_size))))

    def update(self, state, action, reward, next_state, done):
        s_key = self.get_state_key(state)
        ns_key = self.get_state_key(next_state)
        if s_key not in self.q_table:
            self.q_table[s_key] = np.zeros(self.action_size)
        if ns_key not in self.q_table:
            self.q_table[ns_key] = np.zeros(self.action_size)
        q_predict = self.q_table[s_key][action]
        q_target = reward if done else reward + self.gamma * np.max(self.q_table[ns_key])
        self.q_table[s_key][action] += self.lr * (q_target - q_predict)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.decay)


# Train RL Agent
env = SmartMeterEnv()
rl_agent = QLearningAgent(state_size=2, action_size=3)
episodes = 100
reward_history, baseline_costs, rl_costs = [], [], []

for ep in range(1, episodes + 1):
    state = env.reset()
    total_reward, cost_with_rl, cost_without_rl = 0.0, 0.0, 0.0
    while True:
        base_power, base_tariff = float(state[0]), float(state[1])
        cost_without_rl += (base_power / 1000.0) * base_tariff
        action = rl_agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        rl_agent.update(state, action, reward, next_state, done)
        cost_with_rl += (float(next_state[0]) / 1000.0) * float(next_state[1])
        state = next_state
        total_reward += reward
        if done:
            break
    rl_agent.decay_epsilon()
    reward_history.append(total_reward)
    baseline_costs.append(cost_without_rl)
    rl_costs.append(cost_with_rl)

logging.info(f"RL Agent trained, Q-table size: {len(rl_agent.q_table)}")


@app.route("/get_rl_action")
def get_rl_action():
    cur = conn.cursor()
    cur.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return jsonify({"action": "wait", "msg": "No data yet"})
    power = float(row["power"]) if row["power"] is not None else 0.0
    tariff = 6.5
    state = np.array([power, tariff])
    action = rl_agent.choose_action(state)
    action_map = {0: "Keep usage same", 1: "Reduce load by 20%", 2: "Shift load to off-peak"}
    return jsonify({"power": power, "tariff": tariff, "action": action_map.get(action, "unknown")})


@app.route("/get_rl_rewards")
def get_rl_rewards():
    return jsonify([float(r) for r in reward_history])


@app.route("/get_rl_costs")
def get_rl_costs():
    return jsonify({"baseline": [float(c) for c in baseline_costs], "rl": [float(c) for c in rl_costs]})


if __name__ == "__main__":
    logging.info("Starting Flask backend on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
