# EURUSD.0 H1/H2 High-Pip Swing Ensemble (100% Cloud-Native)

This directory is an autonomous, cloud-ready deployment of the **EURUSD.0 Swing Ensemble**.
It is completely decoupled from the MetaTrader 5 desktop application and runs **24/7 in the cloud on any Linux server, container, or free web host**.

---

## 🌟 Key Features

1. **100% MT5-Free / Cloud-Native**:
   - Runs natively on Linux, macOS, or Windows.
   - Zero `.exe` or desktop terminals required.
   - Ingests real-time prices and H1/H2 multi-timeframe candles directly in memory.
2. **Fresh Inflection Threshold Filter**:
   - Detects when the course of the market **JUST changed** ($\le 2$ hours into a new structural pivot).
   - Automatically flags mature moves as `LATE - DO NOT CHASE` to ensure you only enter at the origin of the wave for maximum pip runway (50–120+ pips).
3. **Actionable High-Pip Output**:
   - Synthesizes 60 curated H1/H2 models into a single actionable signal with exact Entry, Stop Loss, Target 1, Target 2 (Runner), and Risk-to-Reward ratio (1:4 to 1:7+).

---

## 🚀 How to Deploy Online (Choose One)

### Option 1: Render.com (100% Free Tier, Zero Server Management)
1. Push this folder to a GitHub repository.
2. Log into [Render.com](https://render.com) and click **New +** ➔ **Web Service**.
3. Connect your repository.
4. Render will automatically detect `render.yaml` and `requirements.txt`.
5. Click **Create Web Service**.
6. **Done!** You receive a permanent public URL like `https://eurusd-swing-ensemble.onrender.com` that you can open from your phone anywhere in the world!

---

### Option 2: AWS Lightsail Linux ($3.50/month)
1. Create an AWS Lightsail instance with **OS Only: Ubuntu 22.04 / 24.04**.
2. Connect via SSH in your browser.
3. Clone or upload this folder to the server:
   ```bash
   sudo apt update && sudo apt install -y python3-pip
   pip install -r requirements.txt
   ./run_linux.sh
   ```
4. Open port `5000` in the Lightsail Firewall tab.
5. Open `http://<YOUR-AWS-IP>:5000` on your mobile phone!

---

### Option 3: Docker (Run Anywhere)
```bash
docker build -t swing-ensemble .
docker run -p 5000:5000 swing-ensemble
```

---

## 💻 Local Testing
To test locally in cloud-native mode:
```cmd
run_cloud_test.bat
```
or
```bash
python gui_server.py
```
Open `http://localhost:5000` in Google Chrome.
