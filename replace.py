import os
import re

# ============================================================
# NEW FUNCTION
# ============================================================
NEW_FUNCTION = r'''
def get_raw_predictions(symbol, timeframe):

    raw = {}

    for name, model in PREDICTORS.items():

        r = model.predict(symbol, timeframe)
        s = r.get("_summary", {})

        signal = r.get("signal") or s.get("signal") or s.get("label")
        conf = r.get("confidence", s.get("signal_conf", s.get("confidence", 0)))
        active = r.get("trade_active", s.get("trade_active", False))

        raw[name] = {
            "signal": signal,
            "confidence": conf,
            "trade_active": active
        }

        logging.info(
            "%s → signal=%s | conf=%.2f | active=%s",
            name.upper(), signal, conf, active
        )

    return raw