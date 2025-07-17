# app.py

import argparse
import pandas as pd
from simulador import HawkesSimulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador Hawkes basado en archivo Excel")
    parser.add_argument("excel_file", type=str, help="Ruta al archivo Excel")
    parser.add_argument("start_date", type=str, help="Fecha de inicio del periodo de simulación (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, help="Fecha de fin del periodo de simulación (YYYY-MM-DD)")
    parser.add_argument("--mu_boost", type=float, default=1.0, help="Factor de multiplicación para mu durante la simulación")

    args = parser.parse_args()

    print("\n📥 Cargando y entrenando modelo...")
    hs = HawkesSimulator(args.excel_file)
    hs.fit()

    print("\n📊 Resumen del modelo entrenado:")
    summary = hs.summary()
    print(f"\n📌 Parámetros estimados del modelo")
    print(f"Mu promedio diario: {summary['mu_avg']:.4f}")
    print(f"Alfa promedio diario: {summary['alpha_avg']:.4f}")
    print(f"Decay estimado (β): {summary['decay']:.4f}")
    print(f"\n📈 Total real: {summary['total_real']}, Media diaria real: {summary['daily_avg_real']:.2f}")

    print("\n🧪 Ejecutando simulación...")
    events_sim = hs.simulate(args.start_date, args.end_date, mu_boost=args.mu_boost)
    days = (pd.to_datetime(args.end_date) - pd.to_datetime(args.start_date)).days
    print(f"\n📉 Total simulado: {len(events_sim)}, Media diaria simulada: {len(events_sim)/days:.2f}")
