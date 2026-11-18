"""Ejecuta ejecutar_alertas() una vez (para programarlo con cron). Usa la misma lógica que la interfaz."""
import argparse
import app

p = argparse.ArgumentParser()
p.add_argument("--maximo", type=int, default=5)
p.add_argument("--minimo-senales", type=int, default=2)
a = p.parse_args()

res = app.ejecutar_alertas(maximo=a.maximo, minimo_senales=a.minimo_senales)
print(res.to_string() if len(res) else "No hay obras pendientes de aviso con esos criterios")
print(f"DRY_RUN={app.DRY_RUN}")
