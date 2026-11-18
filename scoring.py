"""Puntúa un CSV nuevo de obras con los modelos ya entrenados en Colab (modelos/sistema_obras.joblib).
Es la misma lógica del notebook (preparar + puntuar_obras), completada con las columnas que usa la interfaz."""
import os
import numpy as np
import pandas as pd
import joblib

RUTA_MODELO = os.environ.get('MODELO', os.path.join('modelos', 'sistema_obras.joblib'))

VARS_AE = ['log_monto', 'log_dur', 'avancefisico', 'avancefinanciero']
NUM_RF = ['log_dur', 'fechainicioanio', 'avancefisico', 'avancefinanciero', 'clima', 'prog']
CAT_RF = ['sectornombre', 'nombreprovincia', 'entidad']

# Columnas que debe traer el CSV (las mismas del dataset MOP con el que se entrenó)
REQUERIDAS = ['idproyecto', 'nombreobra', 'nombreprovincia', 'sectornombre', 'entidadejecutoranombre', 'etapaobra',
              'montototal', 'duracionobrasdias', 'fechainicioanio', 'avancefisico', 'avancefinanciero',
              'tag_accionclimatica', 'programa_infraestructura', 'moneda_dolares']

_S = None
def cargar():
    """Carga el sistema (escalador, autoencoder, Random Forest, umbrales) una sola vez."""
    global _S
    if _S is None:
        _S = joblib.load(RUTA_MODELO)
    return _S

def puntuar(obras):
    """Devuelve (obras puntuadas, mensaje). Lanza ValueError si faltan columnas."""
    S = cargar()
    faltan = [c for c in REQUERIDAS if c not in obras.columns]
    if faltan:
        raise ValueError('Al CSV le faltan estas columnas: ' + ', '.join(faltan))
    x = obras.copy()
    n0 = len(x)

    # Igual que en el notebook: solo obras en pesos
    x = x[x['moneda_dolares'] == 0]
    n_usd = n0 - len(x)
    # Filas sin los datos numéricos básicos no se pueden puntuar
    for c in ['montototal', 'duracionobrasdias', 'fechainicioanio', 'avancefisico', 'avancefinanciero']:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    ok = (x['montototal'] > 0) & (x['duracionobrasdias'] > 0) & x[['fechainicioanio', 'avancefisico', 'avancefinanciero']].notna().all(axis=1)
    n_inval = int((~ok).sum())
    x = x[ok].reset_index(drop=True)
    if len(x) == 0:
        raise ValueError('Ninguna obra del CSV se pudo puntuar (revisá monto, duración, años y avances).')

    # Campos que usa la tarjeta de la interfaz
    x['fechafinanio'] = pd.to_numeric(x.get('fechafinanio'), errors='coerce').fillna(x['fechainicioanio']) if 'fechafinanio' in x else x['fechainicioanio']
    for c in ['nombredepto', 'url_perfil_obra']:
        x[c] = x[c].fillna('') if c in x else ''
    if 'diferencia_avance' not in x:
        x['diferencia_avance'] = x['avancefinanciero'] - x['avancefisico']

    # ── preparar (notebook 3.1) ──
    x['log_monto'] = np.log10(x['montototal'])
    x['log_dur'] = np.log10(x['duracionobrasdias'])
    x['clima'] = x['tag_accionclimatica'].notna().astype(int)
    x['prog'] = (x['programa_infraestructura'] != 'SIN PROGRAMA').astype(int)
    x['entidad'] = x['entidadejecutoranombre'].where(x['entidadejecutoranombre'].isin(S['entidades']), 'OTRAS ENTIDADES')

    # Señal 1: autoencoder
    Z = S['escalador'].transform(x[VARS_AE])
    x['error_ae'] = ((Z - S['autoencoder'].predict(Z)) ** 2).mean(axis=1)
    x['senal_atipica'] = x['error_ae'] > S['umbral_ae']
    # Señal 2: Random Forest
    pred = S['modelo_rf'].predict(x[NUM_RF + CAT_RF])
    x['monto_esperado'] = 10 ** pred
    x['veces_esperado'] = x['montototal'] / x['monto_esperado']
    x['senal_monto'] = (x['log_monto'] - pred).abs() > S['umbral_rf']
    # Señal 3: adelanto sobre el avance
    en_curso = x['etapaobra'] != 'FINALIZADAS'
    x['dinero_adelantado'] = ((x['diferencia_avance'].clip(lower=0) / 100) * x['montototal']).where(en_curso, 0)
    x['senal_adelanto'] = en_curso & (x['diferencia_avance'] > S['umbral_adelanto'])

    x['n_senales'] = x[['senal_atipica', 'senal_monto', 'senal_adelanto']].sum(axis=1).astype(int)
    msg = f'{len(x)} obras puntuadas'
    if n_usd: msg += f' · {n_usd} en dólares excluidas'
    if n_inval: msg += f' · {n_inval} sin datos válidos excluidas'
    return x, msg
