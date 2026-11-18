"""
ObraSafe Pro · Asistente de revisión de obras públicas
Esta aplicación es el código del notebook (secciones 3.6 a 3.8) empaquetado para correr en un contenedor.
Los resultados de los modelos (señales y prioridad por obra) se leen de la carpeta data/, exportada desde Colab.
"""
import os, re, ssl, smtplib, json
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
import scoring

# ── Datos exportados desde Colab ───────────────────────────────────────────────────────
DATA = os.environ.get('DATA_DIR', 'data')
d = pd.read_csv(os.path.join(DATA, 'obras_puntuadas.csv'))
contactos = pd.read_csv(os.path.join(DATA, 'contactos.csv'))
_cfg = json.load(open(os.path.join(DATA, 'config.json')))

umbral_ae = _cfg['umbral_ae']
# El umbral del Random Forest es el percentil 95 del desvío en log10: se recupera de los datos si no se exportó
umbral_rf = _cfg.get('umbral_rf', float(np.percentile(np.abs(np.log10(d['montototal'] / d['monto_esperado'])), 95)))
UMBRAL_ADELANTO = _cfg.get('umbral_adelanto', 10)

SEN = ['senal_atipica', 'senal_monto', 'senal_adelanto']
SEN_NOMBRE = {'senal_atipica': 'obra atípica (autoencoder)',
              'senal_monto': 'monto inusual (Random Forest)',
              'senal_adelanto': 'adelanto sobre el avance'}

DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() != 'false'    # por defecto NO envía mails
REGISTRO = os.path.join(DATA, 'registro_avisos.csv')

# ── 3.6 · Aviso (notebook) ─────────────────────────────────────────────────────────────
def pesos(x):
    return '$' + f'{x:,.0f}'.replace(',', '.')

def armar_aviso(f):
    activas = ', '.join(SEN_NOMBRE[s] for s in SEN if f[s])
    lineas = ['AVISO DE REVISIÓN',
              f"Obra: {str(f['nombreobra'])[:90]} ({f['nombreprovincia']})",
              f"Entidad ejecutora: {f['entidadejecutoranombre']}",
              f"Riesgo {NIVELES[int(f['n_senales'])][0]}: {int(f['n_senales'])} de 3 señales ({activas})",
              f"Avance físico: {f['avancefisico']:.0f} % | Avance financiero: {f['avancefinanciero']:.0f} %"]
    if f['senal_adelanto']:
        lineas.append(f"Se abonó por encima de lo construido: {pesos(f['dinero_adelantado'])}")
    if f['senal_monto']:
        lineas.append(f"Monto declarado: {pesos(f['montototal'])}; el modelo esperaba cerca de {pesos(f['monto_esperado'])}")
    lineas.append('Se solicita revisar el estado de la obra. Este aviso es automático y no afirma que exista una pérdida.')
    return '\n'.join(lineas)


# ── 3.7 · Automatización: envío de avisos por mail (notebook) ─────────────────────────
def leer_secreto(nombre):
    try:
        from google.colab import userdata
        return userdata.get(nombre)
    except Exception:
        return os.environ.get(nombre)

def enviar_mail(destino, asunto, cuerpo):
    if DRY_RUN:
        print(f'[SIMULADO] Para: {destino} | Asunto: {asunto}')
        return 'simulado'
    try:
        msg = EmailMessage()
        msg['From'], msg['To'], msg['Subject'] = leer_secreto('EMAIL_USER'), destino, asunto
        msg.set_content(cuerpo)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ssl.create_default_context()) as servidor:
            servidor.login(leer_secreto('EMAIL_USER'), leer_secreto('EMAIL_APP_PASSWORD'))
            servidor.send_message(msg)
        return 'enviado'
    except Exception as e:
        return f'error: {e}'

def email_de(f):
    return contactos.set_index('entidad').loc[f['entidad'], 'email']

def registrar(f, destino, estado):
    fila = pd.DataFrame([{'fecha': datetime.now().strftime('%Y-%m-%d %H:%M'), 'idproyecto': f['idproyecto'],
                          'obra': str(f['nombreobra'])[:60], 'destinatario': destino,
                          'señales': int(f['n_senales']), 'estado': estado}])
    fila.to_csv(REGISTRO, mode='a', header=not os.path.exists(REGISTRO), index=False)
    return fila

def ejecutar_alertas(maximo=5, minimo_senales=2):
    ya_enviadas = set()
    if os.path.exists(REGISTRO):
        reg = pd.read_csv(REGISTRO)
        ya_enviadas = set(reg.loc[reg['estado'] == 'enviado', 'idproyecto'])
    candidatas = d[(d['n_senales'] >= minimo_senales) & ~d['idproyecto'].isin(ya_enviadas)]
    candidatas = candidatas.sort_values(['n_senales', 'dinero_adelantado'], ascending=False).head(maximo)
    resultados = []
    for _, f in candidatas.iterrows():
        destino = email_de(f)
        estado = enviar_mail(destino, f"Revisión de obra con riesgo {NIVELES[int(f['n_senales'])][0].lower()}", armar_aviso(f))
        resultados.append(registrar(f, destino, estado))
    return pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()


# ── 3.8 · Interfaz ObraSafe Pro — parte 1: lógica y tarjeta (no necesita gradio) ──────────
import html as _html, base64, json, mimetypes

# Nivel de riesgo según la cantidad de señales activas
NIVELES = {3: ('CRÍTICO',  '#d92d20'),
           2: ('ALTO',     '#f79009'),
           1: ('MODERADO', '#eaaa08'),
           0: ('BAJO',     '#12b76a')}
ACCIONES = {3: 'Acción inmediata', 2: 'Revisión prioritaria', 1: 'Seguimiento', 0: 'Sin acción'}
NIVEL_ORDEN = ['CRÍTICO', 'ALTO', 'MODERADO', 'BAJO']
NIVEL_COLOR = {n: c for n, c in NIVELES.values()}

d['nivel'] = d['n_senales'].astype(int).map(lambda n: NIVELES[n][0])
IDX = d.drop_duplicates('idproyecto').set_index('idproyecto', drop=False)   # búsqueda rápida por ID

LECTURAS = os.path.join(DATA, 'registro_lecturas.csv')
COL_AVISOS = ['fecha', 'idproyecto', 'obra', 'destinatario', 'señales', 'estado']
COL_LECT = ['fecha', 'idproyecto', 'obra', 'revisor']

def leer_csv(ruta, columnas):
    return pd.read_csv(ruta) if os.path.exists(ruta) else pd.DataFrame(columns=columnas)

# Lecturas confirmadas {id: (revisor, fecha)} y fotos adjuntas {id: data-uri}
LECT = {int(r.idproyecto): (r.revisor, r.fecha) for r in leer_csv(LECTURAS, COL_LECT).itertuples()}
FOTOS = {}

def esc(x):
    return _html.escape(str(x))

def millones(x):
    s = f'{x / 1e6:,.1f}'
    return '$' + s.replace(',', '§').replace('.', ',').replace('§', '.') + ' M'

def hallazgos(f):
    """Qué señales se activaron y el protocolo sugerido para cada una."""
    hs = []
    if f['senal_adelanto']:
        hs.append(('Adelanto sobre el avance',
                   f"Avance financiero {f['avancefinanciero']:.0f} % vs. físico {f['avancefisico']:.0f} %: "
                   f"{pesos(f['dinero_adelantado'])} abonados por encima de lo construido.",
                   'Verificar certificados y anticipos contractuales; contrastar el avance en terreno.'))
    if f['senal_monto']:
        hs.append(('Monto inusual',
                   f"Declarado {pesos(f['montototal'])} vs. esperado {pesos(f['monto_esperado'])} "
                   f"({f['veces_esperado']:.1f} veces).",
                   'Solicitar cómputo y presupuesto; comparar con obras similares del sector.'))
    if f['senal_atipica']:
        hs.append(('Obra atípica',
                   f"Combinación inusual de monto, plazo y avances (error {f['error_ae']:.2f}, umbral {umbral_ae:.2f}).",
                   'Revisar la carga de datos y la consistencia entre monto, plazo y avances.'))
    return hs

CSS = """
.osp-wrap{max-width:780px;margin:0 auto;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:#101828}
.osp-card{background:#fff;border-radius:16px;box-shadow:0 10px 30px rgba(16,24,40,.14);border-top:8px solid var(--c);padding:22px 26px 22px}
.osp-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.osp-brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:19px;color:#101828}
.osp-badge{display:inline-flex;align-items:center;gap:8px;border:1px solid #e4e7ec;border-radius:999px;padding:6px 14px;font-size:12px;font-weight:700;letter-spacing:.05em;color:#344054;background:#fff}
.osp-dot{width:10px;height:10px;border-radius:50%;background:var(--c)}
.osp-title{font-size:26px;font-weight:700;line-height:1.2;margin:4px 0 6px;color:#101828}
.osp-sub{font-size:14px;color:#475467;padding-bottom:14px;border-bottom:1px solid #e4e7ec;margin-bottom:14px}
.osp-grid{display:grid;grid-template-columns:repeat(2,1fr);border:1px solid #e4e7ec;border-radius:10px;overflow:hidden}
.osp-cell{padding:10px 14px;border-bottom:1px solid #e4e7ec;border-right:1px solid #e4e7ec;background:#fff}
.osp-cell:nth-child(2n){border-right:none}
.osp-cell:nth-last-child(-n+2){border-bottom:none}
.osp-k{font-size:12.5px;color:#667085}
.osp-v{font-size:14.5px;font-weight:700;color:#101828}
.osp-row{display:grid;grid-template-columns:1.25fr 1fr;gap:14px;margin-top:14px}
@media(max-width:640px){.osp-row{grid-template-columns:1fr}}
.osp-box{border:1px solid #e4e7ec;border-radius:10px;padding:14px;background:#fff}
.osp-box h4{margin:0 0 6px;font-size:14px;font-weight:700;color:#101828}
.osp-item{margin:0 0 10px;font-size:13.5px;color:#344054;line-height:1.35}
.osp-item b{color:#101828}
.osp-ok{color:#067647;font-weight:600;font-size:14px}
.osp-ev{position:relative;border-radius:10px;overflow:hidden;min-height:170px;background:linear-gradient(135deg,#1d2939,#475467);display:flex;flex-direction:column;justify-content:center;padding:14px;color:#fff}
.osp-ev img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.osp-ev-label{position:absolute;left:8px;right:8px;bottom:8px;background:rgba(16,24,40,.72);color:#fff;font-size:12.5px;padding:6px 10px;border-radius:6px;text-align:center}
.osp-bar{height:9px;background:rgba(255,255,255,.25);border-radius:6px;overflow:hidden;margin:3px 0 10px}
.osp-bar span{display:block;height:100%;border-radius:6px}
.osp-foot{display:grid;grid-template-columns:1.5fr 1fr;gap:12px;margin-top:16px}
.osp-link{display:block;text-align:center;background:#0b1b33;color:#fff !important;text-decoration:none;border-radius:8px;padding:11px;font-weight:600;font-size:14px}
.osp-pill{display:block;text-align:center;border:1px solid #d0d5dd;border-radius:8px;padding:10px;font-weight:600;font-size:13.5px;background:#fff;color:#344054}
.osp-pill.ok{background:#ecfdf3;border-color:#abefc6;color:#067647}
.osp-meta{margin-top:10px;font-size:12px;color:#98a2b3;text-align:right}
.osp-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:6px 0 4px}
.osp-kpi{background:#fff;border:1px solid #e4e7ec;border-radius:12px;padding:12px 14px;border-left:5px solid var(--c)}
.osp-kpi .n{font-size:24px;font-weight:700;color:#101828}
.osp-kpi .t{font-size:12px;color:#667085}
.osp-top{display:flex;align-items:center;gap:12px;background:#0b1b33;color:#fff;border-radius:14px;padding:14px 20px;margin-bottom:6px}
.osp-top b{font-size:20px} .osp-top span{color:#98a2b3;font-size:13px}
.osp-btn-dark,.osp-btn-dark button{background:#0b1b33 !important;color:#fff !important;border-radius:8px !important;font-weight:600 !important}
.osp-btn-light,.osp-btn-light button{background:#fff !important;color:#344054 !important;border:1px solid #d0d5dd !important;border-radius:8px !important;font-weight:600 !important}
"""

LOGO = ('<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#101828" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/></svg>')

def cabecera_html():
    return f'<div class="osp-top">{LOGO.replace("#101828", "#ffffff")}<div><b>ObraSafe Pro</b><br><span>Asistente de revisión de obras públicas · Panel de control</span></div></div>'

def kpis_html(x):
    avisos = leer_csv(REGISTRO, COL_AVISOS)
    env = int(avisos['estado'].isin(['enviado', 'simulado']).sum())
    datos = [(len(x), 'Obras en el filtro', '#344054'),
             (int((x['nivel'] == 'CRÍTICO').sum()), 'Riesgo crítico', NIVEL_COLOR['CRÍTICO']),
             (int((x['nivel'] == 'ALTO').sum()), 'Riesgo alto', NIVEL_COLOR['ALTO']),
             (millones(x['dinero_adelantado'].sum()), 'Dinero adelantado', '#7a5af8'),
             (env, 'Avisos emitidos', '#2e90fa'),
             (len(LECT), 'Lecturas confirmadas', '#12b76a')]
    celdas = ''.join(f'<div class="osp-kpi" style="--c:{c}"><div class="n">{n}</div><div class="t">{t}</div></div>' for n, t, c in datos)
    return f'<div class="osp-wrap" style="max-width:none"><div class="osp-kpis">{celdas}</div></div>'

def tarjeta_html(f):
    n = int(f['n_senales'])
    nivel, color = NIVELES[n]
    id_ = int(f['idproyecto'])
    hs = hallazgos(f)

    if hs:
        hall = ''.join(f'<p class="osp-item"><b>{esc(t)}</b><br>{esc(det)}</p>' for t, det, _ in hs)
        prot = ''.join(f'<p class="osp-item">• {esc(p)}</p>' for _, _, p in hs)
    else:
        hall = '<p class="osp-ok">✔ Sin señales de alerta</p>'
        prot = '<p class="osp-item">Mantener el monitoreo habitual.</p>'

    if id_ in FOTOS:
        evid = f'<div class="osp-ev"><img src="{FOTOS[id_]}"><div class="osp-ev-label">Evidencia fotográfica #1</div></div>'
    else:
        fis, fin = float(f['avancefisico']), float(f['avancefinanciero'])
        col_fin = '#f04438' if f['senal_adelanto'] else '#53b1fd'
        evid = (f'<div class="osp-ev"><div style="font-size:12.5px">Avance físico · <b>{fis:.0f} %</b></div>'
                f'<div class="osp-bar"><span style="width:{min(fis,100):.0f}%;background:#32d583"></span></div>'
                f'<div style="font-size:12.5px">Avance financiero · <b>{fin:.0f} %</b></div>'
                f'<div class="osp-bar"><span style="width:{min(fin,100):.0f}%;background:{col_fin}"></span></div>'
                f'<div class="osp-ev-label">Sin foto adjunta · indicadores de avance</div></div>')

    if id_ in LECT:
        lect = f'<span class="osp-pill ok">✔ Leído por {esc(LECT[id_][0])}</span>'
    else:
        lect = '<span class="osp-pill">Lectura pendiente</span>'

    celdas = [('Categoría', f['sectornombre']),
              ('Reportado por', f['entidadejecutoranombre']),
              ('Ubicación', f"{f['nombreprovincia']} · {f['nombredepto']}"),
              ('Estado', f"{str(f['etapaobra']).capitalize()} · {ACCIONES[n]}"),
              ('Monto total', f"{pesos(f['montototal'])}"),
              ('Plazo', f"{int(f['duracionobrasdias'])} días ({int(f['fechainicioanio'])}–{int(f['fechafinanio'])})")]
    grid = ''.join(f'<div class="osp-cell"><div class="osp-k">{k}:</div><div class="osp-v">{esc(v)}</div></div>' for k, v in celdas)

    return f'''<div class="osp-wrap"><div class="osp-card" style="--c:{color}">
  <div class="osp-head"><div class="osp-brand">{LOGO} ObraSafe Pro</div>
    <div class="osp-badge"><span class="osp-dot"></span>RIESGO {nivel}</div></div>
  <div class="osp-title">{esc(str(f['nombreobra']).title()[:110])}</div>
  <div class="osp-sub">{esc(f['entidadejecutoranombre'])} (Cód: {id_}) • Analizado el {datetime.now().strftime('%d/%m/%Y %H:%M')} hs</div>
  <div class="osp-grid">{grid}</div>
  <div class="osp-row">
    <div class="osp-box"><h4>Hallazgos detectados ({n} de 3 señales)</h4>{hall}<h4 style="margin-top:12px">Protocolo / Acción requerida</h4>{prot}</div>
    {evid}
  </div>
  <div class="osp-foot"><a class="osp-link" href="{esc(f['url_perfil_obra'])}" target="_blank">Abrir reporte en plataforma ↗</a>{lect}</div>
  <div class="osp-meta">Esta alerta es automática: indica que conviene revisar la obra, no que exista una irregularidad.</div>
</div></div>'''

VACIO = '<div class="osp-wrap"><div class="osp-card" style="--c:#98a2b3"><b>No hay obras para el filtro elegido.</b><br>Probá ampliar los niveles de riesgo o limpiar la búsqueda.</div></div>'

# ── filtros, tablas y gráficos ──
def filtrar(niveles, provs, sects, ents, texto):
    x = d
    if niveles: x = x[x['nivel'].isin(niveles)]
    if provs:   x = x[x['nombreprovincia'].isin(provs)]
    if sects:   x = x[x['sectornombre'].isin(sects)]
    if ents:    x = x[x['entidad'].isin(ents)]
    if texto and texto.strip():
        t = texto.strip()
        x = x[x['nombreobra'].str.contains(t, case=False, na=False, regex=False) | (x['idproyecto'].astype(str) == t)]
    return x.sort_values(['n_senales', 'dinero_adelantado'], ascending=False)

def opciones(x):
    return [(f"[{f['nivel']}] {str(f['nombreobra'])[:55]} · {f['nombreprovincia']} · id {f['idproyecto']}", int(f['idproyecto']))
            for _, f in x.head(300).iterrows()]

def tabla_ranking(x):
    x = x.head(300)
    return pd.DataFrame({'ID': x['idproyecto'], 'Obra': x['nombreobra'].str.title().str[:70], 'Provincia': x['nombreprovincia'],
                         'Sector': x['sectornombre'], 'Riesgo': x['nivel'], 'Señales': x['n_senales'],
                         'Av. físico %': x['avancefisico'].round(0), 'Av. financ. %': x['avancefinanciero'].round(0),
                         'Adelantado ($M)': (x['dinero_adelantado'] / 1e6).round(1), 'Monto ($M)': (x['montototal'] / 1e6).round(1),
                         'Veces esperado': x['veces_esperado'].round(1)}).reset_index(drop=True)

def graficos(x):
    f1, a1 = plt.subplots(figsize=(4.8, 3.6)); f2, a2 = plt.subplots(figsize=(5.6, 3.6)); f3, a3 = plt.subplots(figsize=(4.8, 3.6))
    if len(x) == 0:
        for a in (a1, a2, a3): a.text(.5, .5, 'Sin obras', ha='center'); a.axis('off')
        return f1, f2, f3
    s = x[SEN].sum().rename(SEN_NOMBRE).sort_values()
    a1.barh([i.split(' (')[0] for i in s.index], s.values, color='#2e90fa'); a1.set_title('Señales activas')
    top = x['nombreprovincia'].value_counts().head(8).index
    t = pd.crosstab(x[x['nombreprovincia'].isin(top)]['nombreprovincia'], x['nivel']).reindex(columns=NIVEL_ORDEN, fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    t.plot(kind='barh', stacked=True, ax=a2, color=[NIVEL_COLOR[c] for c in t.columns]); a2.set_title('Obras por provincia y riesgo'); a2.set_ylabel(''); a2.legend(fontsize=7)
    m = x.sample(min(len(x), 2000), random_state=1)
    a3.scatter(m['avancefisico'], m['avancefinanciero'], s=7, alpha=.5, c=[NIVEL_COLOR[v] for v in m['nivel']])
    a3.plot([0, 100], [0, 100], 'k--', lw=1); a3.set_xlabel('Avance físico %'); a3.set_ylabel('Avance financiero %'); a3.set_title('Físico vs. financiero')
    for f in (f1, f2, f3): f.tight_layout()
    return f1, f2, f3

def historial():
    return leer_csv(REGISTRO, COL_AVISOS).iloc[::-1].reset_index(drop=True), leer_csv(LECTURAS, COL_LECT).iloc[::-1].reset_index(drop=True)

# ── 3.8 · Interfaz ObraSafe Pro — parte 2: Gradio ───────────────────────────────────────
import gradio as gr

NIVELES_INI = ['CRÍTICO', 'ALTO']                      # filtro inicial: lo que más conviene revisar
x0 = filtrar(NIVELES_INI, [], [], [], '')

# ---------- funciones de la interfaz ----------
def ver_obra(id_obra):
    if id_obra is None:
        return VACIO, '', '', '', None
    f = IDX.loc[int(id_obra)]
    return tarjeta_html(f), armar_aviso(f), email_de(f), '', None

def aplicar_filtros(niveles, provs, sects, ents, texto):
    x = filtrar(niveles, provs, sects, ents, texto)
    ops = opciones(x)
    g1, g2, g3 = graficos(x)
    return (gr.update(choices=ops, value=ops[0][1] if ops else None), tabla_ranking(x), kpis_html(x), g1, g2, g3)

def elegir_de_tabla(evt: gr.SelectData, niveles, provs, sects, ents, texto):
    fila = getattr(evt, 'row_value', None)
    id_ = int(fila[0]) if fila else int(filtrar(niveles, provs, sects, ents, texto).head(300).iloc[evt.index[0]]['idproyecto'])
    return gr.update(value=id_), gr.update(selected='panel')

def adjuntar_foto(ruta, id_obra):
    if ruta and id_obra is not None:
        mime = mimetypes.guess_type(ruta)[0] or 'image/jpeg'
        FOTOS[int(id_obra)] = f'data:{mime};base64,' + base64.b64encode(open(ruta, 'rb').read()).decode()
    return tarjeta_html(IDX.loc[int(id_obra)]) if id_obra is not None else VACIO

def confirmar_lectura(id_obra, revisor):
    if id_obra is None:
        return VACIO, 'Elegí una obra primero.', *historial()
    if not revisor or not revisor.strip():
        return tarjeta_html(IDX.loc[int(id_obra)]), '⚠️ Escribí tu nombre en "Revisor" para confirmar la lectura.', *historial()
    f = IDX.loc[int(id_obra)]
    ahora = datetime.now().strftime('%Y-%m-%d %H:%M')
    pd.DataFrame([{'fecha': ahora, 'idproyecto': int(id_obra), 'obra': str(f['nombreobra'])[:60], 'revisor': revisor.strip()}]).to_csv(
        LECTURAS, mode='a', header=not os.path.exists(LECTURAS), index=False)
    LECT[int(id_obra)] = (revisor.strip(), ahora)
    return tarjeta_html(f), f'✔ Lectura confirmada por {revisor.strip()} ({ahora})', *historial()

def enviar_aviso(id_obra, destino, texto):
    if id_obra is None:
        return 'Elegí una obra primero.', *historial()
    f = IDX.loc[int(id_obra)]
    estado = enviar_mail(destino, f"Revisión de obra con riesgo {f['nivel'].lower()}", texto)
    registrar(f, destino, estado)
    return f'Estado: {estado} → {destino}', *historial()

def correr_alertas(simulado, maximo, minimo):
    global DRY_RUN
    DRY_RUN = bool(simulado)
    res = ejecutar_alertas(maximo=int(maximo), minimo_senales=int(minimo))
    if len(res) == 0:
        res = pd.DataFrame({'info': ['No hay obras pendientes de aviso con esos criterios']})
    return res, *historial()

def exportar(niveles, provs, sects, ents, texto):
    ruta = 'obras_filtradas.csv'
    tabla_ranking(filtrar(niveles, provs, sects, ents, texto)).to_csv(ruta, index=False)
    return ruta

# ---------- cargar un CSV nuevo y puntuarlo con los modelos ya entrenados ----------
COLS_EXPORT = ['idproyecto','nombreobra','nombreprovincia','nombredepto','sectornombre','entidadejecutoranombre','entidad',
               'etapaobra','montototal','duracionobrasdias','fechainicioanio','fechafinanio','url_perfil_obra',
               'avancefisico','avancefinanciero','error_ae','monto_esperado','veces_esperado','dinero_adelantado',
               'senal_atipica','senal_monto','senal_adelanto','n_senales','prioridad']

def subir_csv(ruta):
    global d, IDX
    if not ruta:
        return 'Elegí un archivo CSV primero.'
    try:
        nuevas, info = scoring.puntuar(pd.read_csv(ruta))
    except FileNotFoundError:
        return f'⚠️ No se encontró el modelo entrenado ({scoring.RUTA_MODELO}). Copialo a la carpeta modelos/ y reconstruí la imagen.'
    except ValueError as e:
        return f'⚠️ {e}'
    nuevas['nivel'] = nuevas['n_senales'].astype(int).map(lambda n: NIVELES[n][0])
    nuevas['prioridad'] = nuevas['nivel']
    actual = os.path.join(DATA, 'obras_puntuadas.csv')
    copia = os.path.join(DATA, 'obras_puntuadas_original.csv')
    if os.path.exists(actual) and not os.path.exists(copia):
        os.replace(actual, copia)                      # se conserva la exportación original de Colab
    nuevas[COLS_EXPORT].to_csv(actual, index=False)    # queda guardado en el volumen data/
    d = nuevas
    IDX = d.drop_duplicates('idproyecto').set_index('idproyecto', drop=False)
    r = nuevas['nivel'].value_counts().reindex(NIVEL_ORDEN).fillna(0).astype(int)
    return f'✔ {info}. ' + ' · '.join(f'{k}: {v}' for k, v in r.items())

def refrescar_opciones():
    return (gr.update(choices=sorted(d['nombreprovincia'].dropna().unique()), value=[]),
            gr.update(choices=sorted(d['sectornombre'].dropna().unique()), value=[]),
            gr.update(choices=sorted(d['entidad'].unique()), value=[]))

# ---------- armado de la pantalla ----------
with gr.Blocks(title='ObraSafe Pro · Asistente de revisión de obras', css=CSS,
               theme=gr.themes.Soft(primary_hue='slate', neutral_hue='slate')) as demo:

    gr.HTML(cabecera_html())
    kpis = gr.HTML(kpis_html(x0))

    with gr.Accordion('🔎 Filtros', open=True):
        with gr.Row():
            f_niv = gr.CheckboxGroup(NIVEL_ORDEN, value=NIVELES_INI, label='Nivel de riesgo')
            f_txt = gr.Textbox(label='Buscar por nombre o ID', placeholder='Ej.: escuela, puente, 1610')
        with gr.Row():
            f_prov = gr.Dropdown(sorted(d['nombreprovincia'].dropna().unique()), multiselect=True, label='Provincia')
            f_sec = gr.Dropdown(sorted(d['sectornombre'].dropna().unique()), multiselect=True, label='Sector')
            f_ent = gr.Dropdown(sorted(d['entidad'].unique()), multiselect=True, label='Entidad ejecutora')
    filtros = [f_niv, f_prov, f_sec, f_ent, f_txt]

    with gr.Tabs() as tabs:
        # ---- Panel de la obra (estilo tarjeta) ----
        with gr.Tab('📋 Panel de obra', id='panel'):
            ops0 = opciones(x0)
            obra = gr.Dropdown(ops0, value=ops0[0][1] if ops0 else None, label='Obra', filterable=True)
            tarjeta = gr.HTML()
            with gr.Row():
                revisor = gr.Textbox(label='Revisor', placeholder='Tu nombre', scale=2)
                b_leer = gr.Button('Confirmar lectura', elem_classes=['osp-btn-light'], scale=1)
                b_aviso = gr.Button('Enviar aviso al responsable', elem_classes=['osp-btn-dark'], scale=1)
            msg = gr.Markdown()
            with gr.Accordion('✉️ Aviso para el responsable', open=False):
                destino = gr.Textbox(label='Destinatario', info='Tomado de la tabla de contactos; se puede editar.')
                aviso = gr.Textbox(label='Mensaje', lines=8)
                estado_envio = gr.Textbox(label='Estado del envío', interactive=False)
            with gr.Accordion('📷 Adjuntar evidencia fotográfica', open=False):
                foto = gr.Image(type='filepath', label='Foto de la obra (se muestra en la tarjeta)')

        # ---- Ranking ----
        with gr.Tab('🏆 Ranking de obras'):
            gr.Markdown('Las 300 obras con mayor prioridad según los filtros. **Hacé clic en una fila** para abrir su panel.')
            ranking = gr.Dataframe(value=tabla_ranking(x0), interactive=False, wrap=True)
            with gr.Row():
                b_export = gr.Button('Exportar a CSV', elem_classes=['osp-btn-light'], scale=1)
                archivo = gr.File(label='Archivo exportado', scale=3)

        # ---- Análisis ----
        with gr.Tab('📊 Análisis'):
            with gr.Row():
                g1 = gr.Plot(label='Señales'); g2 = gr.Plot(label='Provincias'); g3 = gr.Plot(label='Avances')

        # ---- Historial ----
        with gr.Tab('🗂️ Historial'):
            gr.Markdown('**Avisos emitidos**')
            h_avisos = gr.Dataframe(value=historial()[0], interactive=False)
            gr.Markdown('**Lecturas confirmadas**')
            h_lect = gr.Dataframe(value=historial()[1], interactive=False)

        # ---- Subir obras nuevas ----
        with gr.Tab('📤 Subir obras nuevas'):
            gr.Markdown('Subí un CSV con el **mismo formato del dataset MOP**. Los modelos ya entrenados (red neuronal y Random Forest) '
                        'calculan las señales y el nivel de riesgo de cada obra, y la interfaz pasa a trabajar con esas obras.\n\n'
                        'Columnas obligatorias: ' + ', '.join(f'`{c}`' for c in scoring.REQUERIDAS))
            csv_nuevo = gr.File(label='CSV de obras', file_types=['.csv'])
            b_subir = gr.Button('Puntuar obras', elem_classes=['osp-btn-dark'])
            estado_subida = gr.Markdown()

        # ---- Automatización ----
        with gr.Tab('⚙️ Automatización'):
            gr.Markdown('Ejecuta `ejecutar_alertas`: avisa a los responsables de las obras pendientes, sin repetir las ya enviadas.')
            with gr.Row():
                c_sim = gr.Checkbox(value=DRY_RUN, label='Modo simulado (no envía mails reales)')
                c_max = gr.Slider(1, 50, value=5, step=1, label='Máximo de avisos por ejecución')
                c_min = gr.Slider(1, 3, value=2, step=1, label='Mínimo de señales para avisar')
            b_auto = gr.Button('▶ Ejecutar alertas automáticas', elem_classes=['osp-btn-dark'])
            res_auto = gr.Dataframe(interactive=False, label='Resultado de la ejecución')
            gr.Markdown(f"**Umbrales del sistema** · Autoencoder: error > {umbral_ae:.3f} (percentil 95) · "
                        f"Random Forest: monto > {10**umbral_rf:.1f} veces el esperado · Adelanto: más de {UMBRAL_ADELANTO} puntos de diferencia")

    # ---------- eventos ----------
    salidas_filtro = [obra, ranking, kpis, g1, g2, g3]
    for c in filtros:
        c.change(aplicar_filtros, filtros, salidas_filtro)
    obra.change(ver_obra, obra, [tarjeta, aviso, destino, estado_envio, foto])
    ranking.select(elegir_de_tabla, filtros, [obra, tabs])
    foto.upload(adjuntar_foto, [foto, obra], tarjeta)
    b_leer.click(confirmar_lectura, [obra, revisor], [tarjeta, msg, h_avisos, h_lect])
    b_aviso.click(enviar_aviso, [obra, destino, aviso], [estado_envio, h_avisos, h_lect])
    b_export.click(exportar, filtros, archivo)
    b_auto.click(correr_alertas, [c_sim, c_max, c_min], [res_auto, h_avisos, h_lect])
    b_subir.click(subir_csv, csv_nuevo, estado_subida).then(
        refrescar_opciones, None, [f_prov, f_sec, f_ent]).then(aplicar_filtros, filtros, salidas_filtro)
    demo.load(aplicar_filtros, filtros, salidas_filtro)
    demo.load(ver_obra, obra, [tarjeta, aviso, destino, estado_envio, foto])


if __name__ == '__main__':
    usuario, clave = os.environ.get('APP_USER'), os.environ.get('APP_PASSWORD')
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)), auth=(usuario, clave) if usuario and clave else None)
