# ObraSafe Pro · contenedor

`app.py` es el código del notebook (3.6 a 3.8) tal cual; lee los resultados de los modelos desde `data/`.

## 1. Exportar los datos desde Colab
En la celda 3.8 (parte 2) del notebook, reemplazá la lista `cols` por esta (el resto del bloque no cambia):

```python
cols = ['idproyecto','nombreobra','nombreprovincia','nombredepto','sectornombre','entidadejecutoranombre','entidad',
        'etapaobra','montototal','duracionobrasdias','fechainicioanio','fechafinanio','url_perfil_obra',
        'avancefisico','avancefinanciero','error_ae','monto_esperado','veces_esperado','dinero_adelantado',
        'senal_atipica','senal_monto','senal_adelanto','n_senales','prioridad']
```
Ese bloque genera `data.zip`. Descomprimilo aquí para que queden `data/obras_puntuadas.csv`, `data/contactos.csv` y `data/config.json`.

## 2. Levantar
```bash
cp .env.example .env
docker compose up --build
```
Abrí http://localhost:7860

## 3. Alertas programadas (opcional)
```bash
docker compose run --rm alertas
# cron diario 8:00:  0 8 * * * cd /ruta/obrasafe-docker && docker compose run --rm alertas
```

## 4. Envío real de mails
Poné `DRY_RUN=false`, `EMAIL_USER` y `EMAIL_APP_PASSWORD` (contraseña de aplicación de Gmail) en `.env`.
Antes de publicarlo, definí `APP_USER` y `APP_PASSWORD`.
