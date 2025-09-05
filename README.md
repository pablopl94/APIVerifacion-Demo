# KYC API - Demo

API y servicios para verificación de identidad (KYC) usando Flask y SQLAlchemy. Permite validar DNI, selfie y video, integrable con frontend web o móvil.

## Estructura principal

- `server.py`: Servidor Flask principal, configuración CORS, rutas y blueprints.
- `app/controllers/`: Controladores de endpoints (`kyc_controller.py`, `base_controller.py`, `upload_controller.py`).
- `app/services/`: Lógica de negocio para verificación de DNI, selfie y video.
- `app/models/`: Modelos de base de datos (usuarios, verificaciones).
- `app/database/connection.py`: Conexión y utilidades de base de datos.

## Endpoints principales

- `/kyc/validate-dni`: Validación de datos contra DNI.
- `/kyc/verify-selfie`: Verificación facial entre selfie y DNI.
- `/kyc/verify-video`: Verificación de video (liveness).
- `/`: Página HTML de prueba.

## Instalación rápida

1. Instala dependencias:
   ```
   pip install -r requirements.txt
   ```
2. Configura variables en `.env` (ejemplo: `OPENAI_API_KEY`).
3. Ejecuta el servidor:
   ```
   python server.py
   ```

## Notas

- CORS está habilitado para permitir peticiones desde web y apps móviles.
- Los resultados de verificación incluyen indicadores de fraude y confianza.
- El sistema usa DeepFace y OpenAI para análisis avanzado.
