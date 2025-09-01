# 🛡️ Sistema KYC (Know Your Customer) - API Completa

Sistema completo de verificación de identidad con reconocimiento facial, OCR y análisis de video.

## 📋 Tabla de Contenidos

1. [Instalación](#instalación)
2. [Configuración](#configuración)
3. [Ejecución](#ejecución)
4. [APIs Disponibles](#apis-disponibles)
5. [Interfaces Web](#interfaces-web)
6. [Proceso de Verificación KYC](#proceso-de-verificación-kyc)
7. [Integración en tu Aplicación](#integración-en-tu-aplicación)
8. [Ejemplos de Uso](#ejemplos-de-uso)
9. [Solución de Problemas](#solución-de-problemas)

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.7 o superior
- Cámara web (para interfaces)
- Tesseract OCR

### Paso 1: Clonar/Descargar el proyecto
```bash
cd tu_directorio_de_trabajo
# Si tienes el proyecto, navega a la carpeta
```

### Paso 2: Activar entorno virtual (opcional pero recomendado)
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Paso 3: Instalar dependencias de Python
```bash
pip install cmake
pip install flask
pip install face_recognition
pip install opencv-python
pip install imutils
pip install pytesseract
pip install tensorflow
```

### Paso 4: Instalar Tesseract OCR

#### Windows:
- Opción 1: Descargar desde [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Opción 2: Con Chocolatey (ejecutar PowerShell como administrador):
  ```bash
  choco install tesseract
  ```

#### Linux:
```bash
sudo apt-get install tesseract-ocr
```

#### Mac:
```bash
brew install tesseract
```

---

## ⚙️ Configuración

El proyecto está pre-configurado y listo para usar. Las carpetas necesarias se crean automáticamente:

- `uploads/` - Para archivos temporales
- `templates/` - Interfaces web
- `images/` - Imágenes de ejemplo (opcional)

---

## 🏃‍♂️ Ejecución

### Iniciar el servidor:
```bash
python server.py
```

### Verificar que funciona:
1. Abre tu navegador
2. Ve a: `http://localhost:5000/`
3. Deberías ver: "Bismillah"

**¡El servidor estará corriendo en el puerto 5000!**

---

## 🔗 APIs Disponibles

### API Base
- **GET** `/` - Verificación de estado del servidor

### APIs de Procesamiento

#### 1. OCR - Extracción de Texto
```
GET /textImage?known=ruta_imagen
```
- **Parámetro**: `known` - Ruta completa a la imagen
- **Retorna**: Texto extraído de la imagen
- **Ejemplo**: `http://localhost:5000/textImage?known=C:\imagenes\documento.jpg`

#### 2. Reconocimiento Facial - Imagen
```
GET /faceImage?known=imagen1&unknown=imagen2
```
- **Parámetros**: 
  - `known` - Ruta a imagen de referencia
  - `unknown` - Ruta a imagen a verificar
- **Retorna**: `True` o `False`
- **Ejemplo**: `http://localhost:5000/faceImage?known=dni.jpg&unknown=selfie.jpg`

#### 3. Reconocimiento Facial - Video
```
GET /faceVideo?known=imagen&unknown=video
```
- **Parámetros**:
  - `known` - Ruta a imagen de referencia
  - `unknown` - Ruta al archivo de video
- **Retorna**: `True` o `False`
- **Ejemplo**: `http://localhost:5000/faceVideo?known=dni.jpg&unknown=video.mp4`

### APIs de Interfaces
- **GET** `/interface` - Interfaz de pruebas técnicas
- **GET** `/kyc` - **Proceso completo de verificación KYC**
- **POST** `/upload` - Subida de archivos

---

## 🌐 Interfaces Web

### 1. Interfaz de Pruebas Técnicas
**URL**: `http://localhost:5000/interface`

Interfaz para desarrolladores con 4 pestañas:
- 📄 **OCR**: Prueba extracción de texto
- 👤 **Reconocimiento Facial**: Compara dos imágenes
- 🎬 **Análisis de Video**: Busca cara en video
- 🧠 **Modelos ML**: Info sobre TensorFlow

### 2. **Proceso KYC Completo** ⭐
**URL**: `http://localhost:5000/kyc`

**¡Esta es la interfaz principal para usuarios finales!**

---

## 🔐 Proceso de Verificación KYC

### Flujo Completo (4 Pasos):

#### 📝 **Paso 1: Datos Personales + DNI**
- Formulario: nombre, DNI, fecha nacimiento, teléfono
- Subir foto del documento de identidad
- OCR automático para extraer datos del documento

#### 🤳 **Paso 2: Selfie con Cámara**
- Activación automática de cámara web
- Captura de selfie en tiempo real
- Preview y confirmación

#### 🎬 **Paso 3: Video de Verificación**
- Grabación de video de 5 segundos
- Audio opcional incluido
- Detección automática de rostro en movimiento

#### ✅ **Paso 4: Verificación y Resultado**
**Proceso automático de comparación:**

1. **OCR**: Extrae texto del DNI
2. **DNI vs Selfie**: Compara rostros usando `/faceImage`
3. **Selfie vs Video**: Verifica persona real usando `/faceVideo`
4. **Resultado Final**: ✅ APROBADO o ❌ RECHAZADO

### Criterios de Aprobación:
- ✅ Rostro del DNI coincide con selfie
- ✅ Rostro del selfie aparece en el video
- ✅ Datos del DNI son legibles
- ✅ **Resultado**: VERIFICACIÓN EXITOSA

---

## 🔧 Integración en tu Aplicación

### Opción 1: Iframe (Más Simple)
```html
<iframe src="http://localhost:5000/kyc" 
        width="100%" 
        height="800px" 
        frameborder="0">
</iframe>
```

### Opción 2: API Calls (Más Control)
```javascript
// 1. Subir archivos
const formData = new FormData();
formData.append('image', archivoImagen);
const uploadResponse = await fetch('/upload', {
    method: 'POST', 
    body: formData
});

// 2. Procesar OCR
const ocrResponse = await fetch(`/textImage?known=${rutaArchivo}`);
const textoExtraido = await ocrResponse.text();

// 3. Comparar rostros
const compareResponse = await fetch(`/faceImage?known=${dni}&unknown=${selfie}`);
const coincide = await compareResponse.text(); // "True" o "False"
```

### Opción 3: Como Servicio Independiente
```bash
# Ejecutar en servidor dedicado
python server.py

# Llamar desde otra aplicación
curl "http://tu-servidor:5000/faceImage?known=dni.jpg&unknown=selfie.jpg"
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Verificación Bancaria
```
1. Cliente completa formulario KYC
2. Sube foto del DNI
3. Toma selfie
4. Graba video corto
5. Sistema automáticamente:
   - Extrae datos del DNI
   - Verifica identidad
   - Genera reporte de verificación
```

### Ejemplo 2: Onboarding de App
```javascript
// En tu app móvil o web
window.location.href = "http://tu-api-kyc:5000/kyc";

// O embebido
<iframe src="http://tu-api-kyc:5000/kyc"></iframe>
```

### Ejemplo 3: API Personalizada
```python
import requests

# Verificar usuario
response = requests.get('http://localhost:5000/faceImage', {
    'known': 'path/to/id_photo.jpg',
    'unknown': 'path/to/selfie.jpg'
})

if response.text == 'True':
    print("Usuario verificado ✅")
else:
    print("Verificación fallida ❌")
```

---

## 🛠️ Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'face_recognition'"
```bash
# Solución:
deactivate  # Si estás en un entorno virtual
python server.py  # Ejecutar con Python global
```

### Error: "Camera not accessible"
- Verificar permisos de cámara en el navegador
- Usar HTTPS para producción
- Verificar que no hay otras apps usando la cámara

### Error: "Tesseract not found"
- Windows: Reinstalar Tesseract y verificar PATH
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

### La interfaz no carga
- Verificar que el servidor esté corriendo: `python server.py`
- Revisar la consola por errores
- Probar en modo incógnito

### Problemas de rendimiento
- Los videos grandes pueden tomar tiempo
- Recomendado: videos de máximo 5 segundos
- Imágenes recomendadas: máximo 2MB

---

## 📁 Estructura del Proyecto

```
kyc/
├── server.py                 # API principal
├── app.py                   # Modelo TensorFlow MNIST
├── classification.py        # Modelo Fashion-MNIST
├── ocr.py                   # Pruebas OCR
├── play.py                  # Player de video
├── templates/
│   ├── home.html           # Página simple
│   ├── kyc_interface.html  # Interfaz técnica
│   └── kyc_verification.html # ⭐ Proceso KYC completo
├── uploads/                # Archivos temporales
├── images/                 # Imágenes de ejemplo
├── venv/                   # Entorno virtual
└── README_COMPLETO.md      # Este archivo
```

---

## 🔄 Actualizaciones y Mantenimiento

### Para actualizar el sistema:
1. Hacer backup de tus archivos
2. Actualizar dependencias: `pip install --upgrade face_recognition opencv-python`
3. Reiniciar servidor: `python server.py`

### Para personalizar:
- Editar `templates/kyc_verification.html` para cambiar la interfaz
- Modificar `server.py` para añadir nuevas APIs
- Ajustar lógica de verificación en el JavaScript

---

## 📞 Contacto y Soporte

Si tienes problemas:
1. Revisar la sección "Solución de Problemas"
2. Verificar que todas las dependencias estén instaladas
3. Comprobar que Tesseract funciona: `tesseract --version`

---

## 🎯 Próximos Pasos

Una vez que tengas todo funcionando:

1. **Para Desarrollo**: Usa `/interface` para pruebas técnicas
2. **Para Usuarios**: Implementa `/kyc` en tu aplicación
3. **Para Producción**: Configura HTTPS y base de datos
4. **Para Escalar**: Considera usar Docker y load balancers

**¡Tu sistema KYC está listo para usar! 🚀**
