#!/usr/bin/env python3
"""
🚀 INICIALIZADOR RÁPIDO DEL SISTEMA KYC
"""

import sys
import os

def main():
    print("🔧 INICIALIZANDO SISTEMA KYC")
    print("="*50)
    
    # 1. Verificar que existe .env
    if not os.path.exists('.env'):
        print("⚠️ Archivo .env no encontrado")
        print("💡 Copiando .env.example como .env...")
        try:
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Archivo .env creado")
            print("📝 EDITA el archivo .env y agrega tu OPENAI_API_KEY")
        except Exception as e:
            print(f"❌ Error creando .env: {e}")
            return False
    else:
        print("✅ Archivo .env encontrado")
    
    # 2. Verificar dependencias básicas
    print("\n📦 Verificando dependencias...")
    required_packages = ['flask', 'sqlalchemy', 'pymysql']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Ejecuta: pip install {package}")
    
    # 3. Crear carpeta uploads
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        print("✅ Carpeta uploads creada")
    else:
        print("✅ Carpeta uploads existe")
    
    print("\n🚀 LISTO PARA INICIAR")
    print("="*50)
    print("🔑 1. Edita el archivo .env con tu OPENAI_API_KEY")
    print("🗄️ 2. Configura MySQL y la base de datos")
    print("▶️  3. Ejecuta: python server.py")
    print("🌐 4. Ve a: http://localhost:5000")
    
    return True

if __name__ == '__main__':
    main()
