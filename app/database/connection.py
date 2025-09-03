"""
🗄️ CONFIGURACIÓN DE BASE DE DATOS MYSQL
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Configuración de MySQL
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'mysql+pymysql://root:password@localhost:3306/kyc_db?charset=utf8mb4'
)

print(f"🗄️ Conectando a MySQL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'configuración por defecto'}")

# Crear engine con configuración específica para MySQL
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Cambiar a True para ver SQL queries
    pool_pre_ping=True,  # Verificar conexión antes de usar
    pool_recycle=3600,   # Renovar conexiones cada hora
    connect_args={
        "charset": "utf8mb4",
        "use_unicode": True,
        "autocommit": False
    }
)

# Crear session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base para modelos
Base = declarative_base()

def get_db():
    """Obtener sesión de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Crear todas las tablas"""
    try:
        print("🏗️ Creando tablas en MySQL...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tablas creadas exitosamente")
    except Exception as e:
        print(f"❌ Error creando tablas: {e}")

def drop_tables():
    """Eliminar todas las tablas (solo para desarrollo)"""
    try:
        print("🗑️ Eliminando todas las tablas...")
        Base.metadata.drop_all(bind=engine)
        print("✅ Tablas eliminadas")
    except Exception as e:
        print(f"❌ Error eliminando tablas: {e}")

def test_connection():
    """Probar conexión a la base de datos"""
    try:
        with engine.connect() as connection:
            result = connection.execute("SELECT 1 as test")
            row = result.fetchone()
            if row and row[0] == 1:
                print("✅ Conexión a MySQL exitosa")
                return True
            else:
                print("❌ Error en test de conexión")
                return False
    except Exception as e:
        print(f"❌ Error conectando a MySQL: {e}")
        print("💡 Asegúrate de que:")
        print("   - MySQL esté ejecutándose")
        print("   - La base de datos 'kyc_db' exista")
        print("   - Las credenciales en .env sean correctas")
        return False
