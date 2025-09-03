-- 🗄️ SCRIPT CREACIÓN BASE DE DATOS KYC
-- Ejecutar en MySQL como administrador

-- 1. Crear base de datos
CREATE DATABASE IF NOT EXISTS kyc_db 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- 2. Crear usuario (opcional)
CREATE USER IF NOT EXISTS 'kyc_user'@'localhost' IDENTIFIED BY 'kyc_password_2025';

-- 3. Dar permisos
GRANT ALL PRIVILEGES ON kyc_db.* TO 'kyc_user'@'localhost';
FLUSH PRIVILEGES;

-- 4. Usar la base de datos
USE kyc_db;

-- 5. Crear tablas (el sistema las creará automáticamente, pero aquí está el esquema)

-- Tabla usuarios
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    document_number VARCHAR(50) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    nationality VARCHAR(10) NOT NULL,
    birth_date VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla verificaciones DNI
CREATE TABLE IF NOT EXISTS dni_verifications (
    id VARCHAR(36) PRIMARY KEY,
    document_number VARCHAR(50) NOT NULL,
    
    -- Datos del formulario
    form_first_name VARCHAR(100),
    form_last_name VARCHAR(100),
    form_document_number VARCHAR(50),
    form_nationality VARCHAR(10),
    form_birth_date VARCHAR(20),
    form_issue_date VARCHAR(20),
    form_expiry_date VARCHAR(20),
    
    -- Datos extraídos
    extracted_first_name VARCHAR(100),
    extracted_last_name VARCHAR(100),
    extracted_document_number VARCHAR(50),
    extracted_nationality VARCHAR(10),
    extracted_birth_date VARCHAR(20),
    extracted_issue_date VARCHAR(20),
    extracted_expiry_date VARCHAR(20),
    extracted_full_text TEXT,
    
    -- Comparaciones
    match_name BOOLEAN DEFAULT FALSE,
    match_document_number BOOLEAN DEFAULT FALSE,
    match_birth_date BOOLEAN DEFAULT FALSE,
    match_issue_date BOOLEAN DEFAULT FALSE,
    match_expiry_date BOOLEAN DEFAULT FALSE,
    match_nationality BOOLEAN DEFAULT FALSE,
    
    -- Análisis documento
    document_type VARCHAR(50),
    is_valid_document BOOLEAN DEFAULT FALSE,
    document_country_match BOOLEAN DEFAULT FALSE,
    
    -- Resultado
    confidence INT DEFAULT 0,
    status ENUM('PENDING', 'ACCEPTED', 'REJECTED', 'REVIEW') DEFAULT 'PENDING',
    details TEXT,
    dni_image_path VARCHAR(255),
    
    -- Raw data
    gpt_analysis JSON,
    gpt_raw_response JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (document_number) REFERENCES users(document_number) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla verificaciones selfie
CREATE TABLE IF NOT EXISTS selfie_verifications (
    id VARCHAR(36) PRIMARY KEY,
    document_number VARCHAR(50) NOT NULL,
    
    selfie_image_path VARCHAR(255) NOT NULL,
    dni_image_path VARCHAR(255),
    
    -- Resultados
    match_dni BOOLEAN DEFAULT FALSE,
    confidence INT DEFAULT 0,
    analysis_result TEXT,
    fraud_indicators JSON,
    
    status ENUM('PENDING', 'ACCEPTED', 'REJECTED', 'REVIEW') DEFAULT 'PENDING',
    details TEXT,
    
    -- Raw data
    deepface_analysis JSON,
    technical_details JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (document_number) REFERENCES users(document_number) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tabla verificaciones video (opcional)
CREATE TABLE IF NOT EXISTS video_verifications (
    id VARCHAR(36) PRIMARY KEY,
    document_number VARCHAR(50) NOT NULL,
    
    video_path VARCHAR(255) NOT NULL,
    reference_image_path VARCHAR(255),
    
    -- Resultados
    is_live_person BOOLEAN DEFAULT FALSE,
    matches_reference BOOLEAN DEFAULT FALSE,
    confidence INT DEFAULT 0,
    frames_analyzed INT DEFAULT 0,
    analysis_result TEXT,
    
    status ENUM('PENDING', 'ACCEPTED', 'REJECTED', 'REVIEW') DEFAULT 'PENDING',
    details TEXT,
    
    -- Raw data
    deepface_analysis JSON,
    technical_details JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (document_number) REFERENCES users(document_number) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Índices para optimización
CREATE INDEX idx_users_document ON users(document_number);
CREATE INDEX idx_dni_document ON dni_verifications(document_number);
CREATE INDEX idx_dni_status ON dni_verifications(status);
CREATE INDEX idx_selfie_document ON selfie_verifications(document_number);
CREATE INDEX idx_selfie_status ON selfie_verifications(status);
CREATE INDEX idx_video_document ON video_verifications(document_number);
CREATE INDEX idx_video_status ON video_verifications(status);

-- Mostrar tablas creadas
SHOW TABLES;

SELECT 'Base de datos KYC creada exitosamente!' as mensaje;
