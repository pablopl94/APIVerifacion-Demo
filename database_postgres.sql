-- SCRIPT CREACIÓN BASE DE DATOS KYC (PostgreSQL)
-- Ejecutar en PostgreSQL como administrador

-- 1. Crear base de datos
CREATE DATABASE kyc_db;

-- 2. Crear usuario (opcional)
CREATE USER kyc_user WITH PASSWORD 'kyc_password_2025';

-- 3. Dar permisos
GRANT ALL PRIVILEGES ON DATABASE kyc_db TO kyc_user;

-- 4. Conectarse a la base de datos
\c kyc_db;

-- 5. Crear tablas

-- Tabla usuarios
CREATE TABLE IF NOT EXISTS users (
    document_number VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    nationality VARCHAR(10) NOT NULL,
    birth_date VARCHAR(20) NOT NULL,
    issue_date VARCHAR(20) NOT NULL,
    expiry_date VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(10) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'ACCEPTED', 'REJECTED', 'REVIEW'))
);

-- Tabla verificaciones DNI
CREATE TABLE IF NOT EXISTS dni_verifications (
    id VARCHAR(36) PRIMARY KEY,
    document_number VARCHAR(50) NOT NULL,
    form_first_name VARCHAR(100),
    form_last_name VARCHAR(100),
    form_document_number VARCHAR(50),
    form_nationality VARCHAR(10),
    form_birth_date VARCHAR(20),
    form_issue_date VARCHAR(20),
    form_expiry_date VARCHAR(20),
    extracted_first_name VARCHAR(100),
    extracted_last_name VARCHAR(100),
    extracted_document_number VARCHAR(50),
    extracted_nationality VARCHAR(10),
    extracted_birth_date VARCHAR(20),
    extracted_issue_date VARCHAR(20),
    extracted_expiry_date VARCHAR(20),
    extracted_full_text TEXT,
    match_name BOOLEAN DEFAULT FALSE,
    match_document_number BOOLEAN DEFAULT FALSE,
    match_birth_date BOOLEAN DEFAULT FALSE,
    match_issue_date BOOLEAN DEFAULT FALSE,
    match_expiry_date BOOLEAN DEFAULT FALSE,
    match_nationality BOOLEAN DEFAULT FALSE,
    document_type VARCHAR(50),
    is_valid_document BOOLEAN DEFAULT FALSE,
    document_country_match BOOLEAN DEFAULT FALSE,
    confidence INT DEFAULT 0,
    status VARCHAR(10) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'ACCEPTED', 'REJECTED')),
    details TEXT,
    dni_image_path VARCHAR(255),
    gpt_analysis JSON,
    gpt_raw_response JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_number) REFERENCES users(document_number) ON DELETE CASCADE
);

-- Tabla verificaciones selfie
CREATE TABLE IF NOT EXISTS selfie_verifications (
    id VARCHAR(36) PRIMARY KEY,
    document_number VARCHAR(50) NOT NULL,
    selfie_image_path VARCHAR(255) NOT NULL,
    dni_image_path VARCHAR(255),
    match_dni BOOLEAN DEFAULT FALSE,
    confidence INT DEFAULT 0,
    analysis_result TEXT,
    fraud_indicators JSON,
    status VARCHAR(10) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'ACCEPTED', 'REJECTED', 'REVIEW')),
    details TEXT,
    deepface_analysis JSON,
    technical_details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_number) REFERENCES users(document_number) ON DELETE CASCADE
);

-- Tabla verificaciones video (opcional)
CREATE TABLE IF NOT EXISTS video_verifications (
    id VARCHAR(36) PRIMARY KEY,
    document_number VARCHAR(50) NOT NULL,
    video_path VARCHAR(255) NOT NULL,
    reference_image_path VARCHAR(255),
    is_live_person BOOLEAN DEFAULT FALSE,
    matches_reference BOOLEAN DEFAULT FALSE,
    confidence INT DEFAULT 0,
    frames_analyzed INT DEFAULT 0,
    analysis_result TEXT,
    status VARCHAR(10) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'ACCEPTED', 'REJECTED', 'REVIEW')),
    details TEXT,
    deepface_analysis JSON,
    technical_details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_number) REFERENCES users(document_number) ON DELETE CASCADE
);

-- Índices para optimización
CREATE INDEX idx_users_document ON users(document_number);
CREATE INDEX idx_dni_document ON dni_verifications(document_number);
CREATE INDEX idx_dni_status ON dni_verifications(status);
CREATE INDEX idx_selfie_document ON selfie_verifications(document_number);
CREATE INDEX idx_selfie_status ON selfie_verifications(status);
CREATE INDEX idx_video_document ON video_verifications(document_number);
CREATE INDEX idx_video_status ON video_verifications(status);
