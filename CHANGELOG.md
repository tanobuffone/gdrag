# Changelog

Todos los cambios notables en este proyecto se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [2.0.0] - 2026-03-24

### Added
- **Dual Embedding**: Soporte para embeddings locales (sentence-transformers) y OpenAI
- **Chunking Inteligente**: Estrategias sliding window, semántico, párrafo e híbrido
- **Re-ranking**: Cross-encoder para mejorar relevancia de resultados
- **Memoria de Sesión**: Persistencia de contexto entre queries
- **Compresión de Contexto**: Resúmenes automáticos para optimizar tokens
- **Atención Focalizada**: Decaimiento temporal y scoring de relevancia
- **API v2**: Endpoints mejorados con soporte de sesiones
- **MCP v2**: Herramientas MCP con gestión de sesiones
- **Testing**: Tests básicos para config, chunker y schemas
- **Deployment**: script start.sh, .env.example, documentación completa
- **Docker Compose**: Configuración para despliegue con PostgreSQL, Qdrant y Memgraph
- **Integración Cline**: Plan completo de integración con hooks, skills y workflows

### Changed
- Arquitectura completamente rediseñada para soporte multi-agente
- Pipeline de query reescrito con orquestación mejorada
- Configuración centralizada con Pydantic Settings

### Fixed
- Estructura del repositorio corregida

## [1.0.0] - 2026-03-20

### Added
- Implementación inicial del sistema RAG
- Integración básica con Qdrant para almacenamiento vectorial
- API REST básica
- Documentación inicial

---

## Convenciones

- **Added**: Nuevas funcionalidades
- **Changed**: Cambios en funcionalidades existentes
- **Deprecated**: Funcionalidades que serán eliminadas
- **Removed**: Funcionalidades eliminadas
- **Fixed**: Corrección de errores
- **Security**: Cambios relacionados con seguridad