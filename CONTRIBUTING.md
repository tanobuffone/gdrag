# Contribuir a gdrag

¡Gracias por tu interés en contribuir a gdrag! Este documento proporciona guías para contribuir al proyecto.

## Sistema de Ramas

- `main`: rama principal con código estable
- `develop`: rama de desarrollo activo
- `feat/*`: ramas para nuevas funcionalidades
- `fix/*`: ramas para corrección de bugs
- `docs/*`: ramas para documentación

## Proceso de Contribución

1. **Fork** el repositorio
2. Crea una rama desde `develop`:
   ```bash
   git checkout develop
   git checkout -b feat/tu-funcionalidad
   ```
3. Realiza tus cambios
4. Ejecuta los tests:
   ```bash
   pytest tests/ -v
   ```
5. Asegúrate de que el código sigue los estándares:
   ```bash
   ruff check src/
   mypy src/
   ```
6. Commit con mensaje descriptivo:
   ```bash
   git commit -m "feat: descripción de tu cambio"
   ```
7. Push y crea un Pull Request

## Estándares de Código

### Python
- Python 3.12+
- Type hints obligatorios
- Formato: ruff
- Linting: ruff + mypy
- Docstrings estilo Google

### Commits
Usar [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` nueva funcionalidad
- `fix:` corrección de bug
- `docs:` documentación
- `test:` tests
- `refactor:` refactorización
- `chore:` tareas de mantenimiento

### Testing
- Tests unitarios con pytest
- Cobertura mínima recomendada: 80%
- Tests en directorio `tests/` con misma estructura que `src/`

## Configuración de Desarrollo

```bash
# Clonar repositorio
git clone https://github.com/tanobuffone/gdrag.git
cd gdrag

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias de desarrollo
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy

# Copiar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Ejecutar tests
pytest tests/ -v

# Iniciar servidor de desarrollo
./start.sh
```

## Reportar Issues

Al reportar un issue, incluye:
1. Descripción clara del problema
2. Pasos para reproducir
3. Comportamiento esperado vs actual
4. Versión de Python y sistema operativo
5. Logs relevantes (sin credenciales)

## Áreas de Contribución

- **Core**: mejoras al pipeline RAG, embeddings, chunking
- **API**: nuevos endpoints, optimizaciones
- **MCP**: herramientas MCP adicionales
- **Testing**: aumentar cobertura
- **Documentación**: mejoras al README, ejemplos, guías
- **Integraciones**: conectores con otros sistemas

## Código de Conducta

- Ser respetuoso y constructivo
- Mantener conversaciones técnicas y profesionales
- Ayudar a nuevos contribuidores
- Reportar comportamiento inapropiado

## Licencia

Al contribuir, aceptas que tus cambios serán licenciados bajo la licencia MIT del proyecto.