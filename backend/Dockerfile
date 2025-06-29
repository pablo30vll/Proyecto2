# Usar imagen base de Python
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para scripts y copiarlos
COPY scripts/ /app/scripts/

# Copiar código de la aplicación
COPY . .

# Crear directorio para modelos
RUN mkdir -p /app/models

# Añadir scripts al PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]