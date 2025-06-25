# Lee el directorio desde la variable de entorno LAMBDA_LOGS_PATH
$path = [System.Environment]::GetEnvironmentVariable('LAMBDA_LOGS_PATH', 'User')

# Verifica si la ruta no está vacía
if (-not [String]::IsNullOrWhiteSpace($path)) {
    # Tiempo límite (archivos más antiguos de 3 días serán eliminados)
    $limit = (Get-Date).AddDays(-10)

    # Elimina archivos más antiguos de 10 días
    Get-ChildItem -Path $path -File -Recurse | Where-Object { $_.LastWriteTime -lt $limit } | Remove-Item -Force

    # Elimina directorios vacíos después de eliminar los archivos
    # Esto se hace en un segundo paso para evitar eliminar directorios que solo contienen archivos viejos
    Get-ChildItem -Path $path -Directory -Recurse | Where-Object { $_.GetFileSystemInfos().Count -eq 0 -and $_.LastWriteTime -lt $limit } | Remove-Item -Force -Recurse
} else {
    Write-Host "La variable de entorno LAMBDA_LOGS_PATH no está definida o está vacía."
}
