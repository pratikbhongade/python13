"""
Configuration file for AspireVision Dashboard
Contains database connection settings and email configuration
"""

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DATABASE_CONFIG = {
    'PROD': {
        'SERVER': r'SDC01ASRSQPD01S\PSQLINST01',
        'DATABASE': 'ASPIRE',
        'DISPLAY_NAME': 'Production',
        'DRIVER': '{SQL Server}',
        'TRUSTED_CONNECTION': 'yes'
    },
    'IT': {
        'SERVER': r'SDC01ASRSQIT01S\PSQLINST01',  # Update with actual IT server
        'DATABASE': 'ASPIRE',
        'DISPLAY_NAME': 'IT Environment',
        'DRIVER': '{SQL Server}',
        'TRUSTED_CONNECTION': 'yes'
    },
    'QV': {
        'SERVER': r'SDC01ASRSQQV01S\PSQLINST01',  # Update with actual QV server
        'DATABASE': 'ASPIRE',
        'DISPLAY_NAME': 'QV Environment',
        'DRIVER': '{SQL Server}',
        'TRUSTED_CONNECTION': 'yes'
    }
}

# ============================================================================
# EMAIL CONFIGURATION
# ============================================================================

EMAIL_CONFIG = {
    # Email recipients
    'TO_ADDRESSES': [
        'Pratik_Bhongade@Keybank.com',
        'karen.a.tiemann-wozniak@key.com'
    ],
    
    # Email sender details (for signature)
    'SENDER': {
        'NAME': 'Pratik Bhongade',
        'TITLE': 'Software Engineer',
        'DEPARTMENT': 'KTO CBTO Equipment Finance',
        'LOCATION': 'Pune, India',
        'EMAIL': 'Pratik_Bhongade@key.com'
    },
    
    # Email subject template
    'SUBJECT_TEMPLATE': 'AspireVision Dashboard - {date}'
}

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_CONFIG = {
    # Default environment
    'DEFAULT_ENVIRONMENT': 'PROD',
    
    # Server settings
    'SERVER_PORT': 8050,
    'SERVER_HOST': '127.0.0.1',
    
    # Data retrieval settings
    'DAYS_HISTORICAL_DATA': 50,
    'MONTHS_JOB_DURATION': 6,
    
    # Logging settings
    'LOG_LEVEL': 'INFO',
    'LOG_DIR': 'logs',
    'LOG_FILE': 'dashboard.log'
}

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================

PATHS_CONFIG = {
    # Development paths
    'DEV': {
        'EDGE_DRIVER': r"C:\Aspire Dashboard\build_files\edgedriver_win64\msedgedriver.exe",
        'LOGO': r'C:\Aspire Dashboard\assets\Aspire.png',
        'TEMPLATE': r'C:\Aspire Dashboard\templates\layout.html'
    },
    
    # PyInstaller paths (relative)
    'PYINSTALLER': {
        'EDGE_DRIVER': r'edgedriver_win64\msedgedriver.exe',
        'LOGO': r'assets\Aspire.png',
        'TEMPLATE': r'templates\layout.html'
    }
}

# ============================================================================
# NOTES
# ============================================================================

"""
CONFIGURATION NOTES:

1. DATABASE_CONFIG:
   - Update IT and QV server names with actual values
   - If using SQL authentication instead of Windows auth, update TRUSTED_CONNECTION

2. EMAIL_CONFIG:
   - Update TO_ADDRESSES with actual recipient email addresses
   - Update SENDER details with your information

3. APP_CONFIG:
   - DEFAULT_ENVIRONMENT: Which database to connect to on startup
   - Adjust SERVER_PORT if 8050 is already in use

4. SECURITY:
   - This file contains sensitive information
   - DO NOT commit this file to public repositories
   - Add config.py to .gitignore
   - Consider using environment variables for production deployments
""" 
