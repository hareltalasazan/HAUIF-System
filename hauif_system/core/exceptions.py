cat > hauif_system/core/exceptions.py << 'EOL'
class HAUIFError(Exception):
    def __init__(self, message: str, context: dict = None):
        self.context = context or {}
        super().__init__(message)

class ValidationError(HAUIFError):
    pass

class BlockchainError(HAUIFError):
    pass
EOL
