cat > hauif_system/blockchain/ledger.py << 'EOL'
import json
import aiofiles

class BlockchainLedger:
    def __init__(self, difficulty: int = 2, file_path: str = "enhanced_ledger.json"):
        self.difficulty = difficulty
        self.file_path = file_path
        self.chain = [{"index": 0, "data": {}, "hash": "genesis_block"}]  # Genesis block
    
    async def add_block(self, data: dict) -> dict:
        previous_block = self.chain[-1]
        new_block = {"index": len(self.chain), "data": data, "hash": "mock_hash"}  # Simplified for brevity
        self.chain.append(new_block)
        async with aiofiles.open(self.file_path, "w") as f:
            await f.write(json.dumps(self.chain, indent=2))
        return new_block
EOL
