import json
from web3 import Web3
from eth_account import Account

# Load the proof
with open('proof.json', 'r') as f:
    proof_data = json.load(f)

# Initialize Web3 - replace with your RPC endpoint
w3 = Web3(Web3.HTTPProvider('YOUR_RPC_ENDPOINT'))

# Load the contract ABI and address
# Replace with your deployed verifier contract address
VERIFIER_ADDRESS = 'YOUR_VERIFIER_CONTRACT_ADDRESS'

# The verifier contract ABI - this should match your deployed verifier
VERIFIER_ABI = [
    {
        "inputs": [
            {
                "internalType": "bytes",
                "name": "proof",
                "type": "bytes"
            },
            {
                "internalType": "uint256[]",
                "name": "publicInputs",
                "type": "uint256[]"
            }
        ],
        "name": "verify",
        "outputs": [
            {
                "internalType": "bool",
                "name": "",
                "type": "bool"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Initialize the contract
verifier_contract = w3.eth.contract(address=VERIFIER_ADDRESS, abi=VERIFIER_ABI)

# Prepare the proof and public inputs
proof = bytes(proof_data['proof'])
public_inputs = [int(x, 16) for x in proof_data['pretty_public_inputs']['outputs'][0]]

# Verify the proof
try:
    result = verifier_contract.functions.verify(proof, public_inputs).call()
    print(f"Verification result: {'Success' if result else 'Failed'}")
except Exception as e:
    print(f"Error during verification: {str(e)}") 