"""Small deterministic serialization helpers without execution/model dependencies."""
from decimal import Decimal
from hashlib import sha256
import json

STRATEGIES=('A','B','C')
FEES=('0','0.0003','0.0005','0.001')
PRIMARY='0.0003'

def require(condition: bool,message: str) -> None:
    if not condition:raise ValueError(message)

def account_id(strategy: str,fee: str) -> str:
    require(strategy in STRATEGIES and fee in FEES,'Unregistered account')
    return strategy+'_fee_'+fee.replace('.','p')

def stable_bytes(value: object) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')

def digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()

def decimal(value: object) -> Decimal:
    require(not isinstance(value,bool),'Boolean monetary value')
    result=Decimal(str(value));require(result.is_finite(),'Nonfinite monetary value')
    return result
