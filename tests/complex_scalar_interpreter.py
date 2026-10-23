"""Evaluate emitted scalar IR with NumPy IEEE arithmetic for CPU oracle tests."""
import numpy as np
from xdsl.dialects.builtin import Float32Type,IntegerType
from xdsl.dialects import arith

def evaluate(ops,args,values,results):
 env=dict(zip(args,values))
 unary={'math.absf':np.abs,'math.sqrt':np.sqrt,'math.exp':np.exp,'math.expm1':np.expm1,'math.log':np.log,'math.log1p':np.log1p,'math.sin':np.sin,'math.cos':np.cos,'arith.negf':np.negative}
 binary={'arith.addf':np.add,'arith.subf':np.subtract,'arith.mulf':np.multiply,'arith.divf':np.divide,'math.atan2':np.arctan2,'math.copysign':np.copysign,'arith.andi':np.logical_and,'arith.ori':np.logical_or}
 with np.errstate(all='ignore'):
  for op in ops:
   xs=[env[a] for a in op.operands];name=op.name
   if name=='arith.constant':v=op.value.value.data
   elif name in unary:v=unary[name](*xs)
   elif name in ('arith.andi','arith.ori') and isinstance(op.operands[0].type,IntegerType) and op.operands[0].type.bitwidth > 1:
    v=(np.bitwise_and if name=='arith.andi' else np.bitwise_or)(*xs)
   elif name in binary:v=binary[name](*xs)
   elif name=='arith.bitcast':
    dtype=np.int32 if isinstance(op.results[0].type,IntegerType) and op.results[0].type.bitwidth==32 else np.int64 if isinstance(op.results[0].type,IntegerType) else np.float32 if isinstance(op.results[0].type,Float32Type) else np.float64
    v=np.asarray(xs[0]).view(dtype)
   elif name in ('arith.extf','arith.truncf'):v=xs[0]
   elif name=='arith.select':v=np.where(*xs)
   elif name=='arith.cmpf':
    pred=arith.CMPF_COMPARISON_OPERATIONS[op.predicate.value.data];a,b=xs;nan=np.isnan(a)|np.isnan(b)
    if pred=='uno':v=nan
    elif pred=='ord':v=~nan
    else:
     test={'eq':np.equal,'ne':np.not_equal,'lt':np.less,'le':np.less_equal,'gt':np.greater,'ge':np.greater_equal}[pred[1:]](a,b)
     v=(test|nan) if pred[0]=='u' else (test&~nan)
   else:raise ValueError(name)
   typ=op.results[0].type
   env[op.results[0]]=np.asarray(v,dtype=np.float32 if isinstance(typ,Float32Type) else np.bool_ if isinstance(typ,IntegerType) and typ.bitwidth==1 else np.int32 if isinstance(typ,IntegerType) and typ.bitwidth==32 else np.int64 if isinstance(typ,IntegerType) else np.float64)
 return [env[r] for r in results]
