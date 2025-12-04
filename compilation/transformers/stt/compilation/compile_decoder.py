#!/usr/bin/env python3
"""
Compile Whisper Decoder to MXQ format
Separated from encoder compilation for efficiency
"""

import os
import torch
from transformers import AutoModelForSpeechSeq2Seq
from qubee import mblt_compile, mxq_compile, get_llm_config
import qubee
print(qubee.model_dict.parser.parser.__file__)
"""
IMPORTANT NOTE

1. After installing qubee==0.11.0.1, go to file printed above('<qubee_location>/model_dict/parser/parser.py').
2. In line 538, modify line as below:
    From 
        if self.backend == "hf":
    To
        if False:
"""

def compile_decoder(calib_path, output_dir="./compiled"):
    """Compile Whisper decoder to MXQ"""
    
    print("=" * 60)
    print("🎯 Compiling Whisper Decoder")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        'openai/whisper-small', torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = model.eval().cpu()
    
    mblt_path = os.path.join(output_dir, "whisper-small_decoder.mblt")
    mblt_compile(
        model=model,
        mblt_save_path=mblt_path,
        backend='hf',
        target='decoder',
        device='cpu'
    )
    
    # Compile to MXQ
    print("Compiling to MXQ...")
    mxq_path = os.path.join(output_dir, "whisper-small_decoder.mxq")
   
    try:
        llm_config=get_llm_config(use_full_seq_len_calib=True)
        mxq_compile(
            model=mblt_path,
            calib_data_path=calib_path,
            save_path=mxq_path,
            device="cuda",
            llm_config=llm_config,
        )
        
        print(f"✅ Decoder compiled: {mxq_path}")
        return mxq_path
        
    except Exception as e:
        print(f"❌ Decoder compilation failed: {e}")
        return None


def main():
    """Compile Whisper decoder"""
    
    print("🚀 Compiling Whisper Decoder to MXQ")
    print("=" * 70)
    
    # Check calibration data
    decoder_calib_path = "../calibration/decoder/whisper_decoder_calib.json"
    
    if not os.path.exists(decoder_calib_path):
        print(f"❌ Decoder calibration not found: {decoder_calib_path}")
        print("Please run create_calibration.py first!")
        return 1
    
    # Compile decoder
    decoder_mxq_path = compile_decoder(decoder_calib_path)
    
    print("\n" + "=" * 70)
    print("🎉 Decoder Compilation Complete!")
    print("=" * 70)
    
    if decoder_mxq_path:
        print(f"✅ Decoder: {decoder_mxq_path}")
    else:
        print("❌ Decoder compilation failed")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
