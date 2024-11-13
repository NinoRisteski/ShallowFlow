import torch

class Quantizer:
    def __init__(self, bits):
        self.bits = bits
        # Set dtype based on bit width
        self.dtype = torch.int16 if bits > 8 else torch.int8
        
    def quantize(self, x):
        # Calculate scale to use full range of bits
        q_max = 2**(self.bits-1) - 1  # Use signed range
        q_min = -(2**(self.bits-1))
        
        # Compute scale and zero_point
        x_max = x.max()
        x_min = x.min()
        scale = (x_max - x_min) / (q_max - q_min)
        zero_point = q_min - x_min / scale
        
        # Quantize
        x_q = torch.clamp(torch.round(x / scale + zero_point), q_min, q_max)
        x_q = x_q.to(self.dtype)
        
        return x_q, scale, zero_point

    def dequantize(self, x_q, scale, zero_point):
        return scale * (x_q.float() - zero_point) 
    
