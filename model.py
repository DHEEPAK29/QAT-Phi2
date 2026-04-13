import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
from torchao.quantization import prepare_qat
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QATConfig:
    """Configuration for QAT training."""
    model_name = "microsoft/phi-2"
    group_size = 128   
    torch_dtype = torch.bfloat16
    device_map = "auto"   
    

def verify_quantization_compatibility(model, group_size):
    """  
    Args:
        model: The model to verify
        group_size: Quantization group size
        
    Raises:
        ValueError: If incompatible dimensions found
    """
    incompatible_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.shape[1] % group_size != 0:
                incompatible_layers.append(
                    f"{name}: in_features={module.weight.shape[1]}"
                )
    
    if incompatible_layers:
        raise ValueError(
            f"The following layers have incompatible dimensions for group_size={group_size}:\n"
            + "\n".join(incompatible_layers)
        )
    
    logger.info(f"✓ All linear layers compatible with group_size={group_size}")


def get_phi2_qat_model(config: QATConfig = None):
    """
    Load Phi-2 and prepare it for Quantization-Aware Training.
    
    Args:
        config: QATConfig instance. If None, uses default config.
        
    Returns:
        tuple: (qat_model, tokenizer)
        
    Raises:
        RuntimeError: If CUDA not available or model loading fails
    """
    if config is None:
        config = QATConfig()
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for QAT training. Please ensure GPU is available."
        )
    
    logger.info(f"Loading base model: {config.model_name}")
    
    try:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            trust_remote_code=True,  # Required for Phi-2
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
         
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
        
        # Verify compatibility before quantization
        verify_quantization_compatibility(model, config.group_size)
        
        # QAT quantization
        logger.info("Applying QAT quantization...")
        quantizer = Int8DynActInt4WeightQATQuantizer(
            group_size=config.group_size
        )
        qat_model = prepare_qat(model, quantizer)
        
        logger.info("✓ QAT preparation complete")
        
        return qat_model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def count_qat_modules(model):
    """
    Count the number of FakeQuantize modules inserted by QAT.
    
    Args:
        model: QAT-prepared model
        
    Returns:
        dict: Counts of different quantization module types
    """
    from collections import defaultdict
    counts = defaultdict(int)
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "Fake" in module_type or "Quant" in module_type:
            counts[module_type] += 1
    
    return dict(counts)


if __name__ == "__main__":
    # Test model loading
    model, tokenizer = get_phi2_qat_model()
    
    # Print QAT module statistics
    qat_stats = count_qat_modules(model)
    print("\n=== QAT Module Statistics ===")
    for module_type, count in qat_stats.items():
        print(f"{module_type}: {count}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✓ Forward pass successful. Logits shape: {outputs.logits.shape}")