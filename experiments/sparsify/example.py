import time
import logging
import threading
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

MODEL = "HuggingFaceTB/SmolLM2-135M"
dataset = load_dataset(
    "EleutherAI/SmolLM2-135M-10B", split="train",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer)


gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_gpu_stats():
    """Get current GPU memory and utilization stats."""
    if not torch.cuda.is_available():
        return None
    
    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
        
        stats[f"gpu_{i}"] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        }
    
    return stats

def monitor_gpu(interval=10, stop_event=None):
    """Monitor GPU usage in a separate thread."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU monitoring")
        return
    
    while not stop_event.is_set():
        stats = get_gpu_stats()
        if stats:
            for gpu_id, gpu_stats in stats.items():
                logger.info(
                    f"{gpu_id}: "
                    f"Allocated: {gpu_stats['allocated_gb']:.2f} GB, "
                    f"Reserved: {gpu_stats['reserved_gb']:.2f} GB, "
                    f"Max Allocated: {gpu_stats['max_allocated_gb']:.2f} GB"
                )
        stop_event.wait(interval)

cfg = TrainConfig(SaeConfig(), batch_size=16)
trainer = Trainer(cfg, tokenized, gpt)

# Log initial GPU state
logger.info("=" * 60)
logger.info("Starting training")
logger.info("=" * 60)
if torch.cuda.is_available():
    initial_stats = get_gpu_stats()
    if initial_stats:
        for gpu_id, gpu_stats in initial_stats.items():
            logger.info(f"Initial {gpu_id} state: {gpu_stats['allocated_gb']:.2f} GB allocated")

# Start GPU monitoring thread
stop_monitoring = threading.Event()
monitor_thread = threading.Thread(target=monitor_gpu, args=(10, stop_monitoring), daemon=True)
monitor_thread.start()

# Track training time
start_time = time.time()
try:
    trainer.fit()
finally:
    end_time = time.time()
    stop_monitoring.set()
    
    # Log final stats
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.info("=" * 60)
    logger.info("Training completed")
    logger.info("=" * 60)
    logger.info(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time:.2f} seconds)")
    
    if torch.cuda.is_available():
        final_stats = get_gpu_stats()
        if final_stats:
            for gpu_id, gpu_stats in final_stats.items():
                logger.info(
                    f"Final {gpu_id} state: "
                    f"Allocated: {gpu_stats['allocated_gb']:.2f} GB, "
                    f"Reserved: {gpu_stats['reserved_gb']:.2f} GB, "
                    f"Max Allocated: {gpu_stats['max_allocated_gb']:.2f} GB"
                )
    
    logger.info(f"Log file saved to: {log_file}")