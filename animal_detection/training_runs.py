import traceback
import logging
from datetime import datetime
import time

from .yolo import train_yolo26s, train_yolo26m
from .rt_detr import train_rtdetr_v1_l

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_output.log"),  # All output
        logging.StreamHandler()                     # Print to console
    ]
)
error_logger = logging.getLogger('error_logger')
error_logger.addHandler(logging.FileHandler("training_errors.log"))
error_logger.setLevel(logging.ERROR)


def run_with_error_handling(func, func_name):
    """Run a function, handle errors, and log everything."""
    start_time = time.time()
    try:
        result = func()
        duration = time.time() - start_time
        logging.info(f"Function '{func_name}' succeeded in {duration:.2f} seconds.")
        return {"success": True, "duration": duration, "result": result}
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Function '{func_name}' failed after {duration:.2f} seconds: {str(e)}\n{traceback.format_exc()}"
        error_logger.error(error_msg)
        logging.error(f"Function '{func_name}' failed: {str(e)}")
        return {"success": False, "duration": duration, "error": str(e)}

def main():
    # Define your functions to run
    functions = [
        (train_yolo26s, "YOLO26s"),
        (train_yolo26m, "YOLO26m"),
        (train_rtdetr_v1_l, "RT-DETR V1 L"),
    ]

    results = []
    for func, name in functions:
        logging.info(f"\nStarting {name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = run_with_error_handling(func, name)
        results.append((name, result))

    # Print result table
    logging.info("\n" + "="*60)
    logging.info("TRAINING RESULTS SUMMARY")
    logging.info("="*60)
    logging.info(f"{'Function':<20} | {'Success':<8} | {'Duration (s)':<12} | {'Result/Error'}")
    logging.info("-"*60)
    for name, res in results:
        status = "Yes" if res["success"] else "No"
        duration = f"{res['duration']:.2f}"
        output = res.get("result", res.get("error", "N/A"))
        logging.info(f"{name:<20} | {status:<8} | {duration:<12} | {output}")
    logging.info("="*60)

if __name__ == "__main__":
    main()
