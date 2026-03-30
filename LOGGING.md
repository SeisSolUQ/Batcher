# Batcher Logging

## Overview
The batcher includes simple logging to help debug crashes and track batch processing.

## Log File Location
Logs are written to `batcher.log` in the current working directory.

**Fallback behavior:** If the log file cannot be created (e.g., unwritable directory in containers), logs will be written to stderr instead.

**Log rotation:** Log files automatically rotate when they reach 10MB, keeping up to 3 backup files (batcher.log.1, batcher.log.2, batcher.log.3). This prevents unbounded disk usage in long-running services.

## Log Format
Each log entry follows this format:
```
YYYY-MM-DD HH:MM:SS | MESSAGE
```

## What Gets Logged

### 1. Request Received
Every time a request comes in (summary to reduce log volume):
```
Request received: config_order=3, num_parameters=1, parameter_lengths=[2]
```

### 2. Batch Submitted
When a batch is full or timeout is reached and sent to the simulator:
```
Batch submitted: batch_id=3_1774862845543, config_order=3, real_count=1, total_count=2
```
- `batch_id` uniquely identifies this batch instance
- `config_order` shows which configuration this batch belongs to
- `real_count` shows how many are real submissions (excluding padding)
- `total_count` shows total parameters including padding

### 3. Output Received
When the simulator returns output (summary only):
```
Output received: batch_id=3_1774862845543, config_order=3, output_length=2
```

Or if it fails:
```
Output FAILED: batch_id=3_1774862845543, config_order=3, error=Connection timeout
```

## Example Log

Here's a complete trace of one batch:
```
2026-03-30 11:02:20 | Request received: config_order=3, num_parameters=1, parameter_lengths=[2]
2026-03-30 11:02:20 | Batch submitted: batch_id=3_1774862845543, config_order=3, real_count=1, total_count=2
2026-03-30 11:02:20 | Output received: batch_id=3_1774862845543, config_order=3, output_length=2
```

## Performance Notes

Logs are kept concise to minimize performance impact:
- Only summaries are logged (sizes, counts, IDs) not full data payloads
- This keeps log volume manageable under high load
- Full parameter and output data is NOT logged to avoid performance degradation

## Debugging Tips

### View logs in real-time
```bash
tail -f batcher.log
```

### Find failed batches
```bash
grep "Output FAILED" batcher.log
```

### Count requests vs outputs
```bash
grep -c "Request received" batcher.log
grep -c "Output received" batcher.log
```

### Find a specific batch by ID
```bash
grep "batch_id=3_1774862845543" batcher.log
```

### Find batches with specific config
```bash
grep "config_order=3" batcher.log
```

That's it! Simple, efficient logging designed for production use.

