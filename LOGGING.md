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
YYYY-MM-DD HH:MM:SS | LEVEL | MESSAGE
```

Where LEVEL is INFO or ERROR.

## What Gets Logged

### 1. Request Received
Every time a request comes in:
```
2026-03-30 11:02:20 | INFO | Request received: config_order=3, num_parameters=1, parameter_lengths=[2], parameters=[[0.1, 0.2]]
```

### 2. Batch Submitted
When a batch is full or timeout is reached and sent to the simulator:
```
2026-03-30 11:02:20 | INFO | Batch submitted: batch_id=3_a1b2c3d4e5f6g7h8, config_order=3, real_count=1, total_count=2, parameters=[[0.1, 0.2], [0.1, 0.2]]
```
- `batch_id` uniquely identifies this batch instance for tracing
- `config_order` shows which configuration this batch belongs to
- `real_count` shows how many are real submissions (excluding padding)
- `total_count` shows total parameters including padding
- `parameters` shows all actual parameters sent to simulator

### 3. Output Received
When the simulator returns output:
```
2026-03-30 11:02:20 | INFO | Output received: batch_id=3_a1b2c3d4e5f6g7h8, config_order=3, output_length=2, parameters=[[0.1, 0.2], [0.1, 0.2]], output=[0.5, 0.5]
```
- Shows both the parameters that were sent and the output that was received

Or if it fails:
```
2026-03-30 11:02:25 | ERROR | Output FAILED: batch_id=3_a1b2c3d4e5f6g7h8, config_order=3, parameters=[[0.1, 0.2]], error=Connection timeout
```

## Example Log

Here's a complete trace of one batch:
```
2026-03-30 11:02:20 | INFO | Request received: config_order=3, num_parameters=1, parameter_lengths=[2], parameters=[[0.1, 0.2]]
2026-03-30 11:02:20 | INFO | Batch submitted: batch_id=3_a1b2c3d4e5f6g7h8, config_order=3, real_count=1, total_count=2, parameters=[[0.1, 0.2], [0.1, 0.2]]
2026-03-30 11:02:20 | INFO | Output received: batch_id=3_a1b2c3d4e5f6g7h8, config_order=3, output_length=2, parameters=[[0.1, 0.2], [0.1, 0.2]], output=[0.5, 0.5]
```

You can see:
- What parameters came in
- What batch was submitted (including padding)
- What output was received
- The batch_id ties everything together for tracing

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
grep "batch_id=3_a1b2c3d4e5f6g7h8" batcher.log
```

### Find batches with specific config
```bash
grep "config_order=3" batcher.log
```

That's it! Simple, efficient logging designed for production use.

